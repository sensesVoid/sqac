#!/usr/bin/env python3
"""Full benchmark: skill store retrieval + application across 50+ problems.

    python tests/bench_skill_store.py              # retrieval only (fast)
    python tests/bench_skill_store.py --llm        # full application test

Design notes:
  - Application pass is batched (left-padded generate) for speed.
  - Baseline answers (no skill) are cached in --baseline-cache and reused
    across runs — they do not depend on retrieval.
  - With-skill answers are cached in --ckpt per experiment; use a fresh file
    whenever triggers/retrieval change.
  - --min-conf sets an injection floor: a top-1 hit below this confidence is
    NOT injected (bare question instead). Guards against misdirection from
    weak matches.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqac.skills import load_skills, pack_cartridge
from sqac.store import SqacStore

SKILL_PATHS = ["examples/logic_skills.yaml", "examples/coding_skills.yaml"]


def load_problems(path: str) -> list[dict]:
    try:
        import yaml
    except ImportError:
        raise ImportError("pip install pyyaml")
    raw = yaml.safe_load(open(path))
    return [p for p in raw if isinstance(p, dict) and "q" in p]


def load_all_skills() -> list[dict]:
    try:
        import yaml
    except ImportError:
        raise ImportError("pip install pyyaml")
    skills = []
    for sp in SKILL_PATHS:
        raw = yaml.safe_load(open(sp))
        for s in raw:
            if isinstance(s, dict) and "name" in s:
                skills.append(s)
    return skills


def build_store(skills: list[dict]) -> SqacStore:
    """Pack skills into a .sqac cartridge with semantic tier."""
    try:
        import yaml
    except ImportError:
        raise ImportError("pip install pyyaml")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(skills, f, default_flow_style=False, allow_unicode=True)
        tmp = f.name

    out = tempfile.mktemp(suffix=".sqac")
    skill_objs = load_skills(tmp)
    pack_cartridge(skill_objs, out, name="benchmark-skills", semantic=True)
    os.unlink(tmp)
    return SqacStore.load(out)


def retrieval_bench(problems: list[dict], store: SqacStore) -> dict:
    """Measure retrieval accuracy without LLM."""
    print("=== retrieval accuracy ===")
    results = {"total": 0, "correct": 0, "by_domain": {}, "by_difficulty": {}}
    for p in problems:
        want = p.get("skill", "")
        domain = p.get("domain", "unknown")
        diff = p.get("difficulty", "medium")
        hits = store.search(p["q"], top_k=3)
        got = ""
        for h in hits:
            if want in h.content:
                got = want
                break
        ok = got == want
        results["total"] += 1
        results["correct"] += int(ok)
        results["by_domain"].setdefault(domain, {"total": 0, "correct": 0})
        results["by_domain"][domain]["total"] += 1
        results["by_domain"][domain]["correct"] += int(ok)
        results["by_difficulty"].setdefault(diff, {"total": 0, "correct": 0})
        results["by_difficulty"][diff]["total"] += 1
        results["by_difficulty"][diff]["correct"] += int(ok)

    print(f"\noverall: {results['correct']}/{results['total']}")
    for domain, stats in results["by_domain"].items():
        print(f"  {domain:12}: {stats['correct']}/{stats['total']}")
    for diff, stats in results["by_difficulty"].items():
        print(f"  {diff:12}: {stats['correct']}/{stats['total']}")
    return results


def build_prompt(tok, problem: dict, skill_content: str | None) -> str:
    is_coding = problem.get("domain") in ("coding", "refactoring", "architecture", "testing")
    if is_coding:
        sysmsg = "You are a software engineer. Explain your approach concisely in 2-3 sentences."
    else:
        sysmsg = "You are a problem solver. Show brief reasoning, then give the answer."
    if skill_content:
        user = f"Here is a useful technique:\n{skill_content}\n\nQuestion: {problem['q']}"
    else:
        user = f"Question: {problem['q']}"
    msgs = [{"role": "system", "content": sysmsg}, {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def grade(answer: str, problem: dict) -> bool:
    check = problem.get("check", [])
    if isinstance(check, str):
        check = [check]
    answer_lower = answer.lower()
    return any(kw.lower() in answer_lower for kw in check)


def _load_jsonl(path: str | None) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if path and os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    out[rec["q"]] = rec
    return out


def _batched_generate(model, tok, prompts: list[str], max_new: int, batch_size: int) -> list[str]:
    import torch

    texts: list[str] = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        inputs = tok(chunk, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        texts.extend(tok.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                      skip_special_tokens=True))
    return texts


def application_bench(problems: list[dict], store: SqacStore, model_name: str,
                      ckpt_path: str | None = None, baseline_cache: str | None = None,
                      gen_max_new: int = 70, batch_size: int = 10,
                      min_conf: float = 0.0) -> dict:
    """Measure application accuracy: with skill vs baseline.

    Baseline answers are cached persistently (independent of retrieval);
    with-skill answers are cached per experiment in ckpt_path.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(os.cpu_count() or 4)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # required for batched decoder-only generation
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    model.eval()

    # ── retrieval + injection decision ───────────────────────────────────
    injected: dict[str, str | None] = {}
    n_suppressed = 0
    for p in problems:
        hits = store.search(p["q"], top_k=1)
        if hits and hits[0].confidence >= min_conf:
            injected[p["q"]] = hits[0].content
        else:
            injected[p["q"]] = None
            n_suppressed += 1

    # ── baseline answers (persistent cache — retrieval-independent) ──────
    base_done = _load_jsonl(baseline_cache)
    missing_base = [p for p in problems if p["q"] not in base_done]
    if missing_base:
        print(f"  baseline: {len(missing_base)} to generate")
        prompts = [build_prompt(tok, p, None) for p in missing_base]
        t0 = time.time()
        texts = _batched_generate(model, tok, prompts, gen_max_new, batch_size)
        with open(baseline_cache, "a") as f:
            for p, text in zip(missing_base, texts):
                f.write(json.dumps({"q": p["q"], "ans_without": text}) + "\n")
                base_done[p["q"]] = {"q": p["q"], "ans_without": text}
        print(f"  baseline done ({time.time() - t0:.0f}s)")
    else:
        print(f"  baseline: all {len(problems)} cached")

    # ── with-skill answers (per-experiment cache) ────────────────────────
    with_done = _load_jsonl(ckpt_path)
    missing_with = [p for p in problems if p["q"] not in with_done]
    if missing_with:
        print(f"  with-skill: {len(missing_with)} to generate")
        prompts = [build_prompt(tok, p, injected[p["q"]]) for p in missing_with]
        t0 = time.time()
        texts = _batched_generate(model, tok, prompts, gen_max_new, batch_size)
        if ckpt_path:
            with open(ckpt_path, "a") as f:
                for p, text in zip(missing_with, texts):
                    rec = {"q": p["q"], "ans_with": text}
                    f.write(json.dumps(rec) + "\n")
                    with_done[p["q"]] = rec
        else:
            for p, text in zip(missing_with, texts):
                with_done[p["q"]] = {"q": p["q"], "ans_with": text}
        print(f"  with-skill done ({time.time() - t0:.0f}s)")
    else:
        print(f"  with-skill: all {len(problems)} cached")

    # ── grade ────────────────────────────────────────────────────────────
    results = {"with": 0, "without": 0, "n": 0, "by_domain": {}, "by_difficulty": {}}
    for p in problems:
        b = base_done.get(p["q"])
        w = with_done.get(p["q"])
        if not b or not w or "ans_with" not in w:
            continue
        ok_with = grade(w["ans_with"], p)
        ok_without = grade(b["ans_without"], p)
        domain = p.get("domain", "unknown")
        diff = p.get("difficulty", "medium")
        results["n"] += 1
        results["with"] += int(ok_with)
        results["without"] += int(ok_without)
        for key, val in (("by_domain", domain), ("by_difficulty", diff)):
            results[key].setdefault(val, {"with": 0, "without": 0, "total": 0})
            results[key][val]["with"] += int(ok_with)
            results[key][val]["without"] += int(ok_without)
            results[key][val]["total"] += 1

    n = results["n"]
    print(f"\ninjection: {n - n_suppressed} skills injected, {n_suppressed} suppressed (min_conf={min_conf})")
    print(f"overall (n={n}): with skills {results['with']}/{n} | baseline {results['without']}/{n}")
    for domain, stats in results["by_domain"].items():
        print(f"  {domain:12}: with {stats['with']}/{stats['total']} | baseline {stats['without']}/{stats['total']}")
    for diff, stats in results["by_difficulty"].items():
        print(f"  {diff:12}: with {stats['with']}/{stats['total']} | baseline {stats['without']}/{stats['total']}")
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--problems", default="examples/problem_set.yaml")
    ap.add_argument("--ckpt", default=None, help="per-experiment with-skill cache (JSONL)")
    ap.add_argument("--baseline-cache", default=".bench_baseline.jsonl",
                    help="persistent baseline answer cache (JSONL)")
    ap.add_argument("--max-new", type=int, default=70)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--min-conf", type=float, default=0.0,
                    help="injection floor; hits below this confidence are not injected")
    args = ap.parse_args()

    skills = load_all_skills()
    problems = load_problems(args.problems)
    print(f"loaded {len(skills)} skills, {len(problems)} problems\n")

    store = build_store(skills)

    ret = retrieval_bench(problems, store)
    if args.llm:
        application_bench(problems, store, args.model,
                          ckpt_path=args.ckpt, baseline_cache=args.baseline_cache,
                          gen_max_new=args.max_new, batch_size=args.batch_size,
                          min_conf=args.min_conf)
    return 0 if ret["correct"] == ret["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
