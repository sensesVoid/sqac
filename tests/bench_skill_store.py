#!/usr/bin/env python3
"""Full benchmark: skill store retrieval + application across 50+ problems.

    python tests/bench_skill_store.py              # retrieval only (fast)
    python tests/bench_skill_store.py --llm        # full application test

Application pass is batched (left-padded generate) and checkpointed to a JSONL
file (--ckpt), so a killed run resumes instead of losing progress.
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


def application_bench(problems: list[dict], store: SqacStore, model_name: str,
                      ckpt_path: str | None = None, gen_max_new: int = 70,
                      batch_size: int = 10) -> dict:
    """Measure application accuracy: with skill vs baseline. Batched + resumable."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(os.cpu_count() or 4)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # required for batched decoder-only generation
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    model.eval()

    # ── load prior progress ──────────────────────────────────────────────
    done: dict[str, dict] = {}
    if ckpt_path and os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    done[rec["q"]] = rec
    todo = [p for p in problems if p["q"] not in done]
    if len(done):
        print(f"  resuming: {len(done)} cached, {len(todo)} to run")

    # ── retrieval for new problems ───────────────────────────────────────
    retrievals: dict[str, str | None] = {}
    for p in todo:
        hits = store.search(p["q"], top_k=1)
        retrievals[p["q"]] = hits[0].content if hits else None

    # ── build all jobs: two variants per problem ─────────────────────────
    jobs: list[tuple[dict, str, str]] = []
    for p in todo:
        jobs.append((p, "with", build_prompt(tok, p, retrievals[p["q"]])))
        jobs.append((p, "without", build_prompt(tok, p, None)))

    answers: dict[str, dict] = {}
    ckpt_f = open(ckpt_path, "a") if ckpt_path else None
    t0 = time.time()
    n_batches = (len(jobs) + batch_size - 1) // batch_size
    for bi, start in enumerate(range(0, len(jobs), batch_size)):
        chunk = jobs[start:start + batch_size]
        prompts = [j[2] for j in chunk]
        inputs = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=gen_max_new,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                 skip_special_tokens=True)
        for (p, variant, _), text in zip(chunk, texts):
            answers.setdefault(p["q"], {})[f"ans_{variant}"] = text
            if ckpt_f:
                rec = answers[p["q"]]
                # only persist complete pairs
                if "ans_with" in rec and "ans_without" in rec:
                    ckpt_f.write(json.dumps({"q": p["q"], **rec}) + "\n")
                    ckpt_f.flush()
        print(f"  batch {bi + 1}/{n_batches} done ({time.time() - t0:.0f}s)")

    if ckpt_f:
        ckpt_f.close()

    # ── grade everything (cached + fresh) ────────────────────────────────
    results = {"with": 0, "without": 0, "n": 0, "by_domain": {}, "by_difficulty": {}}
    for p in problems:
        rec = answers.get(p["q"]) or done.get(p["q"])
        if rec is None or "ans_with" not in rec or "ans_without" not in rec:
            continue
        ok_with = grade(rec["ans_with"], p)
        ok_without = grade(rec["ans_without"], p)
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
    print(f"\noverall (n={n}): with skills {results['with']}/{n} | baseline {results['without']}/{n}")
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
    ap.add_argument("--ckpt", default=None, help="JSONL checkpoint file (resumable)")
    ap.add_argument("--max-new", type=int, default=70)
    ap.add_argument("--batch-size", type=int, default=10)
    args = ap.parse_args()

    skills = load_all_skills()
    problems = load_problems(args.problems)
    print(f"loaded {len(skills)} skills, {len(problems)} problems\n")

    store = build_store(skills)

    ret = retrieval_bench(problems, store)
    if args.llm:
        application_bench(problems, store, args.model,
                          ckpt_path=args.ckpt, gen_max_new=args.max_new,
                          batch_size=args.batch_size)
    return 0 if ret["correct"] == ret["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
