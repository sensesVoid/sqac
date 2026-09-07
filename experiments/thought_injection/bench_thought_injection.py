"""Thought-injection benchmark v2 — does a stored SQAC procedure, injected
into a small instruct model, give it a reliable way to reason?

Arms (per task), same model:
  A) baseline       - direct answer, no scaffold
  B) free CoT       - "Let's think step by step." (model invents its own path)
  C) verbose inject - full verbose procedure + "follow every step" instruction
  D) concise inject - terse procedure + "apply silently, ANSWER = <number>"
                       (budget-forced style: shorter output ceiling)
  G) policy         - operator cartridge + plan(): inject the routed single-pair
                       skill as one arithmetic expression only when the router is
                       confident; deterministic direction pin; else answer direct

Tests the researched hypothesis that minimal-verbosity skill injection
preserves weak-model accuracy while still steering the procedure.

Usage:
    python bench_thought_injection.py --model allam-2-7b --per-family 10
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine import (
    ThoughtSkillEngine,
    complete,
    eval_expression,
    extract_direction_card,
    load_api_key,
)

ARMS = ("A", "B", "C", "D", "E", "F", "G")
_STYLE = {
    "A": "direct",
    "B": "cot",
    "C": "injected",
    "D": "injected_concise",
    "E": "expression",
    "F": "cod",
}
_RUN_ON_CONCISE = {"D", "E", "F"}
# G asks its own engine for a plan (operator cartridge), not a fixed style.

_ANSWER_RE = re.compile(r"ANSWER\s*[:=]\s*(-?\d+(?:\.\d+)?)")
_LAST_NUM_RE = re.compile(r"(-?\d+(?:\.\d+)?)(?!.*\d)")


def extract_answer(text: str) -> float | None:
    m = list(_ANSWER_RE.finditer(text))
    if m:
        return float(m[-1].group(1))
    m = list(_LAST_NUM_RE.finditer(text))
    if m:
        return float(m[-1].group(1))
    return None


def score(pred: float | None, truth: float, tol: float = 0.02) -> bool:
    if pred is None:
        return False
    if abs(pred - truth) <= 1e-4:
        return True
    return abs(pred - truth) <= tol * abs(truth)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allam-2-7b")
    ap.add_argument("--skills", default=str(Path(__file__).parent / "think_skills.yaml"))
    ap.add_argument("--skills-d", default=str(Path(__file__).parent / "think_skills_concise.yaml"))
    ap.add_argument("--skills-op", default=str(Path(__file__).parent / "think_skills_operator.yaml"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--per-family", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--max-new-d", type=int, default=220)
    ap.add_argument("--max-new-e", type=int, default=120)
    ap.add_argument("--max-new-f", type=int, default=220)
    ap.add_argument("--max-new-g", type=int, default=120)
    ap.add_argument("--env-file", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    key = load_api_key(env_file=args.env_file or None)
    eng_v = ThoughtSkillEngine(args.skills)
    eng_c = ThoughtSkillEngine(args.skills_d)
    eng_op = ThoughtSkillEngine(args.skills_op, min_confidence=0.6)
    print(f"cartridges: verbose={len(eng_v.store)}|concise={len(eng_c.store)} "
          f"all {len(eng_v.skills)} skills")

    from tasks import generate
    tasks = generate(seed=args.seed, per_family=args.per_family)
    if args.limit:
        tasks = tasks[: args.limit]

    out = args.out or f"results_{re.sub(r'[^\\w]+', '_', args.model)}.json"
    records_by_q = {}
    prior_arms = set()
    if Path(out).exists():
        try:
            for r in json.loads(Path(out).read_text()).get("records", []):
                records_by_q[r["question"]] = r
                prior_arms |= {a for a in ARMS if a in r}
        except Exception:
            pass
    missing = [a for a in ARMS if a not in prior_arms]
    print(f"resume: {len(records_by_q)} prior rows, filling arms {missing or 'none'}")

    completed = 0
    records = []
    for i, t in enumerate(tasks):
        row = records_by_q.pop(t.question, None)
        if row is None or not all(a in row for a in ARMS):
            rv = eng_v.route(t.question)
            rc = eng_c.route(t.question)
            if rv is None or rc is None:
                print(f"!! {t.question!r} routed to nothing")
                continue
        if row is None:
            row = {
                "family": t.family,
                "skill": t.skill,
                "question": t.question,
                "truth": t.answer,
                "retrieval": {
                    "verbose": {
                        "mode": rv.mode,
                        "confidence": round(rv.confidence, 3),
                        "routed_skill": rv.skill,
                    },
                    "concise": {
                        "mode": rc.mode,
                        "confidence": round(rc.confidence, 3),
                        "routed_skill": rc.skill,
                    },
                },
            }
        if all(a in row for a in ARMS):
            records.append(row)
            continue
        for name in ARMS:
            if name in row:
                continue
            if name == "G":
                plan = eng_op.plan(t.question, "expression")
                msgs = eng_op.prompt(t.question, plan.style, plan.routed)
                r = complete(key, args.model, msgs, max_tokens=args.max_new_g)
                pred = (
                    eval_expression(r["text"])
                    if plan.style == "expression"
                    else extract_answer(r["text"])
                )
                row[name] = {
                    "output": r["text"],
                    "reasoning": r["reasoning"],
                    "pred": pred,
                    "correct": score(pred, t.answer),
                    "prompt_tokens": r["prompt_tokens"],
                    "output_tokens": r["output_tokens"],
                    "latency_s": r["latency_s"],
                    "plan": {
                        "action": plan.action,
                        "style": plan.style,
                        "skill": plan.routed.skill if plan.routed else None,
                        "conf": round(plan.routed.confidence, 3) if plan.routed else None,
                        "reason": plan.reason,
                    },
                }
                continue
            routed = rc if name in _RUN_ON_CONCISE else rv
            msgs = eng_c.prompt(t.question, _STYLE[name], routed)
            cap = {"E": args.max_new_e, "F": args.max_new_f, "D": args.max_new_d}.get(
                name, args.max_new
            )
            r = complete(key, args.model, msgs, max_tokens=cap)
            if name == "E":
                pred = eval_expression(r["text"])
            else:
                pred = extract_answer(r["text"])
            row[name] = {
                "output": r["text"],
                "reasoning": r["reasoning"],
                "pred": pred,
                "correct": score(pred, t.answer),
                "prompt_tokens": r["prompt_tokens"],
                "output_tokens": r["output_tokens"],
                "latency_s": r["latency_s"],
            }
        records.append(row)
        completed += 1
        marks = "".join("✓" if row[k]["correct"] else "✗" for k in ARMS)
        print(
            f"[{i+1}/{len(tasks)}] {t.skill:20s} x={t.answer:<7.2f}  "
            + " ".join(f"{k}{marks[j]}" for j, k in enumerate(ARMS))
            + f"  (route {rv.skill[:14]} {rv.mode}/{rv.confidence:.2f})"
        )

    merged = records
    n = len(merged)
    if n == 0:
        print("no results")
        return 0

    print("\n== RESULTS (Groq / %s) ==" % args.model)
    tok = lambda arm: statistics.mean(r[arm]["output_tokens"] for r in merged)
    acc = {k: sum(1 for r in merged if r[k]["correct"]) / n for k in ARMS}
    out_tok = {k: tok(k) for k in ARMS}
    in_tok = {k: statistics.mean(r[k]["prompt_tokens"] for r in merged) for k in ARMS}
    for e in ("verbose", "concise"):
        ok = sum(1 for r in merged if r["retrieval"][e]["routed_skill"] == r["skill"])
        print(f"retrieval ({e}): {ok}/{n} ({100 * ok / n:.0f}%)")
    ok = 0
    confs = []
    for r in merged:
        exp = extract_direction_card(r["question"]) or "rate-times-time"
        if r["G"]["plan"]["skill"] == exp:
            ok += 1
        if r["G"]["plan"].get("conf") is not None:
            confs.append(r["G"]["plan"]["conf"])
    inj = sum(1 for r in merged if r["G"]["plan"]["action"] == "inject")
    print(f"policy (operator): {ok}/{n} exact card ({100 * ok / n:.0f}%), "
          f"{inj}/{n} injected ({100 * inj / n:.0f}%), "
          f"mean conf {statistics.mean(confs):.2f}")

    print(f"{'arm':<4} {'acc':<7} {'prompt':<8} {'out tok':<9} {'total':<8}")
    for k in ARMS:
        print(f"{k:<4} {acc[k]:<7.0%} {in_tok[k]:<8.0f} {out_tok[k]:<9.0f} "
              f"{in_tok[k] + out_tok[k]:<8.0f}")
    for lhs, rhs in (("C", "B"), ("D", "B"), ("E", "B"), ("E", "D"), ("F", "D"),
                     ("G", "A"), ("G", "E")):
        print(f"{lhs} vs {rhs}: acc {acc[lhs] - acc[rhs]:+.0%}, "
              f"out tok {out_tok[lhs] - out_tok[rhs]:+.1f}, "
              f"total tok {in_tok[lhs] + out_tok[lhs] - in_tok[rhs] - out_tok[rhs]:+.1f}")

    fams = sorted({r["family"] for r in merged})
    print("\nper-family accuracy:")
    for f in fams:
        sub = [r for r in merged if r["family"] == f]
        line = f"  {f:<12}"
        for k in ARMS:
            line += f"  {k}={sum(1 for r in sub if r[k]['correct'])/len(sub):.0%}"
        print(line)

    print("\nE-arm output samples:")
    for r in merged[:6]:
        print(f"  {r['question']}")
        print(f"    -> {r['E']['output'][:120]!r}  pred={r['E']['pred']}")
    print("\nG-arm output samples (plan -> output):")
    for r in merged[:6]:
        p = r["G"]["plan"]
        print(f"  {r['question']}")
        print(f"    {p['action']}/{p['skill']} ({p['reason']}) -> "
              f"{r['G']['output'][:90]!r} pred={r['G']['pred']}")

    with open(out, "w") as fh:
        json.dump({"model": args.model, "records": merged}, fh, indent=2)
    print(f"\nsaved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())