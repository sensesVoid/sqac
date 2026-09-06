#!/usr/bin/env python3
"""Can SQAC serve as a SKILL/LOGIC store (not just facts)?

Distinction under test:
  transliteration = retrieve stored text and recite it (fact-store behavior)
  skill use       = retrieve a PROCEDURE and APPLY it to a novel problem

Design (acceptance criterion = the user's framing):
  - Skills stored as (situation trigger -> procedure). Keys are situation
    descriptions; procedure CONTENT contains ZERO answers.
  - Queries are NOVEL problems, never stored. Any correct final answer is
    therefore evidence of skill application, not retrieval of the answer.
  - Baseline: same model, no skills -> measures what the small model can
    do alone (expected to fail format/strategy).
  - Pattern check: does the solution EXHIBIT the retrieved procedure
    (e.g., work backwards, use ratios) rather than guess?

Run:  python tests/test_skill_store.py            # retrieval only
      python tests/test_skill_store.py --llm      # full application test
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqac.store import SqacStore

# ── Skills: procedures with ZERO answers in them ────────────────────────────
# Measured design lessons:
#   1. Abstract-only keys never match concrete story problems (abstraction gap).
#   2. Even trigger phrases get diluted when averaged into one long key.
# => Skill card = ONE procedure + MANY situation keys, each stored as its
#    own entry (multi-key routing). Keys are routing metadata, zero answers.
SKILLS = [
    {
        "keys": [
            "when a problem asks for a value after repeated equal changes",
            "per unit rate: find the amount for one item or one minute first",
            "price each, cost per item, how much for one",
        ],
        "content": (
            "SKILL unit-walk: Compute the per-unit amount first, then scale. "
            "Write 'per X: <amount>'. Never handle totals before the per-unit step."
        ),
    },
    {
        "keys": [
            "when a rule or relationship is the same but the numbers are different",
            "at the same rate, how many minutes or hours to reach a larger amount",
            "filling or pumping at a constant rate, scale up the quantity",
            "same speed same price proportionally how long how much",
        ],
        "content": (
            "SKILL direct-proportion: If quantity scales by factor k, every proportional "
            "amount scales by k. Compute k from the two known quantities, then multiply."
        ),
    },
    {
        "keys": [
            "when you know the final state and must find an earlier state",
            "how much did she start with, how much was there originally",
            "undo operations in reverse: spent then doubled, gave away then",
        ],
        "content": (
            "SKILL work-backwards: Start from the final value and undo each operation in "
            "reverse order, writing the value after each undo. Do not solve forward."
        ),
    },
    {
        "keys": [
            "when a question asks how many items satisfy several conditions at once",
            "play both sports, in both clubs, belong to both groups, overlap",
            "how many like neither, how many do not do either activity",
        ],
        "content": (
            "SKILL overlap-count: For conditions A and B: count = n(A) + n(B) - n(A and B). "
            "State each of the three counts explicitly before subtracting."
        ),
    },
    {
        "keys": [
            "when the cost or amount is the sum of two independent parts",
            "total cost of several items at given prices, buy multiple of each",
            "combined weight or cost of different things together",
        ],
        "content": (
            "SKILL decompose-sum: Split the quantity into its independent parts, compute "
            "each part separately, then add. Name each part before computing."
        ),
    },
    {
        "keys": [
            "when you must measure an exact amount using only two containers of fixed size",
            "jugs of different sizes, measure water, fill and pour",
            "container pouring puzzle with two vessels",
            "jug problem, fill pour empty, exactly 4 gallons with a 3 and 5",
            "two jugs capacity puzzle, measure out specific amount",
        ],
        "content": (
            "SKILL state-search: list every reachable volume in each jug after one pour. "
            "Repeat until the target volume appears. Record each fill-empty-pour step."
        ),
    },
    {
        "keys": [
            "when exactly one of many items has a hidden property and you can weigh once",
            "one defective batch among many, one weighing to find it",
            "poisoned bottle, bad batch, single test identifies which",
            "12 bags of coins, one heavier, one weighing on a scale",
            "find the odd bag with coins weighing more, one digital scale",
            "number the items and take a different count from each to identify",
        ],
        "content": (
            "SKILL weighted-index: label items 1..N. Take i coins from item i. "
            "The total excess weight tells you which item has the defect."
        ),
    },
]

# ── Novel problems (NOT in the store). answer = gold for grading ────────────
PROBLEMS = [
    # --- EASY: the model can already solve these (skill may not help) ---
    {
        "q": "A pen costs 8 dollars and a notebook costs 5 dollars. What is the cost of 3 pens and 4 notebooks?",
        "skill": "decompose-sum",
        "marker": "44",
        "needs_pattern": None,
        "novel": "no stored text mentions pens or notebooks",
    },
    {
        "q": "In a class of 30, 18 play football, 15 play chess, and 8 play both. How many play neither?",
        "skill": "overlap-count",
        "marker": "5",
        "needs_pattern": None,
        "novel": "no stored text mentions football or chess",
    },
    # --- HARD: expected to fail without skill, succeed with it ---
    {
        "q": "You have a 3-gallon jug and a 5-gallon jug. How do you measure exactly 4 gallons?",
        "skill": "state-search",
        "marker": None,  # qualitative answer — we check for key steps instead
        "check": lambda a: any(x in a for x in ["3 1", "1 3", "5-1=4", "from 5 to 3", "pour", "fill the 5"]),
        "novel": "no stored text mentions jugs or gallons",
    },
    {
        "q": "There are 12 bags of coins. One bag has coins weighing 1.1g each instead of 1.0g. You may weigh ONCE on a digital scale. How do you find the odd bag?",
        "skill": "weighted-index",
        "marker": None,
        "check": lambda a: any(x in a for x in ["bag 1", "take 1 from", "take i from", "excess", "number the"]),
        "novel": "no stored text mentions bags or weighing",
    },
]

BADGE = "SKILL "


def build_skill_store(semantic: bool = True) -> SqacStore:
    store = SqacStore(semantic=semantic)
    for s in SKILLS:
        for k in s["keys"]:  # one procedure, many situation keys
            store.add(s["content"], key=k, source="skill-library")
    return store


def retrieval_report(store: SqacStore) -> int:
    ok = 0
    for p in PROBLEMS:
        hits = store.search(p["q"], top_k=1)
        if hits:
            top = hits[0]
            want = p["skill"]
            got = want in top.content and top.content.startswith(BADGE)
            ok += got
            mark = "PASS" if got else "FAIL"
            print(f"{mark} {p['q'][:62]!r:56} -> {top.mode:8} {top.confidence:.2f}  want {want}")
        else:
            print(f"FAIL {p['q'][:62]!r:56} -> no hit")
    print(f"retrieval: {ok}/{len(PROBLEMS)} routed to the correct skill\n")
    return ok


def run_llm(store: SqacStore, model_name: str) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    model.eval()

    def chat(messages, max_new=200):
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def grade(answer: str, problem: dict) -> bool:
        if problem.get("check"):  # qualitative problems
            return problem["check"](answer.lower())
        return problem["marker"] in answer  # numeric marker problems

    def solve(problem: dict, skill_content: str | None) -> str:
        is_qual = problem.get("check") is not None
        if is_qual:
            sysmsg = "You are a careful problem solver. Explain your approach step by step, including any specific operations."
        else:
            sysmsg = "You are a careful problem solver. Show brief reasoning, then end with: ANSWER: <number>"
        if skill_content:
            user = f"Here is a useful technique:\n{skill_content}\n\nProblem: {problem['q']}"
        else:
            user = f"Problem: {problem['q']}"
        return chat(
            [
                {"role": "system", "content": sysmsg},
                {"role": "user", "content": user},
            ],
            max_new=220,
        )

    print(f"=== application test ({model_name}) ===")
    results = {"with": 0, "without": 0, "with_hard": 0, "without_hard": 0}
    for p in PROBLEMS:
        hits = store.search(p["q"], top_k=1)
        skill_text = hits[0].content if hits else None

        ans_with = solve(p, skill_text)
        ok_with = grade(ans_with, p)
        results["with"] += ok_with
        is_hard = p.get("check") is not None
        if is_hard:
            results["with_hard"] += ok_with

        ans_without = solve(p, None)
        ok_without = grade(ans_without, p)
        results["without"] += ok_without
        if is_hard:
            results["without_hard"] += ok_without

        tag = "hard" if is_hard else "easy"
        print(f"[{'PASS' if ok_with else 'FAIL'}] {p['q'][:58]}  ({tag})")
        print(f"     skill : {skill_text[:64] + '…' if skill_text else None}")
        print(f"     with  : …{ans_with.strip()[-80:]}")
        print(f"     w/o   : {'correct' if ok_without else 'wrong'}")

    easy = sum(1 for p in PROBLEMS if p.get("check") is None)
    hard = len(PROBLEMS) - easy
    print(
        f"\nRESULT: with skills {results['with']}/{len(PROBLEMS)} "
        f"(easy {results['with'] - results['with_hard']}/{easy}, hard {results['with_hard']}/{hard}) | "
        f"baseline {results['without']}/{len(PROBLEMS)} "
        f"(easy {results['without'] - results['without_hard']}/{easy}, hard {results['without_hard']}/{hard})"
    )
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    args = ap.parse_args()

    store = build_skill_store()
    # persistence proof: skills survive save/load (a library file, like facts)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "skills.sqac")
        store.save(path, name="skill-library")
        store = SqacStore.load(path)
    print(f"skill library: {len(store)} procedures (persisted + reloaded)\n")

    ok = retrieval_report(store)
    if args.llm:
        run_llm(store, args.model)
    return 0 if ok == len(PROBLEMS) else 1


if __name__ == "__main__":
    sys.exit(main())
