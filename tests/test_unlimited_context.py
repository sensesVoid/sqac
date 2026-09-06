#!/usr/bin/env python3
"""Is 'unlimited context' legit or overclaim? — SQAC stress test.

Decomposes the claim:
  S1 "unlimited storage, constant cost": recall must stay FLAT as N grows
     (multi-vector registry = independent traces, no superposition noise)
  S2 "unlimited context": the model must be able to USE what's stored
     (multi-needle aggregation, multi-hop chains, distractor noise,
     stale-fact override)

Run:
  python tests/test_unlimited_context.py            # retrieval-level (fast)
  python tests/test_unlimited_context.py --llm      # + end-to-end with Qwen
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqac.store import SqacStore

WORDS = (
    "auth deploy repository middleware scope image kebab config payment service "
    "schema migration rollback cache queue worker metric trace log alert policy "
    "quota rate limit token secret vault rotate audit access role owner team "
    "pipeline staging prod canary feature flag experiment cohort segment funnel "
    "rust python golang kernel parser compiler runtime memory thread signal "
    "packet socket retry backoff ledger invoice refund dispute chargeback"
).split()


def make_entity(i: int) -> tuple[str, str, str]:
    """Synthetic distinct entity: key, fact, answer token."""
    w1 = WORDS[i % len(WORDS)]
    w2 = WORDS[(i * 7 + 3) % len(WORDS)]
    w3 = WORDS[(i * 13 + 1) % len(WORDS)]
    code = f"QX{i:05d}"
    key = f"{w1} {w2} {w3} spec {code}"
    fact = f"Procedure {code}: the {w1} {w2} {w3} budget is {i % 97} units and owner is {w1.upper()}-{i}."
    return key, fact, code


# ─────────────────────────────────────────────────────────────────────────────
# S1: does recall stay flat as the store grows? (storage claim)
# ─────────────────────────────────────────────────────────────────────────────

def test_recall_vs_n(ns=(100, 1_000, 5_000, 10_000, 25_000), probes=200, seed=7):
    rng = random.Random(seed)
    print("=== S1: recall vs store size (constant-cost claim) ===")
    max_n = max(ns)
    # build the full store once, probe at each size tier
    store = SqacStore()
    entities = [make_entity(i) for i in range(max_n)]
    t0 = time.perf_counter()
    for key, fact, _ in entities:
        store.add(fact, key=key)
    build_s = time.perf_counter() - t0

    print(f"built {max_n} rules in {build_s:.1f}s")
    print(f"{'N':>7} | {'exact':>7} | {'fuzzy(recall@1)':>15} | {'fuzzy ms':>9}")
    print("-" * 50)
    for n in ns:
        # exact probes on random subset
        ex_ok = 0
        ex_keys = rng.sample(range(n), min(probes, n))
        t0 = time.perf_counter()
        for i in ex_keys:
            key, _, _ = entities[i]
            hits = store.search(key, top_k=1)
            if hits and hits[0].mode == "exact":
                ex_ok += 1
        exact_ms = (time.perf_counter() - t0) * 1000 / len(ex_keys)

        # fuzzy probes: paraphrase = first half of the fact text
        fz_ok = 0
        fz_idx = rng.sample(range(n), min(probes, n))
        for i in fz_idx:
            _, fact, _ = entities[i]
            hits = store.search(fact[: max(len(fact) // 2, 20)], top_k=1)
            if hits and hits[0].mode == "fuzzy" and hits[0].content == fact:
                fz_ok += 1
        t0 = time.perf_counter()
        for i in fz_idx[:50]:
            _, fact, _ = entities[i]
            store.search(fact[: max(len(fact) // 2, 20)], top_k=1)
        fuzzy_ms = (time.perf_counter() - t0) * 1000 / min(50, len(fz_idx))

        print(f"{n:>7} | {ex_ok / len(ex_keys):>7.1%} | {fz_ok / len(fz_idx):>15.1%} | {fuzzy_ms:>8.1f}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# S2a: multi-needle aggregation — the model must SUM across many memories
# ─────────────────────────────────────────────────────────────────────────────

TEAM_A, TEAM_B, TEAM_C = "atlas", "borealis", "cobalt"

SERVICES = [
    # (query-ish key, fact, team)
    ("auth service", "The auth service is owned by Team Atlas.", TEAM_A),
    ("token issuer", "The token issuer is owned by Team Atlas.", TEAM_A),
    ("login portal", "The login portal is owned by Team Atlas.", TEAM_A),
    ("search index", "The search index is owned by Team Borealis.", TEAM_B),
    ("crawler", "The crawler is owned by Team Borealis.", TEAM_B),
    ("ranker", "The ranker is owned by Team Borealis.", TEAM_B),
    ("billing api", "The billing api is owned by Team Cobalt.", TEAM_C),
    ("invoice worker", "The invoice worker is owned by Team Cobalt.", TEAM_C),
    ("refund engine", "The refund engine is owned by Team Cobalt.", TEAM_C),
]


def s2a_retrieval(store) -> dict:
    """Retrieval half: can a non-key query surface the whole team in top-5?
    (Using the exact key would short-circuit and tell us nothing about
    aggregation.)"""
    out = {}
    for team, keys in (
        (TEAM_A, [k for k, _, t in SERVICES if t == TEAM_A]),
        (TEAM_C, [k for k, _, t in SERVICES if t == TEAM_C]),
    ):
        # non-key query: shares service n-grams but is not an exact key
        hits = store.search(f"who owns the {keys[0]} exactly", top_k=5)
        team_hits = [h for h in hits if team in h.content.lower()]
        out[team] = (len(team_hits), len(hits))
    return out


def s2a_llm(model, tok, store) -> None:
    """LLM half: aggregation over injected memories."""
    import torch

    def gen(prompt):
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            o = model.generate(**inputs, max_new_tokens=32, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(o[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # fetch memories for atlas (multiple), inject ALL, ask for a count
    atlas_keys = [k for k, _, t in SERVICES if t == TEAM_A]
    hits = []
    for k in atlas_keys:
        h = store.search(k, top_k=1)
        if h:
            hits.append(h[0])
    mem = "\n".join(f"- {h.content}" for h in hits)
    prompt = (f"MEMORY:\n{mem}\n\n"
              "Q: How many services are owned by Team Atlas? Answer with just a number.\nA:")
    ans = gen(prompt)
    got = sum(str(len(atlas_keys)) in ans[:15] for _ in [0])
    print(f"S2a aggregation: asked 'how many Atlas services' with {len(hits)} memories injected")
    print(f"   answer: {ans.strip()[:60]!r} -> {'PASS' if got else 'FAIL'}")


# ─────────────────────────────────────────────────────────────────────────────
# S2b: multi-hop chain A -> B -> C stored as separate facts
# ─────────────────────────────────────────────────────────────────────────────

CHAIN = [
    ("backup owner", "The backup pipeline is owned by Team Atlas."),
    ("atlas lead", "Team Atlas is led by Dana Ortiz."),
    ("dana location", "Dana Ortiz works from the Lisbon office."),
]


def s2b_llm(model, tok, store) -> None:
    """The model must chain: backup pipeline -> Atlas -> Dana -> Lisbon.
    No single memory contains the answer. The agentic pattern: each hop
    queries memory by the RESOLVED ENTITY from the previous hop (not the
    full NL question, which has no lexical bridge to the next fact), then
    ALL hop memories are injected and the LLM composes the final answer."""
    import torch

    def gen(messages, max_new=32):
        """Instruct models must be driven via the chat template — raw
        Q:/A: completion prompts trigger essay mode and break ReAct."""
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            o = model.generate(**inputs, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(o[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # ReAct-style agentic loop: the model's short intermediate answer
    # becomes the next lookup query; already-seen memories are excluded.
    q = "who owns the backup pipeline"
    memories: list[str] = []
    seen: set[str] = set()
    for _hop in range(3):
        hits = [h for h in store.search(q, top_k=3) if h.content not in seen]
        if not hits:
            break
        m = hits[0].content
        seen.add(m)
        memories.append(m)
        answer = gen(
            [
                {"role": "system", "content": "You output ONLY the entity name. Maximum 4 words. No explanations, no punctuation."},
                {"role": "user", "content": f"Memory: {m}\n\nQuestion: {q}\nEntity name:"},
            ],
            max_new=12,
        ).strip().rstrip(".")
        print(f"   hop: {q!r} -> {m[:52]!r} -> next query {answer!r}")
        q = answer
    mem = "\n".join(f"- {m}" for m in memories)
    final_q = "Where does the owner of the backup pipeline work?"
    final = gen(
        [
            {"role": "system", "content": "Answer with just the city name."},
            {"role": "user", "content": f"MEMORY:\n{mem}\n\nQuestion: {final_q}"},
        ]
    )
    ok = "lisbon" in final.lower()
    print("S2b multi-hop (ReAct loop + composition):")
    for m in memories:
        print(f"   memory: {m[:60]!r}")
    print(f"   final: {final.strip()[:60]!r} -> {'PASS' if ok else 'FAIL'} (want lisbon)")


# ─────────────────────────────────────────────────────────────────────────────
# S2c: distractor robustness — correct needle among N similar distractors
# ─────────────────────────────────────────────────────────────────────────────

def s2c_retrieval(n_distractors=200) -> None:
    store = SqacStore()
    # target fact with distinctive code
    target_key = "horizon deploy policy"
    target_fact = "Project Horizon deploys at 09:00 UTC on Tuesdays."
    store.add(target_fact, key=target_key)
    # distractors: similar shape, different project
    for i in range(n_distractors):
        store.add(
            f"Project {WORDS[i % len(WORDS)].capitalize()}-{i} deploys at {i % 24}:00 UTC on Mondays.",
            key=f"{WORDS[i % len(WORDS)]} deploy policy {i}",
        )
    hits = store.search(target_key, top_k=1)
    ok = hits and hits[0].content == target_fact
    print(f"S2c distractors: target found among {n_distractors} similar distractors -> {'PASS' if ok else 'FAIL'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="run end-to-end LLM tests (Qwen)")
    ap.add_argument("--max-n", type=int, default=25_000)
    args = ap.parse_args()

    # S1 + S2 retrieval-level
    test_recall_vs_n(ns=[n for n in (100, 1_000, 5_000, 10_000, 25_000) if n <= args.max_n])

    store = SqacStore()
    for k, f, _t in SERVICES:
        store.add(f, key=k)
    r = s2a_retrieval(store)
    for team, (n_team, n_total) in r.items():
        print(f"S2a retrieval ({team}): {n_team}/{n_total} top-5 hits are same-team")

    s2c_retrieval()

    # S2 LLM-level
    if args.llm:
        import torch

        torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
        model_name = os.environ.get("SQAC_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"\n=== loading {model_name} for end-to-end ===")
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
        model.eval()

        mstore = SqacStore()
        for k, f, _t in SERVICES:
            mstore.add(f, key=k)
        s2a_llm(model, tok, mstore)

        cstore = SqacStore()
        for k, f in CHAIN:
            cstore.add(f, key=k)
        s2b_llm(model, tok, cstore)

    return 0


if __name__ == "__main__":
    sys.exit(main())
