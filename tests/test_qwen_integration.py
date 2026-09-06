#!/usr/bin/env python3
"""SQAC x Qwen2.5-1B-Instruct — the thesis demo.

Teach a small LLM facts it has no way of knowing, persist them to a
.sqac cartridge, reload, and verify the model answers correctly —
then verify the same model WITHOUT memory cannot.

Run:  python tests/test_qwen_integration.py [--baseline]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqac.store import SqacStore

MODEL = "Qwen/Qwen2.5-1.5B-Instruct" if os.environ.get("SQAC_BIG") else "Qwen/Qwen2.5-0.5B-Instruct"

FACTS = [
    ("What is the deployment target?", "deployment target",
     "All production deploys go to ARM64 only. AMD64 images are not supported."),
    ("How should the team handle database access?", "database access rule",
     "Always use the repository pattern; controllers must never run raw SQL."),
    ("What naming convention do we use for config files?", "config naming",
     "Config files are kebab-case YAML, e.g. app-config.yaml, never snake_case."),
    ("Who maintains the payments service?", "payments owner",
     "The payments service is owned by Team Atlas (slack: #atlas-payments)."),
]

QUESTIONS = [
    "Q: What is our deployment target?\nA:",
    "Q: How should the team handle database access?\nA:",
    "Q: What naming convention do we use for config files?\nA:",
    "Q: Who maintains the payments service?\nA:",
]


def build_cartridge(path: str) -> SqacStore:
    store = SqacStore()
    for _q, key, fact in FACTS:
        store.add(fact, key=key, source="integration-test")
    store.save(path, name="demo-facts", description="Qwen integration test cartridge")
    return store


def generate(model, tok, prompt: str, max_new: int = 48) -> str:
    import torch

    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def inject_prompt(hits, question: str) -> str:
    """Transparent injection: text + confidence, never a vector."""
    lines = ["You are a helpful assistant. Use the following memory if relevant.",
             "", "MEMORY:"]
    for h in hits:
        lines.append(f"- {h.content} (confidence {h.confidence:.2f}, source: {h.source})")
    lines += ["", question]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true", help="only run the no-memory baseline")
    args = ap.parse_args()

    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading {MODEL} …")
    import torch
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    dtype = torch.bfloat16 if os.environ.get("SQAC_BIG") else torch.float32
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype)  # CPU
    model.eval()
    print(f"loaded in {time.time() - t0:.1f}s\n")

    # ── 1) teach + persist + reload ────────────────────────────────────
    store = build_cartridge("/tmp/demo.sqac")
    n0 = len(store)
    store = SqacStore.load("/tmp/demo.sqac")  # prove persistence
    assert len(store) == n0 == len(FACTS)
    print(f"cartridge: {len(store)} facts, {os.path.getsize('/tmp/demo.sqac')} bytes\n")

    # ── 2) answer WITH memory ──────────────────────────────────────────
    correct = 0
    keywords = ["arm64", "repository", "kebab", "atlas"]
    for q, key, _fact, kw in zip(QUESTIONS, [f[1] for f in FACTS], FACTS, keywords):
        hits = store.search(key, top_k=1)
        if not hits:
            print(f"RETRIEVAL MISS for {key!r}")
            continue
        t0 = time.time()
        answer = generate(model, tok, inject_prompt(hits, q))
        ok = kw in answer.lower()
        correct += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {q.splitlines()[0]}")
        print(f"       memory: {hits[0].content[:60]}…  (conf {hits[0].confidence:.3f}, {hits[0].mode})")
        print(f"       model : {answer.strip()[:100]}")
        print(f"       ({time.time() - t0:.1f}s)\n")

    # ── 3) baseline WITHOUT memory (same questions, cold model) ───────
    if not args.baseline:
        print("── baseline (no memory) ──")
        baseline_leak = 0
        for q, kw in zip(QUESTIONS, keywords):
            answer = generate(model, tok, q)
            leak = kw in answer.lower()
            baseline_leak += leak
            print(f"[{'LEAK' if leak else 'clean'}] {q.splitlines()[0]} -> {answer.strip()[:80]}")
        print()

    n = len(QUESTIONS)
    print(f"RESULT: {correct}/{n} answered correctly WITH memory")
    if not args.baseline:
        print(f"        {baseline_leak}/{n} leaked in baseline WITHOUT memory (want 0)")
    return 0 if correct == n else 1


if __name__ == "__main__":
    sys.exit(main())
