#!/usr/bin/env python3
"""SQAC latency benchmark — replaces thesis estimates with measurements.

Run:  python tests/bench_sqac.py [n_rules]
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqac.store import SqacStore

WORDS = (
    "auth deploy repository middleware scope image kebab config payment service "
    "schema migration rollback cache queue worker metric trace log alert policy "
    "quota rate limit token secret vault rotate audit access role owner team "
    "pipeline staging prod canary feature flag experiment cohort segment funnel"
).split()


def make_rules(n: int) -> list[tuple[str, str]]:
    rules = []
    for i in range(n):
        w1, w2, w3 = WORDS[i % len(WORDS)], WORDS[(i * 7) % len(WORDS)], WORDS[(i * 13) % len(WORDS)]
        rules.append(
            (
                f"Rule {i}: {w1} {w2} {w3} procedure requires review step {i % 97}",
                f"{w1} {w2} {w3} rule {i}",
            )
        )
    return rules


def bench(n: int) -> dict:
    store = SqacStore()
    rules = make_rules(n)

    t0 = time.perf_counter()
    for content, key in rules:
        store.add(content, key=key)
    write_ms = (time.perf_counter() - t0) * 1000 / n

    # exact
    t0 = time.perf_counter()
    for content, key in rules[:: max(1, n // 100)]:
        store.search(key)
    exact_ms = (time.perf_counter() - t0) * 1000 / max(1, len(rules[:: max(1, n // 100)]))

    # fuzzy: paraphrase-ish queries (shared trigrams, not exact)
    t0 = time.perf_counter()
    for i in range(0, n, max(1, n // 50)):
        content, _ = rules[i]
        store.search(content[: max(len(content) // 2, 12)])
    fuzzy_ms = (time.perf_counter() - t0) * 1000 / max(1, len(range(0, n, max(1, n // 50))))

    path = f"/tmp/bench_{n}.sqac"
    store.save(path)
    size = os.path.getsize(path)
    os.remove(path)

    return {
        "n": n,
        "write_ms": round(write_ms, 4),
        "exact_ms": round(exact_ms, 4),
        "fuzzy_ms": round(fuzzy_ms, 2),
        "size_mb": round(size / 1e6, 3),
    }


def main() -> None:
    ns = [int(a) for a in sys.argv[1:]] or [1000, 5000, 10000]
    print(f"{'rules':>7} | {'write/op':>9} | {'exact/op':>9} | {'fuzzy/op':>9} | {'size':>8}")
    print("-" * 55)
    for n in ns:
        r = bench(n)
        print(f"{r['n']:>7} | {r['write_ms']:>7}ms | {r['exact_ms']:>7}ms | {r['fuzzy_ms']:>7}ms | {r['size_mb']:>6}MB")


if __name__ == "__main__":
    main()
