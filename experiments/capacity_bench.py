#!/usr/bin/env python3
"""Capacity + latency benchmark: how far does one cartridge go before it stops being fast?

Each size runs in a subprocess (clean peak RSS). Emits one JSON row per size to a
results .jsonl. Measures the real production path: build -> save -> load -> search.

    python experiments/capacity_bench.py <size> --out results.jsonl

Search modes (5 reps each, median reported):
    exact    — normalized-key lookup, O(1)
    fuzzy_hit  — paraphrased query with trigram overlap, O(n)
    fuzzy_miss — distinctive no-match query, O(n) worst case
"""

from __future__ import annotations

import argparse
import json
import random
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqac.store import SqacStore  # noqa: E402

_TOPICS = [
    "docker build", "KV cache", "SIMD scan", "cartridge format", "prefill latency",
    "prompt injection", "exact dict", "BSC encoder", "numpy matrix", "MMAP layout",
    "binary trigger", "fuzzy threshold", "trust scoring", "compaction tombstone",
    "HTTP server", "rack search", "offloader window", "cooldown eviction",
]
_NOUNS = ["deploy", "retrieval", "encode", "offload", "evict", "recall", "promote",
          "compact", "bind", "bundle", "scan", "flush"]
_VERBS = ["timed out", "profiled at", "measured at", "regressed to", "stabilized at",
          "climaxed at", "dipped to", "reached"]
_UNITS = ["ms", "KB", "MB", "tokens", "entries", "µs", "GB", "RPS"]
_DECISIONS = ["postpone", "accept into durable pack", "drop silently", "keep in bucket",
              "mark stale", "graduate to fact"]

_lang = random.Random(20260908)


def _content(idx: int) -> str:
    t = _lang.choice(_TOPICS)
    n = _lang.choice(_NOUNS)
    v = _lang.choice(_VERBS)
    u = _lang.choice(_UNITS)
    val = round(_lang.uniform(3.0, 9500.0), 2)
    d = _lang.choice(_DECISIONS)
    ref = _lang.randint(1, 10_000_000)
    # ~45-token distilled exchange. Query cost is independent of content
    # length (vectors are fixed dims); build_s and file_mb scale with it.
    return (
        f"Exchange {idx}: {n} in {t} {v} {val} {u}; decision {d}. "
        f"Candidate #{ref}, floor 0.6."
    )


def _key(idx: int) -> str:
    return f"{_lang.choice(_TOPICS)} {_lang.choice(_NOUNS)}-{idx} decision"


def bench(size: int) -> dict:
    row: dict = {"size": size}
    t0 = time.perf_counter()
    store = SqacStore()
    t1 = time.perf_counter()
    for i in range(size):
        store.add(_content(i), key=_key(i), source=f"syn:{i}", kind="turn")
    t2 = time.perf_counter()
    row["build_s"] = round(t2 - t1, 3)

    cart = Path(f"/tmp/sqac_cap_{size}.sqac")
    store.save(cart, name=f"cap-{size}")
    t3 = time.perf_counter()
    row["save_s"] = round(t3 - t2, 3)
    row["file_mb"] = round(cart.stat().st_size / 1e6, 2)

    t4 = time.perf_counter()
    loaded = SqacStore.load(cart)
    t5 = time.perf_counter()
    row["load_s"] = round(t5 - t4, 3)
    row["loaded_entries"] = len(loaded)

    queries = {
        "exact": _key(0),
        "fuzzy_hit": _content(0)[:90] + " latency",
        "fuzzy_miss": ("zyxw avalanche quadraphonic kerfuffle " * 6).strip()[:120],
    }
    for mode, q in queries.items():
        times = []
        for _ in range(5):
            t = time.perf_counter()
            hits = loaded.search(q, top_k=3, kind="turn")
            times.append((time.perf_counter() - t) * 1000)
            row[f"{mode}_hits"] = len(hits)
        times.sort()
        row[f"{mode}_ms"] = round(times[len(times) // 2], 3)
        row[f"{mode}_min_ms"] = round(times[0], 3)
    # warm path: matrices now cached; re-measure the cold-unpack cost separately
    cold = SqacStore.load(cart)
    q = queries["fuzzy_miss"]
    t = time.perf_counter()
    cold.search(q, top_k=3, kind="turn")
    row["first_query_ms"] = round((time.perf_counter() - t) * 1000, 3)

    row["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    cart.unlink()
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("size", type=int)
    ap.add_argument("--out", default="experiments/capacity_results.jsonl")
    args = ap.parse_args()
    row = bench(args.size)
    print(json.dumps(row, sort_keys=True))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())