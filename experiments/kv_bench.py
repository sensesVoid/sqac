#!/usr/bin/env python3
"""KV cache footprint benchmark — SQAC semantic offload vs full context stuffing.

Question: does deliberately keeping context sparse (SQAC recall) reduce the KV
cache footprint the model must hold, versus stuffing the full accumulated
memory into the context window?

Retrieval quality is MEASURED (does SQAC recall the correct-domain exchange above
threshold). KV bytes come from the canonical estimator in sqac.kvcache, so the
footprint math (stuffing vs sparse) is tested by the suite, not re-derived here.

Usage:
    python experiments/kv_bench.py [--domains M] [--exchanges N] [--topk K ...]
    (--topk accepts one or many; default 3. Use "1 3 10" for a sweep.)
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqac.kvcache import estimate_sparse_recall
from sqac.offloader import ContextOffloader
from sqac.store import SqacStore

# ── canonical KV constant (Llama-3.1-8B, fp16, GQA) ───────────────────────────
KV_BYTES_PER_TOKEN = 32 * 8 * 2 * 128 * 2  # layers*kv_heads*K+V*head_dim*fp16
AVG_TOKENS_PER_EXCHANGE = 45

_R = random.Random(7)

_DOMAINS = [
    "deployment", "billing", "auth", "search", "cache", "database",
    "queueing", "observability", "security", "frontend",
]
_TASKS = [
    "optimize", "migrate", "remove", "pin", "profile", "rate-limit",
    "shard", "rotate", "gateway", "replica",
]
_FACTS = [
    "decision was to adopt postgres read replicas",
    "the p99 drifted to 210ms after the change",
    "we pinned the base image to slim to cut the 504",
    "cost spike traced to the queue polling interval",
    "rate limit set to 120 req/min behind the gateway",
    "migration deferred to Q3, blocked by the schema lock",
    "auth token moved to httpOnly cookie, invalidate on role change",
    "cache TTL lowered to 60s to fix staleness",
]


def _exchange(dom: str, i: int) -> tuple[str, str]:
    task = _R.choice(_TASKS)
    fact = _R.choice(_FACTS)
    q = f"what did we decide to {task} in the {dom} system?"
    a = f"For {dom}, {fact}. Referenced as exchange {i} and confirmed in review."
    return q, a


def sweep(domains: int, exchanges: int, topks: tuple[int, ...]) -> dict:
    # ── build one offloader with the whole memory bucket ────────────────
    off = ContextOffloader(window=exchanges, min_confidence=0.55)
    doms = _DOMAINS[:domains]
    target_dom = doms[0]
    for i in range(exchanges):
        dom = doms[i % len(doms)]
        q, a = _exchange(dom, i)
        off.observe("user", q)
        off.observe("assistant", a)
    off.offload()
    store: SqacStore = off._store
    bucket_entries = len(store)
    bucket_tokens = bucket_entries * AVG_TOKENS_PER_EXCHANGE

    # ── retrieval latency + correctness (one query, once) ──────────────
    query = f"what did we decide to optimize in the {target_dom} system?"
    t0 = time.perf_counter()
    hits = off.recall_detailed(query, top_k=max(topks))
    recall_ms = (time.perf_counter() - t0) * 1000
    retrieved_any_target = any(target_dom in h["content"] for h in hits)
    conf_best = hits[0]["confidence"] if hits else 0.0

    # ── per-topk KV accounting via the canonical estimator ─────────────
    rows = []
    for k in topks:
        e = estimate_sparse_recall(
            bucket_tokens, min(k, bucket_entries) * AVG_TOKENS_PER_EXCHANGE
        )
        rows.append(
            {
                "k": k,
                "recalled_entries": min(k, bucket_entries),
                "recalled_tokens": e.sparse_tokens,
                "sqac_kv_mib": round(e.sparse_kv_bytes / (1024**2), 3),
                "kv_reduction_x": round(e.reduction_x, 2),
                "kv_savings_pct": round(e.savings_pct, 1),
                "stuffing_kv_mib": round(e.stuffing_kv_bytes / (1024**2), 1),
            }
        )

    return {
        "domains": domains,
        "bucket_entries": bucket_entries,
        "bucket_tokens": bucket_tokens,
        "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
        "recall_ms": round(recall_ms, 3),
        "retrieval_hit": retrieved_any_target,
        "best_confidence": round(conf_best, 3),
        "model": "llama-3.1-8b-fp16-gqa",
        "sweep": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", type=int, default=3)
    ap.add_argument("--exchanges", type=int, default=1000)
    ap.add_argument("--topk", type=int, nargs="+", default=[3], help="recall budgets (exchanges); e.g. 1 3 10")
    ap.add_argument("--out", default="experiments/kv_results.jsonl")
    args = ap.parse_args()
    row = sweep(args.domains, args.exchanges, tuple(args.topk))
    print(json.dumps(row, sort_keys=True))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())