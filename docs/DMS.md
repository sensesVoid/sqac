# DMS — Dynamic Memory Sparsification (stale/decay eviction)

**Date**: 2026-09-08
**Status**: Implemented + tested (31 offloader/DMS tests green)

## Milestone: one cartridge comfortably holds ~40× this repo

The capacity benchmark (`experiments/CAPACITY_BENCHMARK.md`, settled at 25K) proved a single
cartridge holds context far beyond a normal project — this repo extracts to 620 units, so the
25K sweet spot ≈ **40× this whole project** (3+ apps' full context, up to ~15–40 codebases
before eviction is even needed). That unlocked the DMS work: memory can now grow to the ceiling
*and* self-prune intelligently.

## Why DMS

Pure capacity is not enough. As a bucket approaches the practical ceiling (100K, ~4.5M tokens,
~108ms scan) the O(n) linear scan grows and — per the KV-eviction literature — *irrelevant*
context dilutes attention (Make Each Token Count / DBTrimKV: selective forgetting improves
generation vs. full cache). DMS is the staleness/decay layer:
- **TTL aging** keeps the hot short-cycle and the cold-but-valuable long tail (bimodal reuse in
  Qwen-Bailian traces).
- **Access promotion** (ARC-like) protects repeatedly-recalled exchanges.
- **Demotion, not destruction** — low-utility turns leave the hot basket and land in the
  durable fact tier (still recallable), so no memory is ever hard-lost.

## Model

```
utility(exchange) = salience · e^(−λ·age) + α · access_count
```

Three tiers: `live` (window, always kept) → `bucket` (offloaded turn, evictable) → `durable`
(fact, exempt). `SqacStore.sparsify()` ranks the bucket, demotes the low tail by demoting to
`kind=fact`.

Tunables (`UtilityWeights`): `lambda_decay` (hourly TTL decay, default 0.10), `alpha` (per-recall
boost, 0.15), `demote_threshold` (floor under which aged entries qualify, 0.10), `keep_ratio` (0.80).

## Files

- `sqac/dms.py` — policy (pure functions + `DMS` registrar + `rank_store`).
- `sqac/offloader.py` — integration: `dms=` param, `register` on write, `touch` on recall,
  `sparsify()` demotion.
- `sqac/__init__.py` — export `DMS`, `UtilityWeights`.
- `tests/test_dms.py` — scoring, aging, budget, promotion, demotion, no-op, rank_store.
- `README.md` — "Dynamic Memory Sparsification" subsection under Context Offloader.

## Verified behavior

- Aging decays utility exponentially; access_count protects cold-but-used entries.
- `sparsify()` demotes aged low-salience turns to facts (turn→fact), no-op without DMS.
- Backward compatible: all pre-existing offloader tests still green.
