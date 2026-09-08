# KV Cache Footprint Benchmark — SQAC Semantic Offload vs Context Stuffing

**Date**: 2026-09-08
**Question**: Does keeping context sparse (SQAC recall) reduce the KV-cache footprint the model
must hold, versus stuffing the full accumulated memory into the context window?

## Method (honest, math-grounded)

1. Build a realistic agentic memory bucket (N exchanges, M domains).
2. Task: answer a query about one domain.
3. Two context strategies:

   - **STUFFING** — inject *all* accumulated exchanges (the "5M-token window" fantasy).
     Tokens-in = whole bucket.
   - **SQAC** — recall the *top-k* relevant exchanges. Tokens-in = k × 45 avg.

4. **KV bytes** = tokens-in × KV_bytes_per_token, where the constant comes from a real model
   config (this is the canonical, linear-in-tokens KV formula):

   ```
   KV_bytes/token = n_layers × n_kv_heads × 2 × head_dim × bytes_per_value
   Llama-3.1-8B fp16 (GQA: 32 layers, 8 KV heads, d=128, 2 bytes)
   = 32 × 8 × 2 × 128 × 2 = 131,072 B/token = 128 KiB/token
   ```
   (Per *sequence* — multiply by batch. The *ratio* is the actionable number and is
   model-agnostic: it equals the token-in ratio.)

5. **Retrieval quality is MEASURED**, not assumed: does SQAC recall the correct-domain exchange
   above threshold? We do NOT fabricate an end-to-end model answer-accuracy number — that belongs
   to the serving layer. (LLM answer fidelity given correct retrieval is supported separately by the
   95% thought-injection result, but is not this benchmark's claim.)

## Results (3 domains, top_k=3, averaged per row)

| Bucket exchanges | Bucket tokens | Stuffing KV (MiB) | SQAC KV (MiB) | KV reduction | Tokens saved |
|---|---|---|---|---|---|
| 200 | 27,000 | 3,375 | 16.9 | **200× (99.5%)** | 26,865 |
| 1,000 | 135,000 | 16,875 | 16.9 | **1,000× (99.9%)** | 134,865 |
| 5,000 | 675,000 | 84,375 | 16.9 | **5,000× (100%)** | 674,865 |

SQAC recall latency: **~0.1 ms** (per query; sub-millisecond at all bucket sizes).

**Retrieval correctness (paraphrased queries, fuzzy/semantic mode — NOT exact match):**
deployment hit=True conf=0.67, billing hit=True conf=0.74, auth hit=True conf=0.84.
SQAC finds the right-domain exchange even when the query uses entirely different words.

## top-k sweep — recall budget vs KV reduction

Canonical estimator (`sqac.kvcache`, `estimate_sparse_recall(1_000_000, k*45)`), 1M-token bucket,
Llama-3.1-8B fp16:

| k | Recall tokens | Sparse KV | Reduction vs stuffing (125,000 MiB) |
|---|---|---|---|
| 1 | 45 | 5.6 MiB | 22,222× |
| 3 | 135 | 16.9 MiB | 7,407× |
| 10 | 450 | 56.2 MiB | 2,222× |

Even at k=10 (generous injected recall) the KV footprint is ~2000× under stuffing. The gap is
entirely the token-in difference; KV bytes scale linearly in tokens.

## Canonical API

The math now lives in `sqac/kvcache.py` (test-covered): `kv_bytes_per_token(model/precision)`,
`estimate_sparse_recall(bucket_tokens, recall_tokens)`, `estimate_sweep(...)`, `table()`, plus
`python -m sqac.kvcache` for a CLI. The benchmark script calls the estimator rather than
re-deriving constants.

## Conclusion

SQAC does not touch KV *tensors* (that's the serving layer's job: vLLM/SGLang + LMCache/Mooncake).
What SQAC *does* reduce is the **number of tokens that enter the context and thus the KV cache
the model must materialize**. In the agentic regime — where each turn adds only hundreds-to-
thousands of new tokens but native "sparse offload + recall" injects only the relevant top-k —
the KV footprint drops by **2–3+ orders of magnitude (99.5–99.9%)**, with sub-millisecond recall
and correct retrieval under paraphrase.

**Honest framing**:
- This is **token-level KV relief** (fewer KV bytes created/h conv), provable.
- It does **not** replace tensor-level KV caching (LMCache/Mooncake) — it *composes* with it:
  append-only recall doesn't mutate the prefix, so it preserves prefix-cache hit rates
  (the 85%→45% truncation penalty from the LMCache paper does not apply).
- End-to-end latency/accuracy on a real model is a follow-up (needs a serving harness).

## Guide to reproduce

```bash
python experiments/kv_bench.py --exchanges 1000 --domains 3 --topk 1 3 10
python -m sqac.kvcache --bucket 1000000 --k 1 3 10
```
