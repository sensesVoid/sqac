# Fissure–Fusion–Evocation: Experiment Plan
### Two-Stage Learned Codebook + VSA Addressable Memory

---

## 1. Architecture Under Test

```
RAW DATA
   |
   v
[STAGE 1: Learned Compressor]
chunk -> encoder -> nearest codeword in codebook(K)
   |
   v
discrete code c_i  (log2(K) bits each)
   |
   v
[STAGE 2: VSA Structural Layer]
V_i = ItemMemory[c_i]                         (finite lookup, K entries)
BUCKET = SUM ( X_pos (x) Y_level (x) Z_context (x) V_i )   per chunk group
   |
   v
COMPRESSED + ADDRESSABLE STORE
   |
   v
QUERY: unbind(X,Y,Z) -> noisy V_i -> cleanup(ItemMemory) -> c_i -> decoder -> chunk
```

**Design principle:** Stage 1 does the actual compression (learned, corpus-fit
codebook — same family as VQ-VAE / RVQ audio-codec codebooks). Stage 2 adds
structure and addressability on top; it is not expected to compress further.
Cleanup in Stage 2 only has to search a known, finite codebook of size K
(not "all possible byte strings"), which is what makes retrieval tractable.

**Target consumer:** the output of this pipeline (discrete codes + VSA
addressable buckets) is intended as a data feed for a VSA-native model —
not for training a transformer/LLM. Encoding and decoding operations should
be evaluated against that target (compositional/associative retrieval)
rather than against transformer training objectives.

---

## 2. Stage 1 — Learned Codebook (do this first, validate independently)

| Item | Decision |
|---|---|
| Chunk size | Fixed window (e.g. 256B, or token windows for text) — pick per domain |
| Encoder | Shallow conv/MLP — kept lightweight by design, not tied to a specific device's compute budget |
| Codebook structure | Residual/product quantization: multiple small sub-codebooks per chunk rather than one large K |
| Domain handling | Separate codebook per data domain (text / logs / tabular / etc.) if the 1TB is heterogeneous — one universal codebook will underfit |
| Success gate | Reconstruction quality (decoder output vs. original chunk) meets threshold **before** touching Stage 2 |

**Gate:** Do not proceed to Stage 2 experiments until Stage 1 reconstruction
error is measured and acceptable. Conflating a Stage 1 bug with a Stage 2
retrieval failure is the main way this whole plan produces uninterpretable
results.

---

## 3. Capacity Budget (compute before running anything)

Empirical HDC decoding results put recoverable information at roughly
**1.2–1.4 bits per dimension** at best (smaller codebooks: ~1.4 bits/dim;
larger codebooks: ~1.26 bits/dim), after interference-cancellation-based
decoding — not the naive/older bound of ~0.6–1.2 bits/dim.

For D = 1024: real budget ≈ 1,300 bits per bucket after binding/bundling
overhead.

```
max_chunks_per_bucket ≈ 1300 / (log2(K) + coordinate_overhead_bits)
```

Compute this number for your actual K and coordinate scheme **before**
Experiment A. This is the theoretical collapse point every experiment below
gets checked against.

- Empirical collapse ≈ predicted collapse → implementation is correctly
  capacity-bound (expected, good).
- Empirical collapse well below predicted → bug in binding/unbinding/cleanup
  code, not a fundamental limit.

---

## 4. Stage 2 Experiments

All retrieval accuracy is now **exact discrete-code classification**
(did unbind+cleanup return the correct `c_i`?) — not fuzzy continuous-content
matching. This is the fix that makes numbers comparable across runs.

### Experiment A — Number of subvectors per bucket
Fix D = 1024. Vary items-per-bucket: 1, 2, 4, 8, 16, 32, 64, 128...
Measure retrieval accuracy vs. the Section 3 predicted collapse curve.

### Experiment B — Hyperdimension
Fix bundle degree. Vary D: 512, 1024, 2048, 4096, 8192, 16384, 32768.
Tests whether the bottleneck is fundamentally dimensional capacity.

### Experiment C — Addressing scheme (isolate structure vs. capacity)
Compare, **holding codebook size K and bundle degree constant across all
five**:

| Variant | Description |
|---|---|
| C0 | Fusion only (no positional/coordinate binding) |
| C1 | Fusion + positional address |
| C2 | Fusion + coordinate address (X/Y/Z) |
| C3 | Fusion + coordinate address + metadata |
| C4 | Hierarchical buckets + coordinate address |

Report each variant's accuracy against the Section 3 ceiling, not just
against each other — a variant "winning" a comparison that's still far
below the theoretical ceiling means there's headroom left to find.

### Experiment D — Codebook size (K) as its own axis
Vary K independently of addressing scheme: this was folded into "number of
subvectors" before, which conflated two different capacity knobs. Test K at
several sizes and confirm the Section 3 formula's `log2(K)` term holds
empirically.

### Experiment E — Bound-role count
Vary how many roles get bound before bundling (position only → position+level
→ position+level+context). Each additional bound role adds cross-term
interference at unbind time — measure this cost directly rather than
assuming coordinate binding is free.

---

## 5. Metrics to Log Per Run

- Stage 1 reconstruction error (independent of Stage 2)
- Stage 2 exact-match retrieval accuracy (discrete code classification)
- Bucket fill (items actually bundled vs. Section 3 max_chunks_per_bucket)
- Wall-clock / compute cost per encode and per query
- Overall ratio: (original bytes) / (Stage 1 codes + Stage 2 structural overhead)

## 6. Reporting Table (fill in as data arrives)

| Compression | Stage 1 recon | Stage 2 retrieval | Compute |
|---|---|---|---|
| 1x (baseline) | | | |
| 2x | | | |
| 4x | | | |
| 8x | | | |
| 16x | | | |
| 32x | | | |

The point where Stage 2 retrieval collapses relative to the Section 3
prediction is the useful compression frontier for this design.

---

## 7. Known Open Question

Whether this is best framed as a general compressor or as a structured,
queryable memory substrate feeding a VSA-native model (associative,
compositional retrieval with interpretable similarity thresholds, rather
than lossless byte reconstruction) is still unresolved — worth revisiting
once Stage 1 numbers are in, since it changes which metric in Section 5
actually matters most.