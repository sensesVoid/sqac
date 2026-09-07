# SQ-LM v2 — Consolidated Findings & Methods

> **Date**: 2026-09-05
> **Status**: Active — SQ-LM v2 prototype + scaling experiments
> **Scope**: Architecture, training, compression, knowledge store, TurboVec synergy, benchmarks

---

## 1. Architecture Decisions

### 1.1 Core Model

| Component | Implementation | Line(s) | Notes |
|-----------|---------------|---------|-------|
| Token encoder | Learned `nn.Embedding(vocab, dim)` + position bind | `sq_lm_v2.py:98-131` | THDC-style; real-valued, normalized |
| Position encoding | Fixed random BSC hypervectors, bipolar multiply | `sq_lm_v2.py:112-113, 304-308` | Order-sensitive, cyclic window `BIND_DIM=32` |
| Recurrence | Gated `SequenceMemory` (keep/write gates) | `sq_lm_v2.py:133-169` | Erase-then-write, VSA-flavored delta rule |
| Depth | Configurable `num_layers` (default 2) | `sq_lm_v2.py:267-278` | `nn.ModuleList` of `SequenceMemory` layers |
| Decoder | Cosine similarity + batched MLP cleanup | `sq_lm_v2.py:217-262` | Avoids per-step unroll; one matmul over sequence |
| Knowledge store | Additive VSA lookup, cosine threshold | `sq_lm_v2.py:174-215` | Non-destructive, facts added post-training |
| Binarization | `binarized()` method | `sq_lm_v2.py:421-444` | Sign-quantize embeddings + KV; gates stay float |

### 1.2 Design Rationale

**Why gated recurrence instead of attention?**
- O(1) memory state per layer (unbounded context)
- No quadratic compute; constant per-step cost
- Directly inspired by HDC-Brain v14.1 parallel-scan memory (Oleg Hasjanov, 2026)
- Simplified further: removed binding attention, kept pure recurrence

**Why real-valued hypervectors instead of bipolar codebook?**
- Easier training: straight-through estimator not required
- Gradients flow through normalized real features
- Binarization is a post-training deployment option (`binarized()`)

**Why batched MLP decoder?**
- Per-step MLP made backward 11× slower via unrolled graph
- Batched projection: `states (S,D) → project → cosine against emb (V,D)`
- One matmul, no unroll, same nonlinearity

**Why knowledge store?**
- Non-forgetting: append-only, no overwrites
- Facts live ONLY in knowledge store, not corpus (proves non-forgetting vs memorization)
- Added AFTER training, preserving base model unchanged
- Injection at decode time: `state = (1-α)*state + α*knowledge_hv`

### 1.3 Architectural Lineage

| Prior Work | Contribution to SQ-LM v2 |
|------------|-------------------------|
| **HDC-Brain v14.1** (Hasjanov, 2026) | Gated recurrence, learned embeddings, parallel-scan memory, thought loops → simplified to N-layer VSA recurrence |
| **Augeri Hypertokens (2025)** | Proves VSA works in latent space; motivates external VSA routing |
| **torchhd (JMLR 2023)** | BSC binding, bundling, position hypervectors |
| **Kanerva (1996)** | Orthogonality bounds, HDC foundations |

---

## 2. Training Insights

### 2.1 What Works

| Technique | Result | Notes |
|-----------|--------|-------|
| Batched teacher forcing | ✅ Essential for D≥512 | `train_logits_batch()` is 10-100× faster than single-sequence |
| Adam optimizer, lr=1e-2 | ✅ Converges on template grammar | Default for prototype; may need tuning at scale |
| Cosine similarity logits | ✅ Stable training | `logits = (st @ emb.t()) * 8.0`; scale prevents vanishing gradients |
| Knowledge injection post-training | ✅ Conf 1.0 on exact matches | Facts stored after training; non-destructive |

### 2.2 Scaling Observations

| Config | Val PPL | Val Acc | Time/Epoch | Notes |
|--------|---------|---------|------------|-------|
| D=256, L=2 | ~10.9 | 40% | ~13s | Baseline from earlier runs |
| D=512, L=2 | 1.59 | 84.6% | ~28s | Sweet spot for template grammar |
| D=512, L=3 | 1.63 | 84.5% | ~45s | Same quality, slower |
| D=1024, L=2 | 1.52 | 85.8% | ~70s | Modest quality gain, 2.5× slower |
| D=1024, L=3 | ~3.8* | 67%* | ~155s | Needs more epochs/LR tuning |

*At 5 epochs only; still converging

**Key finding**: D=1024 L=2 gives +1.2% val_acc over D=512 L=2, but at 2.5× time cost. For this corpus size, D=512 L=2 is the practical sweet spot.

### 2.3 Training Bottlenecks

| Bottleneck | Severity | Mitigation |
|------------|----------|------------|
| Python-loop recurrence | HIGH | Vectorize with `torch.scan` or chunked matmul |
| `KnowledgeStore.add` append | HIGH | Pre-allocate buffers or use index structure |
| Per-token `embed_token` | MEDIUM | Batch-encode whole sequence |
| Single-sequence SGD | MEDIUM | Always use `train_logits_batch()` at scale |

### 2.4 Corpus Compression Impact

| Corpus | Full → Compressed | Speedup | Quality Impact |
|--------|------------------|---------|----------------|
| Synthetic duplicates | 60 → 18 (70%) | **4.38x** | Loss +0.13 (significant) |
| Template grammar (200) | 200 → 188 (6%) | **2.62x** | Loss +0.08 (minor) |
| Template grammar (600) | 600 → 503 (16%) | ~2x | Minimal at 16% reduction |
| Demo corpus (20) | 20 → 20 (0%) | 1.0x | None |

**Method**: VSA-based semantic dedup via `SQLM.compress_corpus(threshold=0.85)`
- Encodes each sentence as position-bound bundle
- Greedy removes near-duplicates above cosine threshold
- Preserves original order

**Sweet spot**: 10-20% reduction for corpora with mild duplication; expect 2-3× training speedup with <5% quality loss.

---

## 3. Compression Research Summary

### 3.1 What We Tested

| Approach | Ratio | Accuracy | Verdict |
|----------|-------|----------|---------|
| BSC bundling | 15:1 | 0-6% | ❌ Lossy for sequences |
| Sparse BSDC | 0.03:1 | 100% | ❌ No compression |
| **Product Quantization** | **5,024:1** | **100%** | ✅ On float32, ❌ on binary (codebooks 80× larger) |
| Huffman + VSA | 8.5:1 | 100% | ⚠️ Limited gain |
| Code-specific | 4.2:1 | 100% | ⚠️ Domain-specific |
| Semantic Huffman | 8.5:1 | 100% | ⚠️ Limited gain |
| Elevated stack (4-layer) | 12.6:1 | 100% | ⚠️ Complex |
| Structural VSA | 10-100:1 | Approximate | ⚠️ Lossy |

### 3.2 The Honest Truth

**PQ fails on binary VSA vectors** — codebooks are 80× larger than data at D=10,048. PQ is designed for float32 embeddings (e.g., 768-dim BERT), not binary hypervectors.

**What actually works for binary VSA:**
1. Semantic dedup: 5-20× reduction
2. Vocabulary sharing: 2-5× reduction
3. zstd lossless: 3-10× reduction
4. Combined: **25-200×** on structured knowledge

**Achievable targets:**
- 100K rules → 4 MB (25:1) ✅
- 1TB structured knowledge → 30 GB (33:1) ✅
- 1TB → 1 GB ❌ (below Shannon limit for most data)

### 3.3 Key Lesson

> **Compression must happen at the semantic level, not on binary vectors themselves.**

Binary vectors are already at 1 bit/dimension — you can't compress them further without loss. The wins come from:
- Removing duplicates (semantic dedup)
- Sharing atoms across entries (vocabulary)
- Standard lossless compression on the encoded form

---

## 4. TurboVec Synergy

### 4.1 What TurboVec Adds

| Metric | Raw VSA | TurboVec 2-bit | Improvement |
|--------|---------|---------------|-------------|
| Storage (1K vectors) | 38.3 MB | 7.6 MB | **5.1×** |
| Search time (1K vectors) | 17-60 ms | 1.35 ms | **40× faster** |
| Accuracy | 100% | 100% | No loss |

### 4.2 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              SQA STORE (Production)                          │
│                                                              │
│  ┌──────────────────────┐  ┌──────────────────────────────┐ │
│  │  EXACT LOOKUP         │  │  FUZZY/SEMANTIC SEARCH       │ │
│  │  (Python dict)        │  │  (TurboVec SIMD)             │ │
│  │                       │  │                              │ │
│  │  O(1) lookup          │  │  O(n) SIMD scan              │ │
│  │  62.5 KB for 1K rules │  │  7.6 MB for 1K rules         │ │
│  │  Exact match only     │  │  Fuzzy + semantic + exact    │ │
│  │                       │  │  1.35 ms for 1K vectors      │ │
│  └──────────────────────┘  └──────────────────────────────┘ │
│                                                              │
│  Query flow:                                                 │
│    1. Try dict (O(1), instant)                               │
│    2. If miss → TurboVec search (1.35 ms)                    │
│    3. Return best result                                     │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 Why TurboVec + VSA Works

VSA vectors are binary (0/1). When stored as float32, each bit becomes a float — 32× overhead. TurboQuant compresses back toward optimal using:
1. **Random rotation** — simplifies geometry
2. **PolarQuant** — angle = meaning, radius = strength
3. **QJL (1-bit residual)** — eliminates bias with zero overhead

**Result**: Near-optimal compression, no training required, online/streaming support.

### 4.4 TurboVec vs PQ

| Property | PQ | TurboQuant |
|----------|-----|-----------|
| Training | Required (k-means) | **None** (data-oblivious) |
| Compression | 8-32× | **8-16×** |
| Accuracy | Good | **Near-optimal** |
| Speed | Good | **3.4× faster than FAISS** |
| Online | No (batch only) | **Yes (streaming)** |

---

## 5. Multi-Vector Insight

### 5.1 The Key Finding

**VSA is for fuzzy/semantic search, not exact lookup.**

| Approach | Accuracy | Speed | Use Case |
|----------|----------|-------|----------|
| Bundled keys (VSA) | 100% | Slow (O(n) scan) | Exact match when encoding is consistent |
| Atomic keys (VSA) | 0% | Slow (O(n) scan) | Fuzzy/semantic similarity |
| Python dict (hash map) | 100% | Fast (O(1)) | Exact lookup |
| **Combined** | **100% + fuzzy** | **Best of both** | **Production system** |

### 5.2 The Architecture

```python
class MultiVectorStore:
    def lookup(self, query):
        # 1. Try exact lookup (fast)
        key = (query['type'], query.get('language'), query.get('rule'))
        if key in self.exact_index:
            idx = self.exact_index[key]
            return self.metadata[idx], 1.0, "exact"

        # 2. Fall back to VSA similarity (fuzzy)
        query_hv = self.encode_query(query)
        results = self.vsa_search(query_hv, top_k=1)
        if results:
            return results[0]['rule'], results[0]['confidence'], "fuzzy"

        return None, 0.0, "none"
```

---

## 6. Methods & Recipes

### 6.1 VSA-Based Corpus Deduplication

**When to use**: Training data with near-duplicates or paraphrases
**Benefit**: 2-4× training speedup with <5% quality loss at 10-20% reduction
**Method**:
```python
compressed = model.compress_corpus(texts, threshold=0.85)
```
- Encodes each text via `encode_query()` (position-bound bundle)
- Greedy removes texts with cosine similarity > threshold to any kept text
- Preserves original order

**Sweet spot**: 10-20% corpus reduction. Aggressive dedup (>50%) hurts quality.

### 6.2 Knowledge Store Injection

**When to use**: Adding facts after training without fine-tuning
**Benefit**: Non-destructive, zero forgetting, conf 1.0 on exact matches
**Method**:
```python
khv = model.encode_query(key_text)
vhv = model.encode_query(value_text)
model.knowledge.add(khv, vhv, {"key": key_text, "value": value_text})
```

### 6.3 Model Binarization for Edge

**When to use**: Deployment on CPU/edge devices
**Benefit**: 32× smaller embeddings, XNOR+POPCNT compute
**Method**:
```python
edge_model = model.binarized()
```
- Sign-quantizes token embeddings + knowledge KV to {+1,-1}
- Gates/recurrence weights stay float (need precision)
- Model still normalizes in decoder

### 6.4 Scaled Training

**When to use**: Larger dims (D≥512) or corpora
**Benefit**: 10-100× faster than single-sequence SGD
**Method**:
```python
logits = model.train_logits_batch(ids_batch)  # (B, S, V)
```
- Batch-encodes whole sequence, runs recurrence in parallel
- One decoder call over all positions

### 6.5 Configurable Depth

**When to use**: Scaling model capacity
**Benefit**: Same code, different depth
**Method**:
```python
model = SQLM(tokenizer, dim=1024, num_layers=3)
```
- `nn.ModuleList` of `SequenceMemory` layers
- Default 2 layers; tested up to 3 at D=512, D=1024

### 6.6 TurboVec for Knowledge Store (NOT Training)

**When to use**: Production knowledge store with 1K+ facts
**Benefit**: 40× faster fuzzy search, 5× smaller storage
**Limitation**: Does NOT help with training — TurboVec is a float32 vector quantizer, not a text encoder or training accelerator

**What TurboVec actually does**:
- Compresses float32 embedding vectors to 2-4 bits/dimension
- Enables SIMD-accelerated similarity search
- Requires zero codebook training (data-oblivious)
- Based on Google Research TurboQuant (ICLR 2026)

**Where it fits in SQ-LM v2**:
| Component | TurboVec Applicable? | Why |
|-----------|---------------------|-----|
| Training data encoding | ❌ No | TurboVec needs float32 embeddings, not raw text |
| Knowledge store lookup | ✅ Yes | Knowledge values are float32 HVs; TurboVec accelerates search |
| Decoder embeddings (inference) | ⚠️ Possible | Could quantize embedding table post-training for smaller model |
| Corpus compression | ❌ No | TurboVec compresses vectors, not text; text must be encoded first |

**Current knowledge store bottleneck**:
- `KnowledgeStore.lookup()` does O(K) cosine similarity on full float32 matrices
- At 10K facts: ~10 ms per lookup
- With TurboVec 2-bit: ~0.25 ms per lookup (40× faster)

**Recommendation**: Integrate TurboVec into `KnowledgeStore` for production deployment. Do NOT expect training speedups.

---

## 7. Benchmark Results

### 7.1 Scaling Benchmark (template grammar)

| D | L | Val PPL | Val Acc | Time |
|---|---|---------|---------|------|
| 512 | 2 | 1.59 | 84.6% | 143s/10ep |
| 512 | 3 | 1.63 | 84.5% | ~200s/10ep |
| 1024 | 2 | 1.52 | 85.8% | ~350s/10ep |
| 1024 | 3 | 3.83* | 67%* | ~155s/5ep |

*Still converging

### 7.2 Corpus Compression Benchmark

| Corpus | Full | Compressed | Speedup | Quality Loss |
|--------|------|-----------|---------|-------------|
| Synthetic (60) | 60 sents | 18 sents | 4.38x | +0.13 loss |
| Template (200) | 200 sents | 188 sents | 2.62x | +0.08 loss |
| Demo (20) | 20 sents | 20 sents | 1.0x | 0.00 |

### 7.3 Knowledge Recall

| Fact | Confidence | Status |
|------|-----------|--------|
| "the capital of france is" | ~1.0 | ✅ Confident hit |
| "the largest planet is" | ~1.0 | ✅ Confident hit |
| "water freezes at" | ~1.0 | ✅ Confident hit |
| Open-ended generation | ~0.0 | ❌ Repetition artifacts (expected at tiny scale) |

---

## 8. VSA/HDC Training Acceleration Techniques

> **Sources**: THDC (Dejonghe & Leroux, 2026), TrainableHD (Kim et al., 2024), OnlineHD (Hernandez-Cano et al., 2021), LARS-VSA (Mejri et al., 2024), RESOLVE (Mejri et al., 2024)

### 8.1 Trainable Embeddings (THDC)

**Paper**: "THDC: Training Hyperdimensional Computing Models with Backpropagation" (ESANN 2026)

**Key finding**: Replace random hypervectors with trainable embeddings, backprop via straight-through estimator.

| Metric | Traditional HDC | THDC |
|--------|----------------|------|
| Dimensionality | 10,000 | **64** |
| Accuracy (MNIST) | ~90% | **~98%** |
| Accuracy (CIFAR-10) | ~60% | **~80%** |
| Memory | 10K dims per HV | 64 dims per HV |

**How it works**:
1. Replace static Item Memory (IM) with `nn.Embedding` tables
2. Train embeddings + one-layer BNN classifier via backprop
3. Straight-through estimator for binarization: gradients flow through real-valued shadow weights
4. After training, binarize weights for inference

**Relevance to SQ-LM v2**: Our encoder already uses trainable embeddings (`nn.Embedding`). THDC validates this design choice and shows we can potentially reduce D from 1024 to 64-256 while maintaining accuracy.

### 8.2 Encoder Interval Training (TrainableHD)

**Paper**: "Advancing Hyperdimensional Computing Based on Trainable Encoding and Adaptive Training" (ACM TODAES 2024)

**Key finding**: Don't re-encode feature hypervectors every iteration. Cache them and update periodically.

**How it works**:
1. Initial iteration: encode all features normally
2. Store encoded feature HVs in memory
3. For N subsequent iterations: reuse cached HVs, only update embeddings
4. Refresh cache every M iterations

**Benefit**: Encoding is often the bottleneck. EIT reduces encoding operations by 5-10× with minimal accuracy loss.

**Relevance to SQ-LM v2**: Our `_embed_ids` and `encode_query` methods could benefit from EIT during training. Position-bound bundles are expensive to recompute every step.

### 8.3 Adaptive Optimizers for HDC (TrainableHD)

**Paper**: Same as above

**Key finding**: Standard SGD works poorly for HDC. Adaptive optimizers (Adam, AdaGrad, RMSprop) significantly improve convergence.

| Optimizer | HDC Accuracy | Notes |
|-----------|-------------|-------|
| SGD | Baseline | Sensitive to LR, oscillates in high-D |
| **Adam** | **+7% avg** | Best for most HDC tasks |
| AdaGrad | +5% avg | Good for sparse features |
| RMSprop | +4% avg | Stable but slower |

**Relevance to SQ-LM v2**: We already use Adam. This validates our choice. Key insight: HDC benefits from per-parameter adaptive learning rates because different dimensions encode different semantic features.

### 8.4 Quantization-Aware Training (TrainableHD)

**Paper**: Same as above

**Key finding**: Train with quantized hypervectors from the start, not as post-processing.

**How it works**:
1. During forward pass: quantize HVs to low-bit (e.g., 2-bit, ternary)
2. During backward pass: use straight-through estimator
3. Model learns to work within quantization constraints

**Benefit**: No accuracy drop when deploying to low-precision hardware. Enables edge deployment without retraining.

**Relevance to SQ-LM v2**: Our `binarized()` method does post-training quantization. QAT could give better quality by training with binary constraints from the start.

### 8.5 Single-Pass Online Learning (OnlineHD)

**Paper**: "OnlineHD: Robust, Efficient, and Single-Pass Online Learning" (DATE 2021)

**Key finding**: HDC can learn in one pass over the data, no epochs needed.

**How it works**:
1. For each sample: encode → predict → update class HV
2. Class HV update: `class_hv = class_hv + η * (sample_hv) * correct`
3. No batch, no epoch, no gradient

**Benefit**: O(n) training time, constant memory, natural for streaming data.

**Relevance to SQ-LM v2**: Could be used for knowledge store updates or fine-tuning on new domains without full retraining. However, our current model uses backprop, so OnlineHD is more applicable to the knowledge store than the core LM.

### 8.6 Binarized Attention (LARS-VSA / RESOLVE)

**Papers**: 
- "LARS-VSA: A Vector Symbolic Architecture For Learning with Abstract Rules" (2024)
- "RESOLVE: Relational Reasoning with Symbolic and Object-Level Features" (2024)

**Key finding**: VSA-based attention is 25× faster than transformer softmax attention.

**How it works**:
1. Compute Q, K, V via bipolar binding (Hadamard product)
2. Attention scores: sigmoid(Q·K^T / sqrt(D)) instead of softmax
3. Output: bundle(value, attention_score) using VSA operations

**Benefit**: 
- 25× faster than dot-product attention
- 17× more memory efficient
- 9× faster than transformer at medium encoder-decoder

**Relevance to SQ-LM v2**: Our current model has NO attention. Adding binarized VSA attention could boost quality while maintaining speed. This is the "binding attention" from HDC-Brain that we removed for simplicity.

### 8.7 Few-Shot Learning with VSA (LARS-VSA)

**Key finding**: LARS-VSA achieves >80% accuracy with just 200 training samples.

| Model | Samples | Accuracy |
|-------|---------|----------|
| Transformer | 200 | ~75% |
| Abstractor | 200 | ~79% |
| **LARS-VSA** | **200** | **>80%** |

**How**: VSA's high-dimensional space has built-in generalization. Binding operations create compositional representations that generalize from few examples.

**Relevance to SQ-LM v2**: Could enable rapid adaptation to new domains with minimal data. Store new domain examples in knowledge store, retrieve via VSA similarity.

### 8.8 Relational Bottleneck (LARS-VSA / RESOLVE)

**Key finding**: Separate object features from abstract rules in high-D space.

**Architecture**:
```
Object features:  bundled into object HV
Abstract rules:  learned as relation HVs
Binding:          object_HV ⊗ rule_HV → combined HV
```

**Benefit**: Reduces interference between features and rules. Better compositional generalization.

**Relevance to SQ-LM v2**: Our knowledge store already separates facts from corpus. Could be extended to separate "style" HVs from "content" HVs for better control.

### 8.9 Matryoshka MoE VSA (Novel)

**Key finding**: Nested hypervectors with mixture-of-experts gating enables adaptive compute.

**Architecture**:
- 1024-dim vector = 4 × 256-dim experts
- Learned gate selects which experts activate per query
- Only active experts perform similarity search
- Nested loss supervises at each nesting level

**Benefits**:
| Metric | Full 1024-dim | Matryoshka MoE (trained) |
|--------|---------------|--------------------------|
| Compute per query | O(1024) | O(256) average (4× less) |
| Storage | 1024 per vector | 1024 per vector (same) |
| Capacity | 1024-dim space | 4×256-dim subspaces |
| Training | Standard | Nested loss at each level |
| Inference speed | Fixed | Adaptive (simple → 1 expert, complex → all) |

**Current status**: POC implemented in `sqlm_matryoshka.py`. Gate is untrained (random), so all experts activate for all queries (0.99x speedup). Requires gate training with load-balancing loss for sparse activation.

**Relevance to SQ-LM v2**: This is the path to a truly small model — 256-dim operations dominate runtime, but 1024-dim storage maintains capacity.

---

## 9. Methods & Recipes

### 9.1 VSA-Based Corpus Deduplication

**When to use**: Training data with near-duplicates or paraphrases
**Benefit**: 2-4× training speedup with <5% quality loss at 10-20% reduction
**Method**:
```python
compressed = model.compress_corpus(texts, threshold=0.85)
```
- Encodes each text via `encode_query()` (position-bound bundle)
- Greedy removes texts with cosine similarity > threshold to any kept text
- Preserves original order

**Sweet spot**: 10-20% corpus reduction. Aggressive dedup (>50%) hurts quality.

### 9.2 Knowledge Store Injection

**When to use**: Adding facts after training without fine-tuning
**Benefit**: Non-destructive, zero forgetting, conf 1.0 on exact matches
**Method**:
```python
khv = model.encode_query(key_text)
vhv = model.encode_query(value_text)
model.knowledge.add(khv, vhv, {"key": key_text, "value": value_text})
```

### 9.3 Model Binarization for Edge

**When to use**: Deployment on CPU/edge devices
**Benefit**: 32× smaller embeddings, XNOR+POPCNT compute
**Method**:
```python
edge_model = model.binarized()
```
- Sign-quantizes token embeddings + knowledge KV to {+1,-1}
- Gates/recurrence weights stay float (need precision)
- Model still normalizes in decoder

### 9.4 Scaled Training

**When to use**: Larger dims (D≥512) or corpora
**Benefit**: 10-100× faster than single-sequence SGD
**Method**:
```python
logits = model.train_logits_batch(ids_batch)  # (B, S, V)
```
- Batch-encodes whole sequence, runs recurrence in parallel
- One decoder call over all positions

### 9.5 Configurable Depth

**When to use**: Scaling model capacity
**Benefit**: Same code, different depth
**Method**:
```python
model = SQLM(tokenizer, dim=1024, num_layers=3)
```
- `nn.ModuleList` of `SequenceMemory` layers
- Default 2 layers; tested up to 3 at D=512, D=1024

### 9.6 Encoder Interval Training (EIT)

**When to use**: Large corpora where encoding dominates training time
**Benefit**: 5-10× faster training by caching encoded HVs
**Method**:
```python
# Pseudocode
cache = {}
for epoch in range(epochs):
    for text in texts:
        if text not in cache:
            cache[text] = model.encode_query(text)
        hv = cache[text]
        # ... use hv for training
    if epoch % refresh == 0:
        cache.clear()  # Refresh cache with updated embeddings
```

### 9.7 Quantization-Aware Training (QAT)

**When to use**: Edge deployment with low-precision hardware
**Benefit**: No accuracy loss when quantizing to 2-4 bit
**Method**:
```python
# Forward pass: quantize HVs
hv_quantized = quantize_to_2bit(hv)
# Backward pass: straight-through estimator
loss.backward()
```

### 9.8 Binarized VSA Attention

**When to use**: Need better quality without full transformer attention
**Benefit**: 25× faster than softmax attention, maintains VSA properties
**Method**:
```python
Q = x ⊗ b_q  # bipolar binding
K = x ⊗ b_k
V = x ⊗ b_v
attn = sigmoid(Q @ K.T / sqrt(D))  # binarized attention
output = bundle(V * attn)
```

---

## 8. Open Questions & Next Steps

### 8.1 Immediate

| Task | Priority | Expected Outcome |
|------|----------|-----------------|
| Vectorize `_run_states` recurrence | HIGH | 10-100× faster training |
| Replace `KnowledgeStore.add` append with pre-allocated index | HIGH | Scales to 100K+ facts |
| Integrate TurboVec into knowledge store | HIGH | 40× faster fuzzy search |
| Test on real corpus (WikiText, etc.) | HIGH | Validate beyond toy grammar |
| Train Matryoshka MoE gate | HIGH | Sparse expert activation, 2-4× speedup |

### 8.2 Medium-term

| Task | Priority | Expected Outcome |
|------|----------|-----------------|
| Add beam search / batched generation | MEDIUM | Better generation quality |
| Add gradient checkpointing for large D | MEDIUM | Train D=2048+ on CPU |
| Implement TurboVec index in `.sqac` format | MEDIUM | Production-ready cartridge |
| Benchmark against real LoRA adapter | MEDIUM | Quantify quality/speed tradeoffs |

### 8.3 Long-term

| Task | Priority | Expected Outcome |
|------|----------|-----------------|
| Add binding attention (HDC-Brain style) | LOW | Quality boost at cost of complexity |
| Semantic codebook initialization | LOW | Better starting point, faster convergence |
| XNOR/POPCNT kernels for inference | LOW | True binary compute advantage |
| Multi-GPU training | LOW | Scale to billion-parameter regime |

---

## 9. Files & References

### 9.1 Core Implementation

| File | Purpose |
|------|---------|
| `sq_lm_v2.py` | Core model (Encoder, SequenceMemory, Decoder, KnowledgeStore, SQLM) |
| `eval_sqlm.py` | Compositional generalization eval (template grammar) |
| `train_real.py` | Real-data training (WikiText, BPE tokenizer) |
| `benchmark_sqlm_v2.py` | Scaled (dim × layers) benchmark sweeps |
| `benchmark_corpus_compression.py` | VSA corpus dedup impact on training |

### 9.2 Research & Design

| File | Purpose |
|------|---------|
| `RESEARCH_VALIDATION.md` | Deep research on HDC/VSA foundations |
| `RESEARCH_SOURCES.md` | Bibliography (18 sources) |
| `MERGED_ARCHITECTURE.md` | LoRA + SQA hybrid design |
| `MULTIVECTOR_INSIGHT.md` | VSA vs hash map finding |
| `TURBOVEC_SYNERGY.md` | TurboVec + SQA architecture |
| `TURBOVEC_COMPRESSION.md` | TurboVec compression honest assessment |
| `COMPRESSION_FINAL.md` | PQ breakthrough + all 8 approaches |
| `COMPRESSION_HONEST_TRUTH.md` | PQ reality check on binary vectors |
| `COMPRESSION_PATHWAYS_RESEARCH.md` | 6 compression pathways |
| `COMPRESSION_SYNTHESIS.md` | Synthesis of experiments 1+2 |
| `COMPRESSION_EXPERIMENT_RESULTS.md` | BSC bundling experiment |

### 9.3 Key External References

| Source | Relevance |
|--------|-----------|
| **HDC-Brain v14.1** (Hasjanov, 2026) | Primary architecture basis: pure HDC LM, bipolar codebook, parallel-scan memory |
| **Augeri Hypertokens (2025)** | VSA in transformer latent space; motivates external routing |
| **torchhd (JMLR 2023)** | VSA operations library (BSC binding, bundling) |
| **TurboQuant/TurboVec** (Google Research) | 5.1× compression, 40× faster search, no training |
| **Kanerva (1996)** | HDC orthogonality foundations |
