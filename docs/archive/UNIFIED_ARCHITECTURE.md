# Unified Architecture: SQA + Matryoshka MoE + FissFus

## Design Overview

This integrates three research threads into one model:

1. **SQA/SQ-LM v2**: Learned token embeddings + VSA sequence memory + associative knowledge store
2. **Matryoshka MoE**: Nested hypervectors with expert gating for adaptive compute
3. **FissFus**: Learned codebook compression (Stage 1) + VSA structural memory (Stage 2)

## Architecture

```
Input Text
    │
    ▼
[Stage 1: FissFus VQ-VAE Compressor]
    │  - Char embeddings + PQ subspaces
    │  - Dead code restart + entropy regularization
    │  - Output: discrete codes + codebook
    ▼
Discrete Codes
    │
    ▼
[Stage 2: FissFus VSA Structural Layer]
    │  - Permutation-based addressing
    │  - KROP cleanup for fast retrieval
    │  - Output: addressable memory buckets
    ▼
VSA Memory Buckets
    │
    ▼
[Matryoshka MoE Retrieval]
    │  - Nested hypervectors (D = experts × expert_dim)
    │  - Learned gate selects active experts
    │  - Output: retrieved codes + confidence
    ▼
[SQA Decoder + Reasoning]
    │  - Cleanup network: code → token
    │  - Associative reasoning over retrieved codes
    ▼
Output Text / Answer
```

## Key Innovations

1. **FissFus Stage 1** provides the discrete tokenization layer
   - Replaces SQA's raw character encoding
   - Learns compressed codes from data
   - PQ subspaces + dead code restart ensure full codebook utilization

2. **FissFus Stage 2** provides structural memory
   - Permutation addressing eliminates coordinate interference
   - Each code is stored at a position in VSA space
   - Supports sequence-level retrieval

3. **Matryoshka MoE** provides efficient retrieval
   - Nested vectors: coarse-to-fine search
   - Gate learns which experts activate for each query
   - Only active experts compute → adaptive compute

4. **SQA** provides the overall framework
   - Learned token embeddings
   - Sequence memory with VSA recurrence
   - Decoder/cleanup for token prediction

## Data Flow

### Encoding
```
text → chunks → Stage1 encoder → codes → Stage2 store → memory
```

### Retrieval
```
query → Stage1 encoder → query_code → Stage2 query → candidate codes → Matryoshka gate → active experts → final code
```

### Decoding
```
final code → Stage1 decoder → chunk → SQA decoder → text
```

## Component Sizes

| Component | Parameters | Notes |
|-----------|-----------|-------|
| Stage 1 VQ-VAE | ~500K | Char embeddings + PQ codebooks |
| Stage 2 VSA | ~2M | KROP codebook + coordinates |
| Matryoshka MoE | ~1M | Gate network + expert routing |
| SQA Decoder | ~1M | Cleanup network |
| **Total** | **~5M** | Still lightweight, CPU-friendly |

## Training Strategy

1. **Stage 1**: Train VQ-VAE on text corpus with dead code restart
2. **Stage 2**: Initialize VSA memory from trained codebook
3. **Matryoshka MoE**: Train gate on retrieval tasks
4. **SQA**: Fine-tune decoder on downstream task

Alternatively, end-to-end training with gradient flow through all stages.

## Benefits

1. **Compression**: FissFus Stage 1 provides learned compression
2. **Efficiency**: Matryoshka MoE provides adaptive compute
3. **Structure**: FissFus Stage 2 provides addressable memory
4. **Reasoning**: SQA provides associative memory + cleanup
5. **No Forgetting**: VSA additive memory preserves all knowledge
