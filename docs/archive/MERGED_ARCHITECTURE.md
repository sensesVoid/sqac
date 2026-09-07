# SQ-LM Merged Architecture — Best of Both Worlds

> **Date**: 2026-09-05
> **Status**: Architecture designed, ready for prototype
> **Research**: PRAG (2025), Neuro-VSA (IBM), Hypertokens (2025), Doc-to-LoRA (Sakana)

---

## The Research Insight

### PRAG (2025) — Parametric Retrieval-Augmented Generation
> "Uses LoRA adapters to parameterize each knowledge domain. The routing function
> selects which adapter to activate based on the query."

This is **exactly** what we're building — but with VSA instead of LoRA for knowledge storage.

### Neuro-VSA (IBM)
> "Combines neural network learning with VSA symbolic computation. The neural
> network handles perception, VSA handles reasoning."

### Hypertokens (2025)
> "VSA operations work directly inside transformer latent spaces. Symbolic
> memory addresses enable efficient key-value operations."

---

## The Merged Architecture: SQ-LM v1

### Core Idea

```
┌─────────────────────────────────────────────────────────────┐
│                    SQ-LM ARCHITECTURE                        │
│                                                              │
│  ┌──────────────────┐    ┌──────────────────────────────┐   │
│  │   NEURAL LAYER    │    │      SYMBOLIC LAYER           │   │
│  │   (SLM + LoRA)    │    │      (SQA Cartridge)          │   │
│  │                    │    │                                │   │
│  │  • Language model  │    │  • Deterministic rules         │   │
│  │  • Style/tone      │    │  • Logic/skills               │   │
│  │  • Probabilistic   │    │  • Fast lookup (XOR+POPCNT)    │   │
│  │  • Creative gen    │    │  • Zero forgetting             │   │
│  └────────┬───────────┘    └──────────────┬─────────────────┘   │
│           │                               │                     │
│           └───────────┬───────────────────┘                     │
│                       │                                         │
│              ┌────────▼────────┐                                │
│              │  ROUTING LAYER   │                                │
│              │  (Query Router)  │                                │
│              │                  │                                │
│              │  "Is this a     │                                │
│              │   rule query    │                                │
│              │   or creative?" │                                │
│              └────────┬────────┘                                │
│                       │                                         │
│              ┌────────▼────────┐                                │
│              │  FUSION LAYER    │                                │
│              │  (KV-Cache       │                                │
│              │   Injection)     │                                │
│              └────────┬────────┘                                │
│                       │                                         │
│              ┌────────▼────────┐                                │
│              │  OUTPUT LAYER    │                                │
│              │  (Language Gen)  │                                │
│              └─────────────────┘                                │
└─────────────────────────────────────────────────────────────┘
```

### The Three Modes

#### Mode 1: Pure Symbolic (SQA only)
**Use when**: Query matches a known rule exactly
```
Query: "How do I borrow in Rust?"
→ SQA lookup: exact match found
→ Return: "Use `let x = &y` for immutable borrow"
→ LoRA: NOT activated (zero neural compute)
```

#### Mode 2: Pure Neural (LoRA only)
**Use when**: Query is creative/stylistic, no rule matches
```
Query: "Write a poem about ownership"
→ SQA lookup: no match (confidence < threshold)
→ LoRA: activated for style/tone
→ Return: Creative generation
```

#### Mode 3: Hybrid (SQA + LoRA)
**Use when**: Query needs both rules AND style
```
Query: "Explain Rust ownership like I'm 5"
→ SQA lookup: rule found (deterministic content)
→ LoRA: activated for simplification style
→ Return: Rule-accurate + style-appropriate
```

---

## The Key Innovation: VSA-Guided LoRA

### Problem with Standard LoRA
- Updates destroy prior knowledge (catastrophic forgetting)
- No way to "undo" a training update
- Black box — can't inspect what was learned

### Solution: VSA as LoRA's Memory Layer

```
Standard LoRA:
  W' = W + BA    (one adapter, overwrites everything)

VSA-Guided LoRA:
  W' = W + BA + Σ(vsa_rule_i × routing_weight_i)
                  ↑
                  VSA knowledge injected into LoRA
```

**How it works**:
1. SQA stores rules as VSA hypervectors
2. At inference, VSA lookup determines which rules apply
3. Applicable rules are injected into LoRA's low-rank matrices
4. LoRA generates language around the injected rules

### Benefits
- **No forgetting**: VSA rules are additive, not destructive
- **Selective activation**: Only relevant rules are injected
- **Interpretable**: You can see which rules were used
- **Fast**: VSA lookup is O(n) with SIMD, not O(n²) with attention

---

## The Merged Compression Stack

```
KNOWLEDGE BASE
    │
    ▼  Layer 1: LoRA Adapter (style/tone)
    │  Size: 1-100 MB (rank × hidden × 2)
    │  Purpose: probabilistic adaptation
    │
    ▼  Layer 2: SQA Cartridge (rules/logic)
    │  Size: 4 MB per 100K rules
    │  Purpose: deterministic knowledge
    │
    ▼  Layer 3: VSA Vocabulary (shared atoms)
    │  Size: ~100 KB
    │  Purpose: token ↔ hypervector mapping
    │
    ▼  Layer 4: Routing Table (query → mode)
    │  Size: ~10 KB
    │  Purpose: decide symbolic vs neural vs hybrid
    │
    ▼
TOTAL: 5-200 MB (vs 1-10 GB for full model)
```

---

## The Routing Algorithm

```python
def route(query, sqa_store, lora_adapter, threshold=0.7):
    """
    Route query to symbolic, neural, or hybrid path.
    """
    # Step 1: Encode query as VSA
    query_hv = encode_to_vsa(query)

    # Step 2: SQA lookup
    best_match, confidence = sqa_store.lookup(query_hv)

    if confidence > threshold:
        # MODE 1: Pure Symbolic
        # Rule found with high confidence
        return {
            "mode": "symbolic",
            "rule": best_match,
            "confidence": confidence,
            "use_lora": False,
        }

    elif confidence > threshold * 0.5:
        # MODE 3: Hybrid
        # Partial match — use rule as guidance, LoRA for language
        return {
            "mode": "hybrid",
            "rule": best_match,
            "confidence": confidence,
            "use_lora": True,
            "lora_style": "explain",  # LoRA generates around the rule
        }

    else:
        # MODE 2: Pure Neural
        # No rule match — rely on LoRA
        return {
            "mode": "neural",
            "rule": None,
            "confidence": 0.0,
            "use_lora": True,
            "lora_style": "generate",
        }
```

---

## The Injection Mechanism

### How VSA Rules Enter LoRA

```
Standard KV-Cache injection:
  1. SQA retrieves rule hypervector
  2. Decode hypervector → text
  3. Append text to context window
  4. LLM processes via attention

VSA-Guided LoRA injection (our approach):
  1. SQA retrieves rule hypervector
  2. Project hypervector → LoRA-compatible matrix
  3. Add to LoRA's B matrix: B += α × projection(rule_hv)
  4. Forward pass uses modified B
  5. Revert B after inference (non-destructive)
```

### The Projection

```
rule_hv (10,048 bits) → projection → (rank × hidden) matrix

Steps:
1. Reshape rule_hv: (10,048) → (rank × hidden/8)
2. Pad/truncate to match LoRA dimensions
3. Scale by confidence weight
4. Add to LoRA B matrix
```

---

## Benefits of the Merged Architecture

| Property | LoRA Only | SQA Only | **Merged** |
|----------|----------|---------|-----------|
| Accuracy (rules) | ⚠️ Probabilistic | ✅ 100% | ✅ 100% |
| Accuracy (style) | ✅ Good | ❌ None | ✅ Good |
| Forgetting | ❌ Catastrophic | ✅ None | ✅ None |
| Update speed | ⚠️ Retrain | ✅ O(1) | ✅ O(1) |
| Inference speed | ⚠️ Matrix multiply | ✅ XOR+POPCNT | ✅ Fastest path |
| Interpretability | ❌ Black box | ✅ Explicit | ✅ Explicit |
| Model size | ✅ Small | ⚠️ Large at scale | ✅ Compact |

---

## Implementation Plan

### Phase 1: Router (1 week)
- [ ] Build query → VSA encoding
- [ ] Build confidence threshold routing
- [ ] Test routing accuracy

### Phase 2: VSA-Guided LoRA (2 weeks)
- [ ] Build rule_hv → LoRA matrix projection
- [ ] Implement non-destructive injection
- [ ] Test on coding tasks

### Phase 3: End-to-End SQ-LM (2 weeks)
- [ ] Integrate SLM (Qwen-2.5-1B) + LoRA + SQA
- [ ] Build .sqac cartridge format
- [ ] Benchmark on real coding tasks

### Phase 4: Optimization (1 week)
- [ ] SIMD acceleration for VSA lookup
- [ ] Batch routing for multiple queries
- [ ] Memory optimization

---

## The SQ-LM Promise

```
SQ-LM v1 = Qwen-2.5-1B + LoRA (style) + SQA (rules)
  Size: ~1.1 GB (vs 2-10 GB for full fine-tune)
  Accuracy: 100% on rules, good on style
  Forgetting: Zero
  Update: O(1) for rules, retrain for style
  Inference: Fastest path for known rules
```

This is the **neuro-symbolic architecture** that the research community has been
building toward — but with VSA as the symbolic layer instead of traditional logic.
