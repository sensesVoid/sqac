# SQAC — hot-swappable VSA memory cartridges

Implementation of the pivot defined in `docs/THESIS.md` Part VI: a universal,
CPU-native memory tool for any LLM. Text in, text out, confidence surfaced —
the model never sees a vector.

## Structure

```
sqac/
├── encoder.py         BSCEncoder (lexical, zero deps) + MiniLMSimHashEncoder (torch semantic)
├── static_encoder.py  StaticSimHashEncoder — semantic tier in pure numpy (int8 potion-base-8M)
├── format.py          .sqac v1 binary format — header, packed keys, JSON payloads, ext blocks
├── store.py           SqacStore — 3 tiers: exact O(1) → lexical fuzzy → semantic fuzzy
├── ingest.py          dataset → cartridge (JSONL/text, field auto-detect, dedup)
└── cli.py             teach / search / pack / stats
tests/
├── test_sqac.py            12 unit tests (roundtrip, persistence, recall, tamper rejection)
├── test_semantic_tier.py   4 MiniLM-tier tests (calibration, synonym recall, fail-safe)
├── test_static_tier.py     9 light-tier tests (calibration, roundtrip, encoder compat)
├── test_qwen_integration.py  LLM integration: teach → persist → reload → answer
├── test_unlimited_context.py scaling + aggregation + multi-hop + distractors
└── bench_sqac.py           latency benchmark
docs/THESIS.md             the consolidated research thesis (single source of truth)
archive/                   prior research docs + code (frozen)
```

## Usage

```bash
# CLI
python -m sqac.cli teach "our deploys are ARM64 only" --key deployment --db team.sqac
python -m sqac.cli search "deployment target?" --db team.sqac
python -m sqac.cli pack rules.txt -o rules.sqac

# Dataset → cartridge
python -m sqac.ingest knowledge.jsonl -o kb.sqac --dedup 0.85 --semantic

# Python
from sqac.store import SqacStore
store = SqacStore(semantic=True)             # 3-tier retrieval
store.add("fact text", key="lookup key", source="handbook")
store.save("team.sqac")
store = SqacStore.load("team.sqac")          # hot-swap = just load another file
hits = store.search("deployment?")            # -> [{content, confidence, source, mode}]
```

## Retrieval tiers

| Tier | Catches | Example | Cost |
|---|---|---|---|
| 1. exact | verbatim key | full question → 1.0 | O(1), 5 μs |
| 2. lexical (BSC trigrams) | typos, shared words | "auth middleware" → 0.72 | O(n), 21ms @ 10K |
| 3. semantic (static int8, default) | synonyms, paraphrase | "x86?" → ARM64 rule @ 0.64 | O(n) + 0.1ms embed |
| 3'. semantic (MiniLM SimHash, opt-in) | finer synonym ranking | "x86?" → ARM64 rule @ 0.65 | O(n) + 10-20ms embed |

All tiers share the ~0.5 noise floor and one confidence scale. The default
semantic tier is **SimHash over a pure-numpy int8 static model** (potion-base-8M,
9.8MB artifact, no torch): P[bit agrees] = 1 − θ/π (Goemans–Williamson),
unrelated → ~0.50 (fails safe), paraphrase ~0.66, and it *catches* "db"→repository
at 0.68 where MiniLM missed at 0.56. MiniLM stays available via
`SqacStore(semantic=True, semantic_model="sentence-transformers/all-MiniLM-L6-v2")`
for maximum ranking fidelity (~+10% top-3 routing on large packs).

## Measured results (this machine, CPU-only)

| Metric | Value |
|---|---|
| Exact lookup | **5 μs** (O(1), constant) |
| Fuzzy scan @ 10K rules | **21 ms** (numpy XOR; was 142 ms pre-fast-path) |
| Write throughput | ~3.8 ms/fact |
| Cartridge size @ 10K rules | **3.97 MB** lexical-only · **6.53 MB** with semantic (v2 binary layout; was 5.4/10.9 MB in v1) |
| Semantic tier (static int8) | 9.8MB model, ~0.1ms encode, 118MB total process RSS |
| Skill routing (50-problem bench) | retrieval 50/50 (MiniLM) / 45/50 (static) · application 31/50 vs 23/50 baseline |
| Qwen2.5-0.5B + memory | 4/4 correct on private facts |
| Qwen2.5-1.5B + memory | 4/4 correct |
| Qwen3-1.7B + memory | 3/3 correct — **same cartridge file** |
| Baseline (no memory) | 0/4 — model confabulates ("Federal Reserve maintains payments") |

### "Unlimited context" verdict (`tests/test_unlimited_context.py`)

| Sub-claim | Result | Evidence |
|---|---|---|
| Unlimited **storage**, constant cost | ✅ **LEGIT** | Exact recall 100% flat 100→10K (architectural: independent traces); fuzzy recall flat ~95–97% (99.0 / 94.5 / 96.5 / 97.0 at 100/1K/5K/10K); exact latency O(1) forever |
| Multi-needle **aggregation** | ✅ PASS | 3 memories injected → model answers "3" correctly |
| **Multi-hop** reasoning | ✅ PASS* | 3-hop chain (pipeline→Atlas→Dana→Lisbon) via ReAct loop; *requires agentic pattern — naive single injection fails; whole-sentence query reformulation dilutes the bundle below threshold |
| **Distractor** robustness | ✅ PASS | Target found among 200 similar distractors |
| Unlimited **context** (holistic reasoning over everything) | ❌ **OVERCLAIM** | Model only ever sees what retrieval surfaces; tasks needing joint attention over all memories are out of scope by design |

**Honest framing**: unlimited *retrieval-augmented memory* with constant-cost reads — not an unlimited context window. For lookup-shaped tasks (facts, rules, procedures) the experience is indistinguishable from unlimited context. For reasoning-shaped tasks over the full corpus, the agentic loop is mandatory and holistic co-occurrence is impossible.

## Design notes

- **Payloads are plaintext**: the hypervector is an address, not a message.
  The LLM receives `{content, confidence, source}` — the interface problem
  (LLMs can't read raw vectors) is dissolved at the boundary.
- **Seeded SHA-256 atoms**: the vocabulary is deterministic, so cartridges
  are portable without shipping a vocab; fingerprint in the header rejects
  incompatible encoders.
- **No positional permutation**: measured (FissFus Exp C analog) — position
  shifts crush paraphrase recall; order-free trigram bundling is the right
  operating point for the lexical tier.
- **Fails safe**: semantic-only queries (no lexical overlap) return `[]`,
  so the LLM says "I don't know" instead of hallucinating.
- **Fuzzy is O(n)**: known wall (thesis Part VI). Exact stays O(1) forever;
  Rust SIMD engine is in `archive/rust/` for when 10K→1M scaling matters.

## Known limits (honest)

- The default static tier has a compressed similarity range (~0.50–0.70 vs
  MiniLM's ~0.49–0.78), which costs ~5/50 top-3 routing margin on large skill
  packs; MiniLM remains selectable for maximum fidelity.
- MiniLM-level synonym gaps remain ("db" vs "database" content-side similarity
  ~0.56); the dual key+content scan rescues most such cases via the key side.
- Fuzzy is O(n): ~21ms @ 10K per tier scanned; Rust engine in `archive/rust/`
  when 100K+ matters.
- Cartridge weight at scale: ~653 B/rule with semantic tier on (v2 binary
  vectors; the plaintext payload itself dominates). ~65 MB @ 100K rules.
- Superposition capacity per bundle not yet stressed beyond smoke scale.

## Next steps

1. MCP server wrapper (`mem_search` / `mem_write` / `mem_swap`)
2. Rust XOR+POPCNT engine port from `archive/rust/` for 100K+ rules
3. KV-cache precompute experiment (thesis Part VII, the later moat)
