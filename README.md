# SQAC — Give your LLM a memory it can carry in a file

**One `.sqac` file. Any LLM. Facts and skills that persist, survive restarts, and swap in milliseconds — no retraining, no database, no GPU.**

```
WITHOUT SQAC                          WITH SQAC
─────────────────────────────        ─────────────────────────────
Q: Who maintains the payments         Q: Who maintains the payments
   service?                              service?
A: "The Federal Reserve Bank of       A: "Team Atlas maintains the
    New York is responsible for           payments service."        ✅
    maintaining the payment
    system."                      ❌
```

A frozen, never-trained model answered private questions correctly — because we handed it a 2.5KB memory file at runtime. Without it, it confidently made things up. That gap is the entire product.

---

## The problem

Every LLM you use is amnesiac. It forgets your team's conventions, your product's rules, your customer's context — the moment the session ends. The existing fixes all cost you something:

| Approach | What it costs you |
|---|---|
| Fine-tuning | Weeks of work, per model, redone on every update |
| RAG stack | A vector database, an embedding service, infra to babysit |
| Bigger context window | Money per token, and still resets between sessions |

**SQAC takes none of those.** Your LLM's memory becomes a single file you own: back it up, diff it, version it, email it, swap it per conversation.

## What a cartridge actually is

A `.sqac` file is self-contained portable memory:

- **Plaintext payloads** — the LLM reads text and confidence scores. It never sees a vector.
- **Hyperdimensional addresses** — binary 1024-bit keys, matched by XOR + popcount. No float math in the hot path.
- **Zero training, ever** — write a fact, it's stored in O(1). No embedding pipeline, no index rebuild.
- **Hot-swappable** — load a different cartridge mid-conversation. Team A's knowledge, then Team B's, then your personal notes.

Three retrieval tiers run on every query, sharing one calibrated confidence scale:

| Tier | Catches | Example |
|---|---|---|
| **Exact** | verbatim keys | `deployment-target` → 1.0, in **5 μs** |
| **Lexical** | typos, shared words | "auth midleware" → the auth rule @ 0.72 |
| **Semantic** | paraphrase, synonyms | "can we deploy on **x86**?" → the **ARM64** rule @ 0.64 |

No confident garbage: if nothing matches, SQAC returns empty and the LLM says "I don't know." It fails safe by design.

## Quick start

```bash
# teach it something
python -m sqac.cli teach "our deploys are ARM64 only" --key deployment --db team.sqac

# ask it back (try paraphrasing — "can we ship x86 images?")
python -m sqac.cli search "what do we deploy?" --db team.sqac

# compile a whole rulebook into a cartridge
python -m sqac.ingest knowledge.jsonl -o company.sqac --semantic
```

```python
from sqac.store import SqacStore

store = SqacStore(semantic=True)
store.add("Team Atlas maintains the payments service", key="payments ownership")
store.save("team.sqac")          # memory is now a file on disk

# ...restart, reload, and it still knows
store = SqacStore.load("team.sqac")
store.search("who owns the payments system?")
# → ["Team Atlas maintains the payments service"]  confidence 1.0
```

The semantic tier ships as a **9.8MB int8 model running in pure numpy** — no torch, no vector DB, no GPU. The whole stack fits comfortably in **146MB of RAM with 10,000 rules loaded**.

## Two modes, one file

### 📚 Fact store — *what your team knows*

Point it at your handbook, your runbook, your decisions log. Every LLM you use — Claude, GPT, a local Qwen — answers from *your* knowledge instead of confabulating. Proven across three model families with the same cartridge file, zero changes.

### 🛠️ Skill store — *how your team thinks*

Store **procedures, not answers**. Skill cards pair concrete trigger phrases with pure reasoning patterns:

```yaml
- name: weighted-index
  domain: logic
  content: |
    SKILL weighted-index: label items 1..N. Take i coins from item i.
    The total excess weight tells you which item has the defect.
  keys:
    - when exactly one of many items has a hidden property and you can weigh once
    - bags of identical items where one batch is heavier or lighter
```

The stored skill contains **zero answers** — so when a frozen model solves the novel "12 bags of coins" puzzle after retrieving it, that's proof of *application*, not recitation. Measured on a 50-problem benchmark:

| | Baseline | With skill store |
|---|---|---|
| 0.5B model | 23/50 | **31/50** |
| Architecture-domain problems | 1/4 | **4/4** |
| Hard problems (1.5B model) | 2/7 | **4/7** |

Skills add value exactly at the model's failure boundary — and the bigger the consumer model, the more it gets out of the same cartridge.

### 🧠 Context offloader — *what you talked about*

Conversations outgrow the window; re-deriving lost context costs thousands of reasoning tokens. The offloader flips the economics: **recall is ~100 tokens of input, re-derivation is thousands of tokens of compute.**

```python
from sqac.offloader import ContextOffloader

off = ContextOffloader("session.sqac", window=8)
off.observe("user", "Our CI fails with a 504 on deploy")
off.observe("assistant", "The 504 is the docker build timing out ...")
# ... window overflows -> exchanges are distilled & offloaded automatically

off.recall("what was that 504 about?")
# '[0.76] [turns 0-1] Q: our ci is failing with a 504 ... A: The 504 comes ...'
```

Design rules that came out of measured failure, not intuition:
- **Offload exchanges, not turns** — verbatim turn indexing recalls the *user's question* and shadows the answer (2/8 recall). Exchange indexing: 6/8. Denyxised keys: **8/8**.
- **Denyxis at write time** — "that", "earlier", "you mentioned" appear in *every* back-reference and create ties. Keys carry entity anchors; temporal pointers get resolved, never stored.
- **Fail safe** — no confident hit returns `""`: the model says "I don't have that in memory" instead of acting on a plausible-but-wrong exchange.

The session bucket is a separate `.sqac` (entries stamped `kind=turn`), so it hot-swaps independently of durable knowledge: drop the bucket at session end, graduate the facts worth keeping.

Bulk mode:

```bash
python -m sqac.offloader transcript.jsonl -o session.sqac   # JSONL: role/content per line
python -m sqac.offloader --recall "that flaky test fix" --db session.sqac
```

The heuristic distiller is zero-dependency and measured at 8/8 top-1 back-reference recall; an LLM-backed distiller drops in via `ContextOffloader(distiller=...)` for harder conversations.

## The numbers

| Metric | Value |
|---|---|
| Exact lookup | **5 μs**, O(1) at any size |
| Fuzzy scan @ 10K rules | 21–56 ms (numpy XOR+popcount) |
| Semantic encode | **~0.1 ms**/query |
| Write (teach) | O(1), ~4 ms/fact |
| Cartridge @ 10K rules | 6.5 MB with semantic vectors |
| Recall scaling | 100% exact, flat 100 → 10,000 rules |
| Retrieval routing | **50/50** top-3 on the skill benchmark |
| Runtime deps | numpy. That's it. |
| Memory (RAM) @ 10K rules | 146 MB |

## What SQAC is honestly *not*

- **Not an unlimited context window.** Unlimited *storage* with constant-cost lookup: proven. Joint reasoning over every stored fact at once: not possible, by design — the model sees what retrieval surfaces.
- **Not magic semantics.** Deep synonym gaps exist per encoder ("db" vs "repository"-level). The system fails safe when it can't bridge them.
- **Not distributed.** Fuzzy tiers are O(n); a Rust SIMD engine exists for when 100K+ rules matter.

We publish our negative results too — they're part of the record (see Research, below).

## Roadmap

1. **MCP server** — `mem_search` / `mem_write` / `mem_swap` as native tools for Claude Desktop, Cursor, Zed. The server is ~40 lines over this tested store; the demo is *teach a fact → quit → reopen → still knows*.
2. **Rust XOR+POPCNT engine** — sub-5ms scans at 100K+ rules.
3. **KV-cache precompute** — per-model injection as the performance moat.

## Research sources

SQAC stands on published work. Every link verified; no folklore citations.

**Core HDC / VSA theory**

- P. Kanerva, *Binary Spatter-Coding of Ordered K-tuples*, ICANN 1996 — BSC operators: XOR binding, majority-vote bundling, Hamming similarity
- P. Kanerva, *Hyperdimensional Computing: An Introduction to Computing in Distributed Representations*, Adaptive Behavior 17(3), 2009 — quasi-orthogonality in high dimensions
- K. Schlegel, P. Neubert, P. Protzel, *A Comparison of Vector Symbolic Architectures*, arXiv:[2001.11797](https://arxiv.org/abs/2001.11797) — BSC vs FHRR vs HRR trade-offs
- D. Kleyko, M. Davies, E.P. Frady, P. Kanerva et al., *Vector Symbolic Architectures as a Computing Framework for Emerging Hardware*, Proc. IEEE 110(10), 2022 — VSA on non-GPU hardware
- K.L. Clarkson, S. Ubaru, E. Yang, *Capacity Analysis of Vector Symbolic Architectures*, arXiv:[2301.10352](https://arxiv.org/abs/2301.10352) (JAIR 2026) — bundle capacity bound D ≥ 2n·ln(1/ε)
- D. Kleyko, A. Rachkovskij, E. Osipov, A. Rahimi, *A Survey on Hyperdimensional Computing aka VSA*, Part I arXiv:[2111.06077](https://arxiv.org/abs/2111.06077), Part II arXiv:[2112.15424](https://arxiv.org/abs/2112.15424)

**VSA memory & LLM integration**

- C.J. Augeri, *Hypertokens: Holographic Associative Memory in Tokenized LLMs*, arXiv:[2507.00002](https://arxiv.org/abs/2507.00002) — VSA in transformer latent space; SQAC deliberately operates *outside* the model
- M. Charikar, *Similarity Estimation Techniques from Rounding Algorithms*, STOC 2002; Goemans & Williamson, JACM 1995 — SimHash: P[bit agrees] = 1 − θ/π
- Liu et al., *Linearithmic Clean-up for Vector-Symbolic Key-Value Memory*, 2025 — evaluated and **rejected** at our scale (plain Hamming matched it)

**Neural components**

- W. Wang et al., *MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers*, arXiv:[2002.10957](https://arxiv.org/abs/2002.10957) (NeurIPS 2020) — the opt-in high-fidelity semantic tier
- Minish Lab, [Model2Vec](https://github.com/MinishLab/model2vec) & the `potion-base-8M` model — the default semantic tier: static embeddings distilled from transformer teachers
- Tomaarsen & Minish Lab, [*Train 400x Faster Static Embedding Models with Sentence Transformers*](https://huggingface.co/blog/static-embeddings), Hugging Face blog, 2025 — the static-embedding recipe
- A. Zandieh et al. (Google), *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate*, arXiv:[2504.19874](https://arxiv.org/abs/2504.19874) (ICLR 2026) — quantized-retrieval direction

**Libraries & prior art**

- M. Heddes et al., *Torchhd: An Open Source Python Library for Hyperdimensional Computing*, JMLR 24 (2023), arXiv:[2205.09208](https://arxiv.org/abs/2205.09208) — reference BSC implementation used in early experiments
- [hd-computing.com](https://www.hd-computing.com/) — community hub and software index

**Where the novelty sits** — and where prior art ends:

1. VSA as an *external* RAG-alternative (Hypertokens works inside latent space; SQAC works outside the model — no published precedent found)
2. A cartridge format where the VSA item memory is self-contained and portable (seeded atoms ⇒ vocabulary-as-ABI ⇒ one hot-swappable file)
3. An empirical honesty record: positional permutation kills paraphrase recall, bundling reconstruction collapses, PQ fails on binary vectors — all documented in `docs/THESIS.md` and reproducible from `tests/`

---

**License & status:** research-grade, under active development. The core is stable and tested (25/25); the MCP server is the next milestone.

*Built as an implementation of the SQ thesis — see `docs/THESIS.md` for the full research narrative and `sqac/RESEARCH.md` for claim-by-claim sourcing.*
