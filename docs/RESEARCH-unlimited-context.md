# Academic Grounding — Research Sources & Links

Every design decision in SQAC traces to published work or to an experiment
run in this repo (`tests/`). Links verified as of 2026-09-06; no folklore
citations.

## Core HDC / VSA theory

| Claim used in SQAC | Source |
|---|---|
| BSC operators: XOR binding (self-inverse), majority-vote bundling, Hamming similarity | P. Kanerva, *Binary Spatter-Coding of Ordered K-tuples*, ICANN 1996 |
| HDC framing: quasi-orthogonality of random HVs in high dimensions; concentration of measure | P. Kanerva, *Hyperdimensional Computing: An Introduction…*, Adaptive Behavior 17(3), 2009 |
| Systematic VSA comparison; BSC vs FHRR vs HRR trade-offs | K. Schlegel, P. Neubert, P. Protzel, *A Comparison of Vector Symbolic Architectures*, arXiv:2001.11797 — https://arxiv.org/abs/2001.11797 |
| VSA as a computing framework for emerging (non-GPU) hardware; Turing completeness | D. Kleyko, M. Davies, E.P. Frady, P. Kanerva et al., *Vector Symbolic Architectures as a Computing Framework for Emerging Hardware*, Proc. IEEE 110(10), 2022 |
| Bundle capacity: **D ≥ 2n·ln(1/ε)** (used for the O(n)-wall and capacity estimates) | K.L. Clarkson, S. Ubaru, E. Yang, *Capacity Analysis of Vector Symbolic Architectures*, arXiv:2301.10352 (JAIR 2026) — https://arxiv.org/abs/2301.10352 |
| Comprehensive survey: models, data transforms, applications | D. Kleyko, A. Rachkovskij, E. Osipov, A. Rahimi, *A Survey on HDC aka VSA*, Part I arXiv:2111.06077, Part II arXiv:2112.15424 — https://arxiv.org/abs/2111.06077 , https://arxiv.org/abs/2112.15424 |

## VSA memory, retrieval & LLM integration

| Claim used in SQAC | Source |
|---|---|
| VSA ops work in transformer latent space; motivates external VSA routing for LLMs | C.J. Augeri, *Hypertokens: Holographic Associative Memory in Tokenized LLMs*, arXiv:2507.00002 — https://arxiv.org/abs/2507.00002 |
| Linearithmic (O(log²)) cleanup via Kronecker rotation products — evaluated and **rejected** for our scale (plain cosine/Hamming matched it in tests) | Liu et al., *Linearithmic Clean-up for Vector-Symbolic Key-Value Memory*, 2025 |
| SimHash for the semantic tier: sign-bit random projection preserves cosine geometry; P[bit agrees] = 1 − θ/π (Goemans–Williamson) | M. Charikar, *Similarity Estimation Techniques from Rounding Algorithms*, STOC 2002; Goemans & Williamson, JACM 1995 |

## Neural components

| Claim used in SQAC | Source |
|---|---|
| Semantic-tier embedding encoder (frozen, 384-d) | W. Wang, F. Wei, L. Dong, H. Bao, N. Yang, M. Zhou, *MiniLM: Deep Self-Attention Distillation…*, arXiv:2002.10957 (NeurIPS 2020) — https://arxiv.org/abs/2002.10957 ; model: `sentence-transformers/all-MiniLM-L6-v2` |
| TurboVec/TurboQuant (2–4 bit quantized retrieval; 5.1× storage, 40× search in prior experiments) | A. Zandieh et al. (Google), *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate*, arXiv:2504.19874 (ICLR 2026) — https://arxiv.org/abs/2504.19874 |

## VSA libraries

| Use | Source |
|---|---|
| Reference BSC implementation (prior experiments; torchhd verified against Kanerva) | M. Heddes et al., *Torchhd: An Open Source Python Library…*, JMLR 24 (2023); arXiv:2205.09208 — https://arxiv.org/abs/2205.09208 ; https://github.com/hyperdimensional-computing/torchhd |
| Community hub, software index | https://www.hd-computing.com/ |

## Where SQAC's novelty sits

Per the literature audit in `docs/THESIS.md` (Part I §1.4):

1. **VSA as an external RAG-alternative** — retrieval outside the model with
   XOR+POPCNT hot paths; Hypertokens operates *inside* latent space, SQAC
   deliberately operates *outside* the model. No published precedent found.
2. **Cartridge format where the VSA item memory is self-contained and
   portable** (seeded atoms ⇒ vocabulary-as-ABI, hot-swappable single file).
3. **Empirical honesty record** — negative results documented: positional
   permutation kills paraphrase recall (FissFus Exp C analog, reconfirmed in
   `tests/`), bundling reconstruction collapses (archived compression docs),
   PQ fails on binary vectors.

## Canonical repo-internal evidence

| Claim | Where proven |
|---|---|
| Recall/latency flat 100→10K facts | `tests/test_unlimited_context.py` |
| Synonym recall (x86 → ARM64) via SimHash tier | `tests/test_semantic_tier.py` |
| LLM answers private facts from cartridge; confabulates without | `tests/test_qwen_integration.py` |
| Latency: exact 5μs / fuzzy 21ms @ 10K | `tests/bench_sqac.py` |
