# SynthQuant Adapter — Research Sources

> Curated bibliography for HDC/VSA/SQA research. Organized by relevance.

---

## Core Academic Papers

### VSA Foundations

| # | Paper | Authors | Year | Cited | Key Contribution |
|---|-------|---------|------|-------|-----------------|
| 1 | "Binary Spatter-Coding of Ordered K-tuples" | Kanerva, P. | 1996 | — | Foundational BSC: XOR binding, threshold bundling |
| 2 | "Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with High-Dimensional Random Vectors" | Kanerva, P. | 2009 | — | HDC textbook introduction |
| 3 | "A Comparison of Vector Symbolic Architectures" | Schlegel, Neubert, Protzel | 2022 | 225× | 11 VSA implementations compared |
| 4 | "VSA as a Computing Framework for Emerging Hardware" | Kleyko, Davies, Frady, Kanerva et al. | 2022 | 187× | IEEE reference. Turing completeness proof |
| 5 | "Capacity Analysis of Vector Symbolic Architectures" | Clarkson, Ubaru, Yang (IBM) | 2026 | 20× | D ≥ 2n·ln(1/ε) capacity bound |
| 6 | "A Survey on Hyperdimensional Computing aka VSA" | Kleyko et al. | 2022 | — | ACM Computing Surveys comprehensive survey |

### LLM-HDC Integration (Recent)

| # | Paper | Authors | Year | Key Contribution |
|---|-------|---------|------|-----------------|
| 7 | "Hypertokens: Holographic Associative Memory in Tokenized LLMs" | Augeri, C.J. | 2025 | HDRAM framework. VSA in transformer latent space |
| 8 | "Holographic Features in Language Models" | EmergentMind | 2026 | LLMs exhibit holographic encoding |
| 9 | "HDFLIM: Hyperdimensional Fusion for Language and Image Models" | Various | 2025 | Projects embeddings into shared HD space |

### Software & Tools

| # | Resource | URL | Key Contribution |
|---|----------|-----|-----------------|
| 10 | torchhd (JMLR 2023) | github.com/hyperdimensional-computing/torchhd | Python HDC library, BSC implementation |
| 11 | PyO3 | pyo3.rs | Rust-Python bridge |
| 12 | Rayon | github.com/rayon-rs/rayon | Rust parallelism |
| 13 | simd-popcnt | github.com/kimwalisch/simd-popcnt | SIMD bit counting |
| 14 | memmap2 | crates.io/crates/memmap2 | Rust mmap |
| 15 | mmap-io | crates.io/crates/mmap-io | Zero-copy file I/O |

### VSA Taxonomy

| # | Resource | Key Contribution |
|---|----------|-----------------|
| 16 | bandgap.org/vsas | VSA introduction series (Part 1-3) |
| 17 | hd-computing.com | HD/VSA community hub, history, software list |
| 18 | VSA_Toolbox (TU Chemnitz) | Matlab VSA implementation with experiments |

---

## Key Mathematical References

### Orthogonality (Concentration of Measure)
- **Theorem**: For x, y ∈ {0,1}^D random, Hamming distance ≈ D/2
- **Bound**: P(|d(x,y)/D - 0.5| > ε) ≤ 2·exp(-2ε²D)
- **At D=10,000, ε=0.05**: P ≤ 2·exp(-50) ≈ 1.9×10⁻²²

### Capacity (Clarkson et al. 2026)
- **Bundle capacity**: D ≥ 2n·ln(1/ε) for n items, error < ε
- **At D=10,048, ε=0.01**: n_max ≈ 1,090 items per bundle

### Binding Properties
- **BSC binding (XOR)**: self-inverse (a ⊕ b ⊕ b = a)
- **BSC binding output**: near-orthogonal to both inputs (with high probability)
- **BSC bundling (threshold)**: output similar to all inputs (up to capacity)

---

## Gap Analysis: What's NOT in the literature

| Gap | Status | Implication for SQA |
|-----|--------|-------------------|
| VSA as RAG replacement (outside transformer) | Novel — no published work | SQA architecture is genuinely new |
| .sqac binary cartridge format | Novel — no precedent | Invention, not based on prior art |
| mmap-based VSA module hot-swap | Novel — no published benchmark | The <5μs claim needs empirical validation |
| VSA + SLM KV-cache injection pipeline | Partially validated (HDRAM) but different architecture | Needs implementation and benchmarking |
| "100% hallucination elimination" via VSA | Overclaim — no system achieves this | Symbolic retrieval reduces but doesn't eliminate hallucination |
