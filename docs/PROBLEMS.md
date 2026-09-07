# SQAC: Ten Problems It Solves (And Where It Doesn't)

> **Date**: 2026-09-07
> **Status**: Market positioning research — quantified pain points with primary sources.
> **Method**: All claims backed by cited sources; gaps noted honestly.

---

## The Core Problem SQAC Addresses

LLMs suffer from a fundamental architectural limitation: they encode knowledge in weights, which are expensive to train, impossible to update in-place without forgetting, and opaque to inspect. Every alternative—RAG, fine-tuning, prompt engineering—carries substantial operational overhead that scales with deployment count. SQAC attacks this differently: a VSA-based `.sqac` cartridge file that loads via mmap, retrieves facts with XOR+POPCNT in microseconds, and updates in O(1) without touching the model.

This document maps SQAC's capabilities to ten concrete, evidenced pain points in production LLM deployment.

---

## 1. Session Amnesia

**The problem.** LLMs discard all context when a session ends. Users must re-establish their requirements, preferences, and accumulated knowledge every conversation. This is the single most-cited user complaint in conversational AI: "my assistant doesn't remember me."

The problem is deeper than UI polish. Research on long-context degradation shows that even within a session, information retrieval follows a **U-shaped curve**: LLMs attend most strongly to the beginning and end of their context window, with middle-positioned information significantly neglected. GPT-3.5-Turbo shows pronounced degradation for facts placed in the middle third of context (Liu et al., 2023; arXiv:2307.03172). This means session-level memory is not just lost at session boundary—it is unreliable even during the session.

CXS (Cross-session Continuity System) quantifies the engineering cost: dedicated memory stores, priority-based retrieval, and privacy-preserving session decay—each adding infrastructure that must be deployed, monitored, and scaled per user (Zenodo:17782686).

**How SQAC maps to it.** A cartridge persists across sessions. Rules and facts loaded into the VSA registry remain available on next session start via mmap—zero re-injection, zero context-window slot consumption. The dict-first fallback guarantees exact recall of user-specific rules regardless of session boundary.

**Gaps / honest limit.** SQAC stores discrete rules, not conversational nuance. The continuous, implicit preferences a user develops through dialogue ("I prefer concise responses") require a different storage mechanism (e.g., MemGPT's episodic layer or Zep's temporal graph). SQAC is not a general-purpose session memory.

---

## 2. Context-Window Economics

**The problem.** Long prompts are expensive. Prompt caching (where available) reduces cost but introduces write fees and TTL complexity. The math is severe:

- **Anthropic**: Cache write = 1.25× base token price; cache read = 0.10×. The write surcharge means each cache (re)fill costs more than a plain input read; cached reads are near-free.
- **OpenAI**: Write = 1.25×, read = 0.5× for GPT-4.1.
- **Gemini**: No write surcharge — cache reads bill at 0.10× input price, and explicit caches additionally bill an hourly storage fee (~$4.50 per 1M tokens/hour on Pro). The "4.5× write surcharge" sometimes quoted conflates that hourly storage rate with a per-request write cost.

Without caching, every long-prompt request pays the full input token cost. "Lost in the Middle" (Liu et al., 2023; arXiv:2307.03172) shows that even with long contexts, retrieval accuracy degrades when relevant information is not at the beginning or end—meaning you pay for tokens the model effectively ignores.

**How SQAC maps to it.** Rules retrieved from the cartridge and injected at inference time occupy only the relevant subset of context. Instead of injecting 10,000 rules into a prompt (burning tokens and hitting the U-shaped degradation curve), SQAC retrieves the 5–50 relevant rules in <60ms (fuzzy tier) and injects only those. Cost: O(relevant rules), not O(all rules).

**Gaps / honest limit.** SQAC retrieval latency (21–56ms for fuzzy) adds to end-to-end response time. For latency-critical applications, this may matter. Also, the injection mechanism must be carefully designed to place retrieved rules at the beginning or end of context to avoid the middle-position degradation.

---

## 3. RAG Operational Burden

**The problem.** Production RAG requires: document chunking strategy, embedding model selection and hosting, vector database operations, index maintenance, chunk-retrieval ranking, and re-ranking. Each layer introduces failure modes.

Chunking is a fundamental source of information loss: splitting at fixed boundaries breaks semantic units, while semantic chunking requires its own embedding pass. Embedding drift occurs as models are updated—re-indexing the entire corpus after an embedding model upgrade is a significant operational cost. The RAG pipeline described by unixy.io requires continuous monitoring of retrieval quality, with manual tuning of chunk size, overlap, and retrieval thresholds.

The MCP specification (modelcontextprotocol.io/specification/2024-11-05) defines the agent-tool interaction protocol — resources, prompts, tools, and sampling — but does not address the operational burden of maintaining the retrieval infrastructure itself.

**How SQAC maps to it.** A `.sqac` file replaces the entire pipeline: chunk → embed → index → query → re-rank becomes "write cartridge, mmap it." Retrieval is O(1) for exact matches (dict) and O(n) with XOR+POPCNT for fuzzy (VSA). No embedding server, no vector DB, no chunking strategy. The cartridge is the index.

**Gaps / honest limit.** SQAC requires knowledge to be structured as discrete rules. Unstructured documents (PDFs, natural-language articles) must be pre-processed into rule form—SQAC does not ingest raw text. This is a higher upfront cost than "throw everything in a vector DB."

---

## 4. Fine-Tuning Cost and Catastrophic Forgetting

**The problem.** Fine-tuning LLMs is expensive (GPU hours, data labeling) and fundamentally fragile. The core issue is catastrophic forgetting: updating weights to encode new knowledge overwrites previously learned patterns.

arXiv:2406.04836 demonstrates that LLM fine-tuning operates in a loss landscape where:
- Weight updates for new knowledge overwrite parameters critical for prior knowledge
- The flatness of the loss landscape around a task determines generalization—but fine-tuning pushes weights toward narrow minima that hurt other tasks
- Recovery strategies (EWC, replay buffers) mitigate but do not eliminate the tradeoff

Each fine-tune is a bet: gain new capability, lose old capability. This makes iterative knowledge updates (daily policy changes, evolving product specs) extremely expensive—each update requires full re-validation.

**How SQAC maps to it.** VSA memory is additive by construction. Adding a new rule is O(1) pointer insertion; removing is O(1) deletion. The base model is never touched. SQ-LM v2 demonstrated confidence ≈ 1.0 on stored facts with zero forgetting of prior rules. There is no loss landscape to navigate; there is no weight update to corrupt.

**Gaps / honest limit.** SQAC stores discrete rules, not behavioral adaptations. You cannot "fine-tune" a model's tone, style, or reasoning patterns via cartridge—LoRA or similar is required for that. SQAC and LoRA are complementary, not competing.

---

## 5. Knowledge Freshness

**The problem.** Models have a knowledge cutoff date. Post-cutoff information is absent unless the model is retrained or given retrieval access. Even with RAG, maintaining fresh knowledge requires continuous ingestion pipelines that detect, extract, and index new information.

The cost is not just technical—it is organizational. Every knowledge source requires a pipeline owner, monitoring, and freshness guarantees. Stale information in a vector DB is worse than no information: the system confidently returns outdated answers.

**How SQAC maps to it.** Hot-swapping cartridges means knowledge updates are file-level operations. Replace a cartridge, and the next session loads fresh facts. No re-indexing, no re-training, no embedding pipeline refresh. For regulated domains (compliance rules, legal requirements), this is a direct operational win.

**Gaps / honest limit.** Someone still needs to create the updated cartridge. SQAC solves the "load fresh knowledge" problem, not the "detect what changed" problem.

---

## 6. Multi-Model Portability

**The problem.** Organizations increasingly run multiple LLMs (GPT-4 for reasoning, Claude for long-context, Gemini for multimodal, local models for privacy). Knowledge encoded in a fine-tuned model's weights is locked to that specific model architecture and must be re-learned (re-trained or re-fine-tuned) for each target model.

PortLLM (arXiv:2410.10870) quantifies this: training-free "portable patches" that transfer knowledge between models show that cross-model knowledge sharing is both desirable and technically challenging—existing approaches require alignment training or adapter layers.

**How SQAC maps to it.** A `.sqac` cartridge is model-agnostic. The same cartridge injects rules into GPT-4, Claude, Gemini, or a local Qwen-2.5-1B via the MCP server or direct prompt injection. The knowledge travels with the file, not the weights. This is the MCP ecosystem's native advantage: any MCP-compatible client can consume cartridge facts.

**Gaps / honest limit.** Injection quality varies by model. Some models are better at incorporating retrieved facts into their generation; others may ignore injected context or hallucinate around it. The cartridge guarantees retrieval accuracy; it does not guarantee generation accuracy.

---

## 7. Privacy and Air-Gapped Deployment

**The problem.** Many organizations (healthcare, defense, finance) cannot send data to external API providers. On-premises deployment is an active research area precisely because such privacy-sensitive users require it: SOLID (Huang et al., EMNLP 2025) documents that strict regulations prohibit uploading sensitive data to third-party APIs, that local deployment shifts security risk to vendors (GPU/CPU-resident models invite theft via distillation and memory-side attacks), and that Trusted Execution Environments carry prohibitive computational overhead.

RAG systems that rely on cloud-hosted embedding models (OpenAI, Cohere) violate air-gapped constraints even if the LLM itself is local.

**How SQAC maps to it.** SQAC is CPU-native. The Rust engine uses XOR+POPCNT (no GPU required). The cartridge file is self-contained—no external API calls, no embedding service, no network dependency. Deployable in a VM with no internet access.

**Gaps / honest limit.** The cartridge still needs to be created somewhere. If the knowledge originates from external sources, the ingestion pipeline must respect the air-gap boundary.

---

## 8. Hallucination on Factual Queries

**The problem.** LLMs hallucinate plausible-sounding but incorrect facts. The Vectara Hallucination Leaderboard (github.com/vectara/hallucination-leaderboard) shows hallucination rates of 1.8%–6%+ across leading models on summarization tasks. For factual Q&A, the problem is more severe: models generate confident answers to questions about topics not well-represented in training data.

HalluGuard (arXiv:2510.00880) frames this as a RAG problem: retrieval quality directly determines hallucination rate. Poor retrieval → hallucination. The causal chain is clear: the model generates from what it retrieves; if retrieval fails, hallucination follows.

**How SQAC maps to it.** On rule-covered queries, SQAC provides deterministic retrieval (dict-first) or high-confidence fuzzy match (VSA tier). The retrieved fact is injected directly into context, giving the model a concrete anchor rather than forcing it to "guess" from training data. For pure-lookup paths (no generation needed), hallucination is eliminated entirely.

**Gaps / honest limit.** SQAC reduces hallucination on rule-covered queries; it does not eliminate hallucination globally. The SLM still generates probabilistically around injected facts. Honest framing: SQAC narrows the hallucination surface to the generation gap between retrieved fact and natural-language output.

---

## 9. Interpretability and Audit

**The problem.** Neural network decisions are opaque. When a model produces a wrong answer, diagnosing whether the error came from (a) incorrect knowledge, (b) poor reasoning, or (c) generation artifacts is nearly impossible with weight-based knowledge.

The MCP specification defines tool interfaces but does not address knowledge provenance. Zep (arXiv:2501.13956) demonstrates temporal knowledge graphs for agent memory—but graph traversal is expensive and the underlying knowledge is still opaque.

**How SQAC maps to it.** Every rule in a cartridge is inspectable. You can dump the cartridge, read its entries, verify which rule produced a given retrieval, and trace the causal chain from fact to output. The cartridge is its own audit log.

**Gaps / honest limit.** VSA hypervectors themselves are opaque—the "meaning" of a vector is encoded in its pattern, not readable by humans. The dict-index layer is fully transparent; the VSA fuzzy layer requires reverse-mapping through the codebook.

---

## 10. Onboarding and Knowledge-Worker Context

**The problem.** Knowledge workers (engineers, analysts, domain experts) spend significant time context-switching between tools and re-establishing project context. "Where was I?" costs are substantial: Parnin & Rugaber's analysis of 10,000 developer IDE sessions found programmers spent 15–30 minutes re-establishing context after interruptions, and Mark et al.'s workplace studies measured ~23 minutes to return to a task after a switch. For teams, this scales multiplicatively.

**How SQAC maps to it.** A cartridge can encode project-specific rules, coding conventions, API references, and domain knowledge. Loading the cartridge at session start provides instant domain context without prompt engineering. For engineering teams: a cartridge per project, loaded on demand.

**Gaps / honest limit.** Cartridge quality depends on what was curated. Garbage in, garbage out—SQAC does not solve the knowledge-curation problem, only the knowledge-delivery problem.

---

## Where SQAC Does NOT Fit

1. **Unstructured document ingestion.** SQAC requires pre-structured rules. If your data is raw PDFs/HTML, you need a processing pipeline first. A vector DB with raw embedding ingestion is simpler for unstructured corpora.

2. **Behavioral/style adaptation.** Cartridges store facts, not behaviors. Tone, style, and reasoning-pattern adjustments require LoRA or similar. SQAC is complementary to, not a replacement for, parameter-efficient fine-tuning.

3. **Real-time streaming data.** Cartridge updates are file-level operations. For continuous data streams (sensor feeds, live analytics), a streaming database is more appropriate.

4. **Semantic similarity at scale (>100K rules).** The VSA fuzzy tier is O(n) with SIMD acceleration, but at very large scales, approximate nearest neighbor (ANN) indexes with learned embeddings may outperform for pure similarity search. SQAC's strength is the hybrid (exact + fuzzy), not pure fuzzy at scale.

5. **Multimodal knowledge.** SQAC is text/rule-oriented. Image, audio, or video knowledge requires modality-specific processing before cartridge ingestion.

---

## Sources

1. **Lost in the Middle** — Liu et al. (2023). arXiv:2307.03172. U-shaped attention curve; middle-positioned info retrieval degrades significantly.
2. **MemGPT** — Packer et al. (2023). arXiv:2310.08560. Virtual context management for LLMs.
3. **Revisiting Catastrophic Forgetting in LLM Tuning** — Li, Ding, Fang & Tao (2024). arXiv:2406.04836; Findings EMNLP 2024. Loss-landscape flatness correlates with forgetting severity; fine-tuning overwrites prior knowledge.
4. **PortLLM** — (2024). arXiv:2410.10870. Training-free portable model patches for cross-model knowledge sharing.
5. **Zep** — (2025). arXiv:2501.13956. Temporal knowledge graphs for agent session memory.
6. **Programmer, Interrupted** — Parnin & Rugaber (2011). 10,000 IDE sessions; ~15–30 min context reconstruction. See also Mark, Gudith & Klocke (2008) — ~23 min to return to an interrupted task.
7. **HalluGuard** — (2025). arXiv:2510.00880. RAG-based hallucination mitigation; retrieval quality → hallucination rate.
8. **Vectara Hallucination Leaderboard** — github.com/vectara/hallucination-leaderboard. Competitive benchmark; reported rates vary by model and evaluation version (roughly low-single-digit to ~6%+).
9. **SOLID** — Huang et al. (2025). A Middle Path for On-Premises LLM Deployment. EMNLP 2025, aclanthology.org/2025.emnlp-main.420 (arXiv:2410.11182). Privacy-driven on-prem deployment and model-theft risk; not an infra-cost quantification.
10. **MCP Specification** — modelcontextprotocol.io/specification/2024-11-05. Agent-tool interaction protocol.
11. **CXS Architecture** — Zenodo:17782686. Cross-session continuity; memory infrastructure cost.
12. **Provider Pricing** — tokenpricing.dev (secondary) spot-checked against provider pricing pages. Cache write/read multipliers and storage fees.
13. **RAG Pipeline Complexity** — unixy.io (secondary blog). Chunking, embedding drift, operational burden.
14. **SQAC Benchmarks** — `[repo: tests/]`. Internal SQAC performance data.
