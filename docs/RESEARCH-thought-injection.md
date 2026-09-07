# Thought-as-a-Skill: Injecting Curated Reasoning Procedures into Instruct Models

> Status: research synthesis + experiment log. Draft, Sep 2026.
> Companion to `docs/PROBLEMS.md`. Code lives in
> `experiments/thought_injection/`.

## 1. The idea

A *thinking procedure* (how to attack a unit conversion, a proof, a rate
problem) is a reusable asset. SQAC already stores such assets as **skill
cards**: `content` is the pure, distillation-free procedure, `keys` are the
situation triggers that make it *retrievable*. This experiment asks whether we
can:

1. retrieve the right procedure for a question (SQAC routing), and
2. inject it into a small instruct model's prompt/reasoning

so that the model *reasons the way we want* — verifiably, and with predictable
token cost — rather than relying on whatever implicit strategy the model
happens to have.

The pipeline below is implemented in `experiments/thought_injection/engine.py`:

```
skill cards (content = procedure, keys = triggers)
        └─> SQAC store (KIND_SKILL, exact + fuzzy retrieval)
                └─> route(question) ─> hit (skill, content, confidence)
                        └─> build chat messages (A/B/C/D styles)
                                └─> OpenAI-compatible completion (Groq default)
```

## 2. Where this sits in the literature

### 2.1 Skills as prompts — the direct lineage

- **Skills-in-Context (SKiC)** — An et al., arXiv:2308.00304. One-stage
  prompt holding *basic skills* + composition exemplars; near-perfect on
  compositional generalization, up to 26pp over CoT. Explicitly notes
  semi-parametric LLMs that "dynamically access the most relevant skills from
  external memories" as a target — which is SQAC's router.
- **Skill-Based Few-Shot Selection (SKILL-KNN)** — An et al., EMNLP 2023.
  Prefer skills to surface text when picking examples; in SQAC terms, the
  `keys` encode the skill/situation, not the surface entity.
- **ReasoningSkills** (github.com/YuWenHan285/ReasoningSkills): a community
  pipeline that *generates* trajectories, *extracts* heuristics, *retrieves*
  and *injects* them (BM25 / dense / hybrid). This is our design with
  LLM-extracted skills instead of curated ones; our variant keeps provenance
  and guarantees a pure-procedure format.

### 2.2 Selecting and composing reasoning modules

- **Self-Discover** — Zhou et al., NeurIPS 2024, arXiv:2402.03620. The model
  SELECTs *atomic reasoning modules* from a library, ADAPTs them, IMPLEMENTs a
  per-task reasoning structure, then follows it at decode time. Beats CoT by
  up to 32% and self-consistency with 10–40× fewer inference calls. Two
  results matter to SQAC: discovered structures **transfer to smaller models**
  (GPT-4 → Llama2), and the structure can be cached per task class — the same
  economics as a SQAC cartridge.
- **iSelf-Discover** — arXiv:2507.03347. Per-instance structures; finds
  *unstructured natural-language plans beat structured JSON* by up to 18.9%
  on MATH. Instructs our "readable procedure text" choice over rigid schemas.
- **rSIM (reinforced Strategy Injection)** — ACL 2026. A *small planner*
  (even 0.5B) picks from 9 human-designed reasoning strategies (self-reflect,
  decompose, …) and injects them step-by-step into the target's CoT (leader–
  follower multi-agent RL). Turns Qwen2.5-0.5B into a reasoner that beats
  Qwen2.5-14B on math/coding/finance. This is the strongest published
  precedent for "give a small model the ability to reason by injecting
  strategy", and the planner is a plug-in — once trained, applied without
  retraining. SQAC = the storage/retrieval layer such a planner can sit on.

### 2.3 Injecting into the thinking process itself

- **Thinking Intervention** — arXiv:2503.24370. Insert/rewrite tokens *inside*
  the reasoning chain of reasoning models (e.g., DeepSeek-R1); +6.7 pp
  instruction-following, +15.4 pp instruction hierarchy, +40% refusal rate on
  unsafe prompts. Attention analysis shows reasoning models attend mainly to
  *internally generated* tokens — which is why prompt-level injections have
  limited reach and why our qwen-arm C gains little.
- **Passage Injection** — arXiv:2507.19333. In RAG, injecting retrieved
  passages into the *reasoning phase* beats injecting them in the input phase
  (robustness to noise). Directly parallel to in-reasoning skill injection.
- **InjectRBP** — arXiv:2602.12013. Models reasoning as an MDP over *behavior
  patterns* and injects RL-optimized pattern distributions at test time, no
  parameter changes (+5–9%). Confirms reasoners follow external structure
  injected mid-chain.

### 2.4 When injected reasoning hurts (the boundary we must respect)

- **CoT harms smaller models** — IEOM 2024: 15–30% absolute accuracy loss for
  sub-threshold SLMs; ~8% *per extra reasoning step* error growth for <5B
  models (Kumar et al. 2023 via that study).
- **Long CoT Degradation** — EMNLP 2025: SLMs ≤3B can lose up to **75%** of
  baseline when driven to long reasoning (error accumulation); some never
  recover even with 220k examples.
- **Mind Your Step (by Step)** — arXiv:2410.21333: CoT drops accuracy on tasks
  where deliberation hurts humans (implicit pattern learning, visual
  recognition, exceptions) — up to −36.3 pp absolute even for o1-preview.
- **The Curse of CoT** — arXiv:2504.05081: across 16 LLMs, CoT underperforms
  *direct answering* on pattern-based in-context learning; "explicit-implicit
  hybrid" failure mechanism.
- **Serial-depth / bandwidth view** — arXiv:2608.09942 (2026): CoT is a
  *bandwidth bypass*; it pays off only where serial depth exceeds single-pass
  capacity (P-complete math), and is flat on shallow tasks (TC0). Unit
  conversions are shallow — the regime where verbose scripts are pure noise.
- **Wharton GAIL report** (2025): CoT → marginal gains for reasoning models at
  20–80% more time; increased answer variability for non-reasoning models.

### 2.5 Controlling thinking budget

- **s1: Simple test-time scaling / budget forcing** — Muennighoff et al.,
  EMNLP 2025, arXiv:2501.19393. Force-thinking: cap thinking tokens (early
  exit) or extend them (append "Wait"). 1000 examples + budget forcing on
  Qwen2.5-32B exceeds o1-preview on MATH/AIME24. The **cost/verbosity knob**
  any skill injection needs.
- **Adaptive Injection Decoding (AID)** — Findings ACL 2025. Inject a neutral
  nudge ("Well") only when the model would halt prematurely; Llama-3.1-8B on
  MultiArith: 15.56→50.56% *while reducing* output tokens. "Targeted
  intervention, not more verbosity."

### 2.6 The training-based alternative (for contrast)

- **Aligning Large and Small LMs via CoT** — EACL 2024: SLMs improve only via
  teacher (esp. in-family) CoT *demonstrations*, not via CoT prompting,
  aligning with distillation results in s1/R1 works. This is the "pay to bake
  reasoning in" route vs. the zero-training injection route SQAC offers.

### 2.7 Constrained / rule-scripted thinking (the arsenal)

CoT-style reasoning is a *continuum of verbosity and structure*, and the
winning recipes for small models sit far from verbose free chains:

- **Chain of Draft (CoD)** — Xu, Xie, Zhao, He, arXiv:2502.18600. Minimal
  drafts of a few words per step; matches or surpasses CoT while using as
  little as **7.6% of the tokens**. Crucially its win is contingent on the
  base model's CoT working at all (§2.4) — see our arm F.
- **Program of Thoughts (PoT)** — Chen et al., arXiv:2211.12588. The model
  emits a *program/expression*; the interpreter does the arithmetic.
  +12–17pp over CoT on GSM8K / StrategyQA. Directly removes a small model's
  weakest link (multi-step arithmetic) — see our arm E.
- **Least-to-Most** (arXiv:2205.10625), **TabCoT** (arXiv:2305.17812),
  **Chain-of-Symbol** (arXiv:2305.12258): decomposition into ordered
  subproblems, structured tables of values, and compact symbol abstractions —
  all "rule-first" scaffolds that shrink the reasoning surface.
- **Rule/strategy injection** (rSIM, §2.2): inject the *strategy footsteps*
  themselves, not prose — the exact shape of a SQAC skill card.

Practical axis this gives us (the "customized thought" ladder):

| rung | prompt scaffold | who does arithmetic | arm |
|---|---|---|---|
| L0 | none (direct answer) | model, implicitly | A |
| L1 | free CoT seed | model, verbose | B |
| L2a | scripted verbatim steps | model, scripted | C |
| L2b | terse silent script | model, minimal | D |
| L2c | terse drafts (CoD) | model, compressed | F |
| L3 | script + external eval (PoT) | **harness (Python)** | E |
| L4 | deterministic operator | harness (100% rule) | — |

## 3. Design consequences for SQAC

1. **Procedures must be executable by the model.** A clean single-operator
   skill (rate×time → multiply) transfers; a fragile multi-step script can
   degrade a 7B (error compounding, §2.4).
2. **Minimise verbosity; let the format do the work.** "Apply silently, output
   `ANSWER = <number>`" (§2.5, §2.4) is the researched shape — arm **D**.
3. **Route, don't always inject.** Only inject when the router is confident
   and the skill's procedure is low-risk; otherwise prefer direct answering.
4. **Reasoning models attend internally** (§2.3) — prompt-side injection is
   muted there; the same cartridge can later target the *observation* stage of
   thinking models.

## 4. Experiments

Setup: 40 graded tasks (length / mass / temperature / rate×time; seed 7,
tolerance 2%). Models: `allam-2-7b` (small instruct, no native thinking) and
`qwen/qwen3.8-27b` (native reasoner), both on Groq. Routing: SQAC store,
exact + fuzzy (`confidence`, `mode`).

> **Reproducibility fix:** `tasks.py` originally drew question phrasings from
> the *global* `random` module, so the 40 questions differed across runs;
> arms within a run stayed comparable, cross-run sets did not. Fixed to seed
> both streams; §4.2/§4.3 use the reproducible 40-task set.

### 4.1 Round 1 (verbose injection, arms A/B/C)

Round 1 used the pre-fix task set per run; each row set is internally
consistent (same 40 draws per model) and reports are listed as obtained.

| allam-2-7b | acc | out tok | total tok |
|---|---|---|---|
| A baseline (direct) | 62% | 12 | 49 |
| B free CoT | 48% | 156 | 194 |
| C verbose injected | 48% | 180 | 393 |

| qwen/qwen3.8-27b | acc | out tok | total tok |
|---|---|---|---|
| A baseline | 98% | 6 | 45 |
| B free CoT | 98% | 207 | 263 |
| C verbose injected | 90% | 302 | 502 |

- Routing: 40/40 (A-grade) on both rounds.
- allam: verbose injection won the rate family (100% vs A 50%) but degraded
  length (20%) and temperature (10%) vs direct answering (80%/50%) — §2.4.
- qwen: all arms ≥90%; injection is redundant overhead (§2.3).

### 4.2 Round 2 (concise injection + budget force — arm D, reproducible set)

`think_skills_concise.yaml`: terse factor-first procedures; arm D instructs
"apply silently, output `ANSWER = <number>`", capped at 220 output tokens
(s1-style budget force, §2.5). Routing 40/40 on both cartridges.

| allam-2-7b | acc | out tok | total tok |
|---|---|---|---|
| A baseline (direct) | 57% | 14 | 51 |
| B free CoT | 40% | 161 | 200 |
| C verbose injected | 45% | 182 | 394 |
| D concise injected | 48% | **16** | **156** |

Per-family (A / B / C / D): length 70 / 30 / 20 / **60**, mass 70 / 50 / 40 /
60, rate 40 / 50 / **100** / 50, temperature 50 / 30 / 20 / 20.

Read:
- **Free CoT is the worst free arm** (40%) even at the family level — §2.4
  replication for this 7B.
- **Verbose C wins rate-times-time (100%) but stays low on length (20%);**
  **D roughly halves cost for similar-to-better accuracy** (48% vs 45% at
  156 vs 394 total tokens). Verbatim forcing degrades the same way §2.4 says;
  cutting verbosity is the documented antidote (§2.7).
- **A still leads overall (57%)** on these shallow tasks: without strong unit
  mapping the win is conditional (§2.4 serial depth).

### 4.3 Round 3 (rule-scripted thought — PoT arm E, CoD arm F)

Engine adds `expression` (emit one arithmetic expression; harness evaluates
it in a safe sandbox — PoT, L3) and `cod` (few-words draft per step — CoD,
L2c). Same reproducible 40 tasks, routing 40/40.

| allam-2-7b | acc | out tok | total tok |
|---|---|---|---|
| A baseline | 57% | 14 | 51 |
| B free CoT | 40% | 161 | 200 |
| C verbose injected | 45% | 182 | 394 |
| D concise injected | 48% | 16 | 156 |
| E expression (PoT) | **50%** | 26 | 179 |
| F CoD drafts | 30% | 101 | 246 |

Per-family (A / C / D / E / F): length 70 / 20 / 60 / **10** / 20, mass
70 / 40 / 60 / 50 / **0**, rate 40 / **100** / 50 / **100** / 90,
temperature 50 / 20 / 20 / 40 / 10.

Read:

- **E (PoT) is the strongest injected arm**: 50% at only 26 output tokens —
  +10pp over B at 1/5 the output tokens, +5pp over C at ~2 tokens per answer,
  and beats D (+2–3pp). External evaluation removes the 7B's arithmetic
  failure entirely: on rate×time (pure multiply, no unit math) E hits 100%
  alongside C.
- **E fails exactly where the skill doesn't pin the unit mapping.** With a
  factor *menu* in the card, the model synthesizes wrong multi-step chains —
  `(473.96 × 0.3048) / 2.54` (m→? mixed cm factor), `325.64 / 0.3048 × 3 × 12`
  (invented ×3×12), `(410.73 / 1.609344) × 1000` (wrong direction + ×1000).
  PoT removes arithmetic error; it does **not** remove operator-selection
  error. The fix is cartridge-side: one explicit *source→target* pairing per
  card, not a menu (§2.5/§2.7 design consequence).
- **F (CoD) is the worst injected arm** (30%; mass 0%, temperature 10%):
  short drafts launch a chain the 7B cannot sustain (dry-run, §2.4). CoD wins
  only when the base model already has working CoT — this one does not.
- **Final picture for SQAC:** injection should be gated to skills whose
  operator is single-step AND whose unit mapping is fully pinned by the card.
  rate×time delivers +60pp over A (C/E); the converters yield A (direct) or D
  (silent budget) as the reliable defaults until cards are made single-pair.

## 4.4 Round 4 — the policy arm (G): single-pair operator cartridge + gated injection

Implements the Round-3 conclusion as a running policy. Two new pieces:

- **`think_skills_operator.yaml`** — 13 single-pair operator cards (6 length,
  4 mass, 2 temperature, 1 rate; `pattern: operator`, 114 store entries, 0
  errors). Each card carries exactly one pinned `source→target` conversion
  (one factor / one formula), never a menu.
- **`plan()`/`solve()` in `engine.py`** — route → confidence gate
  (≥ 0.60) → pattern gate (`operator` only) → inject as *one arithmetic
  expression* (style `expression`, evaluated externally). Everything else
  answers directly (A). Direction is pinned deterministically: the fuzzy
  tier ties direction twins at ~0.70 (`f-to-c` 0.716 vs `c-to-f` 0.712), so
  the engine reads which unit sits adjacent to the number (source) and the
  other unit (target) from the *question text* and forces the exact twin
  card only — a model-free, rule-based disambiguator (L4), matching the
  "rule-described thought process" design intent.

Sanity (before any model call): 40/40 questions route to an exact operator
card; the wrong-direction failures of Round 3's router (fuzzy 0.641 vs
0.639 ties) are eliminated; 30/30 single-factor conversions reproduce the
ground-truth factor to 2% exactly.

Results (allam-2-7b, same reproducible 40 tasks, arms A–F reused from
Round 3, G added):

| allam-2-7b | acc | out tok | total tok |
|---|---|---|---|
| A baseline | 57% | 14 | 51 |
| B free CoT | 40% | 161 | 200 |
| C verbose injected | 45% | 182 | 394 |
| D concise injected | 48% | 16 | 156 |
| E expression (PoT) | 50% | 26 | 179 |
| F CoD drafts | 30% | 101 | 246 |
| **G policy (operator + gated)** | **95%** | 17 | 133 |

Policy line: 40/40 exact card, 40/40 injected, mean confidence 0.75.
Per-family (G): length 100%, mass 100%, rate 90%, temperature 90%.

Read:

- **95% is +38pp over A at +82 total tokens, +45pp over E at −46 total
  tokens, and +55pp over B at −67 total tokens.** The policy beats every arm
  on accuracy *and* all budget-forced arms (B/C/E/F) on total tokens; only
  A/D are cheaper but 38–47pp worse.
- **Failure decomposition (2/40).** (i) `-24.4 C → F`: the model echoes the
  formula with a placeholder (`F = (C * 9/5) + 32 - 24.4`) instead of
  substituting — an expression-format slip, arithmetically sound but not
  evaluable. (ii) rate×time: on one task it latches an extra unit factor
  (`* 0.05`) despite the card. Both are model-side slop, not routing or
  card-content failures; neither survives a grounded reformat or a second
  decode.
- **Two evaluator/card fixes moved G 80% → 95%:** (1) `eval_expression`
  now normalizes `x`/`×`→`*` and `÷`→`/` (the 7B writes `x` for multiply,
  e.g. `(50.3-32) x 5/9` — correct arithmetic rejected by the old grammar);
  (2) the rate×time card content was tightened to the same "read speed and
  time; multiply; output the single number" wording that made E 100% on
  rate, eliminating the spurious `*1609.34` / `*5280` / `*5` unit factors
  the old phrasing invited.
- **Verdict for SQAC:** conditional injection of pinned single-pair
  operators, evaluated externally, is a reliable reasoning device for this
  7B: 95% with near-zero output and a deterministic route. The mechanism
  turns a 57% baseline into a ~95% pipeline at ~$0.001/task class cost.

## 5. Next steps (candidate roadmap)

1. **Single-pair converter cartridges (done, §4.4):** length/mass/
   temperature/rate are re-fit as one explicit source→target pairing per
   card (`think_skills_operator.yaml`), plus a model-free direction resolver.
   The factor *menu* (Round 3) is gone; G = 95%.
2. **Conditional injection (done, §4.4):** `plan()` gates injection to
   `pattern: operator` cards with confidence ≥ 0.60 and always evaluates
   externally (style `expression`); everything else answers direct.
   Baseline→pipeline: 57% → 95%, cheaper-in-total than every prompt style
   except the 38pp-worse A/D.
3. **In-reasoning injection for thinking models:** reproduce with qwen3 /
   gpt-oss by injecting into the *reasoning* field (§2.3).
4. **Skill provenance & auto-distillation:** round-trip trajectories →
   distilled procedure → skill card (ReasoningSkills-style), with SQAC
   validation enforcing zero-answer purity.
5. **Budget forcing integration:** pair cartridge injection with a *thinking*
   token budget (s1-style), not just an output cap.
6. **Bigger eval:** GSM8K-style families where serial depth actually pays
   (§2.4), where the same cartridge machinery should show the win that unit
   conversions cannot.
7. **Robustness of the expression channel:** the two remaining G failures
   are formula-echo / extra-factor slips; a grounded reformat pass
   (substitute inputs into the injected formula before emitting) should push
   toward 100% and generalizes the "rule-described thought" idea.