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

A frozen, never-trained model answered private questions correctly — because we handed it a 2.5KB memory file at runtime. Without it, it confidently made things up. **That gap is the entire product.**

---

## Table of Contents

- [Quick Start (30 seconds)](#quick-start-30-seconds)
- [What SQAC Actually Does](#what-sqac-actually-does)
- [Core Concepts](#core-concepts)
- [CLI Reference](#cli-reference)
- [Python API](#python-api)
- [Auto-Build & Realtime Tracking](#auto-build--realtime-tracking)
- [Fact Store](#fact-store)
- [Skill Store](#skill-store)
- [Context Offloader](#context-offloader)
- [Cartridge Rack](#cartridge-rack)
- [MCP Server](#mcp-server)
- [HTTP API Server](#http-api-server)
- [Performance](#performance)
- [Research](#research)

---

## Quick Start (30 seconds)

### Install

```bash
pip install -e .                  # core: numpy only
pip install -e ".[mcp]"           # + the MCP server for LLM integration
pip install -e ".[server]"        # + HTTP API server (FastAPI + uvicorn)
pip install -e ".[simd]"          # + Rust SIMD for 261x faster fuzzy scan
```

### Option A: Auto-build from your project (recommended)

Point SQAC at any project directory. It walks the files, extracts knowledge, and builds a searchable cartridge:

```bash
# Build a cartridge from your project
sqac init .

# Search it
sqac search "how does authentication work"
sqac search "what are the deploy targets"
sqac search "coding standards"

# Track changes in realtime
sqac track . --interval 5
```

### Option B: Manual (teach individual facts)

```bash
# Teach it something
sqac teach "our deploys are ARM64 only" --key deployment --db team.sqac

# Ask it back (try paraphrasing — "can we ship x86 images?")
sqac search "what do we deploy?" --db team.sqac

# Pack a whole rulebook from a text file
sqac pack rules.txt -o rules.sqac
```

### Option C: Python API

```python
from sqac import SqacStore

store = SqacStore()
store.add("Team Atlas maintains the payments service", key="payments ownership")
store.save("team.sqac")

# ...restart, reload, and it still knows
store = SqacStore.load("team.sqac")
store.search("who owns the payments system?")
# → ["Team Atlas maintains the payments service"]  confidence 1.0
```

---

## What SQAC Actually Does

SQAC turns **knowledge into a portable file** that any LLM can read at runtime.

Think of it as a USB drive for your LLM's brain:

1. **You put knowledge in** — teach facts, pack rulebooks, or auto-extract from code
2. **SQAC stores it** — in a binary cartridge with hyperdimensional addresses (XOR + popcount, no floats)
3. **LLM reads it at query time** — SQAC retrieves the relevant facts, injects them into the prompt
4. **The LLM answers from YOUR knowledge** — not from its training data, not from confabulation

The cartridge is **one file you own**: back it up, diff it, version it, email it, swap it per conversation.

---

## Core Concepts

### Cartridge

A `.sqac` file is self-contained portable memory:

- **Plaintext payloads** — the LLM reads text and confidence scores. It never sees a vector.
- **Hyperdimensional addresses** — binary 1024-bit keys, matched by XOR + popcount. No float math in the hot path.
- **Zero training, ever** — write a fact, it's stored in O(1). No embedding pipeline, no index rebuild.
- **Hot-swappable** — load a different cartridge mid-conversation. Team A's knowledge, then Team B's.

### What SQAC actually stores

SQAC isn't just a search engine. It stores **four kinds of knowledge**, each with a different job:

| Kind | What it is | What it does for the LLM |
|---|---|---|
| `fact` | A rule, decision, or piece of knowledge | "We deploy only to ARM64" — the LLM stops confabulating |
| `skill` | A reasoning **procedure** with trigger phrases | "When you need to find the odd coin..." — the LLM learns HOW to think |
| `doc` | Documentation or reference material | "The API accepts JSON payloads..." — the LLM knows your system |
| `turn` | A conversation exchange (offloaded) | "Q: what was that bug? A: The 504 was..." — the LLM remembers context |

The critical distinction: **facts tell the LLM what to know. Skills tell the LLM how to think.**

A skill card contains a **pure procedure** — no answers, just the reasoning pattern. When a frozen model retrieves a skill and applies it to a novel problem, that's proof of *application*, not memorization. Measured: a 7B model goes from 57% to **95%** on unit conversion tasks by retrieving the right skill card and evaluating externally (see [Thought Injection](#thought-injection--skill-based-reasoning)).

### Retrieval: how SQAC finds the right knowledge

Every query runs through three tiers, sharing one calibrated confidence scale:

| Tier | What it matches | Speed | Example |
|---|---|---|---|
| **Exact** | Verbatim keys | **5 μs** | `deployment-target` → 1.0 |
| **Lexical** | Typos, word overlap, trigram similarity | 21-56 ms @ 10K | "auth midleware" → the auth rule @ 0.72 |
| **Semantic** | Paraphrase, synonyms (static int8 model) | ~0.1 ms encode | "can we deploy on **x86**?" → the **ARM64** rule @ 0.64 |

For **skills**, retrieval goes further: the router matches the question against trigger phrases (multiple keys per skill), picks the best-matching skill card, and injects its procedure into the prompt. The model then follows the procedure instead of relying on whatever implicit strategy it happens to have.

**Fail-safe by design:** if nothing matches, SQAC returns empty and the LLM says "I don't know." It never confidently injects the wrong context.

### Thought Injection — skill-based reasoning

The strongest result in the codebase. SQAC doesn't just store knowledge — it can **change how a model reasons** by injecting the right procedure at query time.

**How it works:**

1. Store skill cards with pinned source→target operators (e.g., "multiply by 0.3048 to convert feet→meters")
2. Route the question to the right skill card via the retrieval tiers
3. Gate injection to high-confidence, single-operator cards only
4. Inject the procedure as an arithmetic expression, evaluate externally (bypasses the model's arithmetic weakness)

**Measured on 40 unit-conversion tasks (allam-2-7b):**

| Arm | What it does | Accuracy | Output tokens |
|---|---|---|---|
| A — baseline (direct answer) | Model answers from training | 57% | 14 |
| B — free CoT | Model thinks out loud | 40% | 161 |
| C — verbose injection | Full procedure injected | 45% | 182 |
| D — concise injection | Terse procedure, budget-forced | 48% | 16 |
| E — PoT (expression) | Emit expression, evaluate externally | 50% | 26 |
| **G — policy (operator + gated)** | **Pinned operator, confidence gate, external eval** | **95%** | **17** |

**G beats every arm on accuracy AND all budget-forced arms on token cost.** The 57% → 95% jump is +38pp at only +3 total tokens over baseline.

The key insight: **conditional injection of pinned single-pair operators, evaluated externally, is a reliable reasoning device for small models.** The skill card pins the exact conversion; the model just substitutes the number; the harness does the math. No confabulation possible.

See `experiments/thought_injection/` for the full experiment log, `docs/RESEARCH-thought-injection.md` for the literature grounding.

---

## CLI Reference

### `sqac init` — Auto-build from a project

Walks a directory and extracts knowledge into a `.sqac` cartridge. Zero configuration.

```bash
sqac init [root] [-o .sqac] [--semantic]
```

**What it extracts:**

| File type | What's extracted |
|---|---|
| `README.md` / docs | Section headings + bodies as facts |
| `*.py` | Module docstrings |
| `pyproject.toml` | Name, version, description, dependencies, scripts |
| `package.json` | Name, description, scripts, dependencies |
| `requirements.txt` | Pinned dependencies |
| `Makefile` | Make targets |
| `.github/workflows/*.yml` | CI job names |
| `Dockerfile` | Base image |
| `docker-compose.yml` | Service names |
| `.env.example` | Environment variable names |
| `AGENTS.md` / `CLAUDE.md` | Coding conventions |
| `skills.yaml` | Validated skill cards |

**Example:**

```bash
$ sqac init myproject
scanning /path/to/myproject ...
built .sqac/project.sqac (40 entries, 22.1 KB)
  kinds: fact=40
  state: .sqac/state.json
```

### `sqac search` — Query the memory

```bash
sqac search "query" [--db memory.sqac] [--k 3] [--kind fact] [--threshold 0.6]
```

**Options:**
- `--k N` — number of results (default: 3)
- `--kind fact|skill|doc|turn` — filter by knowledge kind
- `--threshold 0.7` — minimum confidence (default: 0.6)

**Example:**

```bash
$ sqac search "deploy target" --db team.sqac
[EXACT 1.000] Deploy only to ARM64; AMD64 images are not supported
          source: ops#deployment-target
```

### `sqac track` — Realtime project tracking

Polls the project directory, diffs against the previous state, and rebuilds the cartridge only when something changes.

```bash
sqac track [root] [-o .sqac] [--interval 5] [--log audit.jsonl] [--rack .rack]
```

**Options:**
- `--interval N` — poll interval in seconds (default: 5)
- `--log FILE` — append JSONL audit log (one event per line)
- `--rack DIR` — auto-register cartridge into a CartridgeRack directory
- `--rack-name NAME` — name for the cartridge in the rack (default: project dir name)

**Example:**

```bash
$ sqac track . --interval 2 --log audit.jsonl --rack .rack --rack-name myproject
audit log: audit.jsonl
tracking /path/to/project -> .sqac/project.sqac (every 2s)
rack: .rack (cartridge name: myproject)
  synced 14:30:01 (no change)
  + API.md#new-endpoints
  synced 14:30:03 (changed (1 source(s)))
  synced 14:30:05 (no change)
```

**Audit log format** (JSONL, one line per sync):

```json
{"ts": "2026-09-07T14:30:03", "added": ["API.md#new-endpoints"], "changed": [], "removed": [], "total_sources": 41, "dirty": true}
```

### `sqac teach` — Teach one fact

```bash
sqac teach "content" [--key "lookup key"] [--kind fact] [--db memory.sqac]
```

### `sqac pack` — Build from a text file

```bash
sqac pack rules.txt -o rules.sqac [--kind fact] [--name "my-rules"]
```

### `sqac stats` — Show cartridge stats

```bash
sqac stats --db memory.sqac
```

### `sqac serve` — Start the HTTP API server

```bash
sqac serve --dir ./memory --port 8420 --api-key sk-secret --rack
```

### `sqac dashboard` — Open the web dashboard

Starts the server and opens the dashboard in your browser:

```bash
sqac dashboard --dir ./memory --port 8420 --api-key sk-secret
```

### `sqac graph` — Open the 3D hyperdimensional graph

Opens an interactive 3D visualization where each entry is a node connected by VSA similarity. Features bloom glow, animated particles, force-directed layout, and hover tooltips.

```bash
sqac graph --dir ./memory --port 8420 --api-key sk-secret --threshold 0.50
```

**Graph visualization features:**
- **3D force-directed layout** — nodes repel, edges attract, cluster structure emerges
- **Bloom post-processing** — glowing nodes and edges with UnrealBloomPass
- **Animated particles** — flow along edges showing similarity connections
- **Hover tooltips** — shows entry key, content, kind, and source
- **Kind-colored nodes** — blue=fact, orange=skill, green=doc, purple=turn
- **Auto-rotate** — smooth camera orbit (toggle on/off)
- **Export** — save the graph as a PNG screenshot

---

## Python API

### SqacStore — The core

```python
from sqac import SqacStore

# Create and populate
store = SqacStore(semantic=True)  # enable paraphrase matching
store.add("Use pytest for all tests", key="testing framework", kind="fact")
store.add("Deploy only to ARM64", key="deployment target", kind="fact")
store.save("team.sqac")

# Load and search
store = SqacStore.load("team.sqac")
hits = store.search("what testing framework do we use", top_k=3)
for hit in hits:
    print(f"[{hit.confidence:.3f}] {hit.content}")
    print(f"  source: {hit.source}, mode: {hit.mode}")
```

### CartridgeRack — Multi-cartridge management

```python
from sqac import CartridgeRack

# Open a directory of cartridges
rack = CartridgeRack("memory/", routes={"fact": "team", "skill": "skills"})

# Write with automatic routing
rack.write_routed("Deploy only to ARM64", kind="fact")    # -> memory/team.sqac
rack.write_routed("SKILL weighted-index ...", kind="skill") # -> memory/skills.sqac

# Search across all cartridges
hits = rack.search("deployment target", top_k=5)
```

### ContextOffloader — Session memory

```python
from sqac import ContextOffloader

off = ContextOffloader("session.sqac", window=8)
off.observe("user", "Our CI fails with a 504 on deploy")
off.observe("assistant", "The 504 is the docker build timing out ...")

# After window overflows, exchanges are auto-distilled and offloaded
text = off.recall("what was that 504 about?")
# → "[0.76] [turns 0-1] Q: our ci is failing with a 504 ... A: The 504 comes ..."
```

---

## Auto-Build & Realtime Tracking

This is the **killer workflow** for teams: point SQAC at your project, and it stays in sync automatically.

### How it works

1. **`sqac init .`** — Scans your project, extracts knowledge from every file type, builds a `.sqac` cartridge. Writes a `state.json` to remember what it saw.

2. **`sqac track .`** — Polls the project every N seconds. When a file changes, it re-extracts only that file and rebuilds the cartridge. When a file is deleted, its entries are removed. The cartridge is always a faithful reflection of the project.

3. **`--log audit.jsonl`** — Every sync appends a timestamped record: what was added, changed, or removed. You get a complete audit trail of how your project's knowledge evolved.

4. **`--rack .rack`** — The built cartridge is also registered into a CartridgeRack directory, making it searchable alongside other cartridges (team knowledge, personal notes, session memory).

### What gets extracted

SQAC's extractors are designed to capture **what an LLM needs to know about your project**:

- **Architecture** from README headings and section bodies
- **API contracts** from docstrings and endpoint documentation
- **Dependencies** from pyproject.toml, package.json, requirements.txt
- **Build/deploy commands** from Makefile targets, CI workflows
- **Coding standards** from AGENTS.md, CLAUDE.md
- **Environment config** from .env.example (names only, never secrets)
- **Infrastructure** from Dockerfile, docker-compose.yml
- **Procedures** from skill YAML files

### Stress test results

On a realistic 25-file Python project (FastAPI + Celery + Redis):

| Metric | Result |
|---|---|
| Files scanned | 25 |
| Units extracted | 40 |
| Cartridge size | 22.1 KB |
| Search accuracy | 9/12 queries returned relevant results |
| File add detected | ✅ (4 new sources in SECURITY.md) |
| File modify detected | ✅ (1 added, 1 changed, 6 removed in README) |
| File delete detected | ✅ (base.html removed) |
| Audit log entries | Correct timestamp, source list, dirty flag |
| Rack sync | Cartridge copied, manifest updated, searchable |

---

## Fact Store

Point SQAC at your handbook, your runbook, your decisions log. Every LLM you use — Claude, GPT, a local Qwen — answers from *your* knowledge instead of confabulating.

```bash
# Pack a JSONL knowledge base
python -m sqac.ingest knowledge.jsonl -o company.sqac --semantic

# Or teach facts one by one
sqac teach "Payment processor is Stripe" --key payment-provider
sqac teach "Error budget is 0.1%" --key sli-slo
sqac teach "On-call rotates weekly, Team Alpha first" --key oncall
```

The semantic tier ships as a **9.8MB int8 model running in pure numpy** — no torch, no vector DB, no GPU.

---

## Skill Store

Store **procedures, not answers**. This is where SQAC goes beyond RAG: you're not just giving the LLM facts to quote, you're giving it **reasoning patterns to follow**.

Skill cards pair concrete trigger phrases with pure reasoning patterns:

```yaml
- name: weighted-index
  domain: logic
  difficulty: hard
  pattern: combinatorial
  content: |
    SKILL weighted-index: label items 1..N. Take i coins from item i.
    The total excess weight tells you which item has the defect.
  keys:
    - when exactly one of many items has a hidden property and you can weigh once
    - bags of identical items where one batch is heavier or lighter
```

**What makes this different from RAG:**
- RAG retrieves a passage the model can quote. SQAC retrieves a **procedure the model follows**.
- The skill card contains zero answers — so when a frozen model solves a novel puzzle, that's *application*, not memorization.
- Multiple trigger phrases per skill (multi-key routing) ensure the right procedure fires for different phrasings of the same problem.

```bash
# Validate skill cards (catches anti-patterns: abstract triggers, answer leakage, single keys)
python -m sqac.skills validate examples/logic_skills.yaml

# Suggest additional trigger keys
python -m sqac.skills suggest examples/logic_skills.yaml

# Pack into a cartridge (multi-key routing: one skill, many triggers)
python -m sqac.skills pack examples/logic_skills.yaml -o skills.sqac --semantic
```

**Measured on a 50-problem reasoning benchmark:**

| | Baseline | With skill store |
|---|---|---|
| 0.5B model | 23/50 | **31/50** |
| Architecture-domain problems | 1/4 | **4/4** |
| Hard problems (1.5B model) | 2/7 | **4/7** |

**The full thought injection pipeline** (pinned operators + external evaluation) pushes this further: 57% → **95%** on unit conversion tasks. See [Thought Injection](#thought-injection--skill-based-reasoning) above.

---

## Context Offloader

Conversations outgrow the window; re-deriving lost context costs thousands of reasoning tokens. The offloader flips the economics: **recall is ~100 tokens of input, re-derivation is thousands of tokens of compute.**

```python
from sqac import ContextOffloader

off = ContextOffloader("session.sqac", window=8)
off.observe("user", "Our CI fails with a 504 on deploy")
off.observe("assistant", "The 504 is the docker build timing out ...")

# Window overflows → exchanges are distilled & offloaded automatically
text = off.recall("what was that 504 about?")
```

**Design rules from measured failure:**

1. **Offload exchanges, not turns** — verbatim turn indexing recalls the user's question and shadows the answer (2/8 recall). Exchange indexing: 6/8. Denyxised keys: **8/8**.

2. **Denyxis at write time** — "that", "earlier", "you mentioned" appear in every back-reference and create ties. Keys carry entity anchors; temporal pointers get resolved, never stored.

3. **Fail safe** — no confident hit returns `""`: the model says "I don't have that in memory" instead of acting on a plausible-but-wrong exchange.

```bash
# Bulk: turn a transcript into a bucket
python -m sqac.offloader transcript.jsonl -o session.sqac

# Query
python -m sqac.offloader --recall "that flaky test fix" --db session.sqac
```

---

## Cartridge Rack

A rack owns several `.sqac` files by name and decides where each fact lands:

```python
from sqac import CartridgeRack

rack = CartridgeRack("memory/", routes={"fact": "team", "skill": "skills"}, default="facts")
rack.write_routed("Deploy only to ARM64", kind="fact")    # -> memory/team.sqac
rack.write_routed("SKILL weighted-index ...", kind="skill") # -> memory/skills.sqac

# Search across all cartridges
hits = rack.search("deployment target")
```

**Features:**
- Auto-mounts all `.sqac` files in a directory on open
- `write_routed` auto-creates routed cartridges on demand
- `create()` never wipes — loads existing cartridges, reset only with `overwrite=True`
- Reserved `session` bucket is never auto-mounted (keeps durable search clean)
- Powers the graduation pass (promote stable session memories into durable facts)

---

## MCP Server

The MCP server exposes SQAC as native tools for Claude Desktop, Cursor, Zed, and any MCP-compatible client:

```bash
# Install with MCP support
pip install -e ".[mcp]"

# Run the server
sqac-mcp
```

**Available tools:**
- `mem_search` — search the memory store
- `mem_write` — teach a new fact
- `mem_swap` — hot-swap to a different cartridge
- `mem_stats` — show cartridge statistics
- `session_recall` — recall from session memory
- `session_observe` — feed a turn into the offloader
- `rack_search` — search across the rack
- `rack_write` — write with automatic routing

---

## HTTP API Server

A production-ready HTTP server that exposes SQAC as a REST API. Any LLM client, any language, any framework can use it.

```bash
# Start the server
sqac serve --dir ./memory --port 8420 --api-key sk-secret

# Or with env vars
SQAC_DIR=./memory SQAC_API_KEY=sk-secret python -m sqac.server
```

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check (no auth). Returns SIMD, LZ4, uptime. |
| `GET` | `/stats` | Cartridge stats (entries, dims, kinds, encoder info). |
| `POST` | `/search` | Search with `{query, top_k, kind, threshold}`. |
| `POST` | `/teach` | Teach a fact with `{content, key, kind, source}`. |
| `POST` | `/compact` | Remove tombstones from a cartridge. |
| `GET` | `/cartridges` | List available cartridges. |
| `POST` | `/rack/search` | Search across all rack cartridges. |
| `POST` | `/rack/write` | Write with automatic kind routing. |
| `POST` | `/session/observe` | Feed a conversation turn into the offloader. |
| `POST` | `/session/recall` | Recall from session memory. |
| `POST` | `/graph` | Graph data: nodes + VSA similarity edges (JSON). |
| `GET` | `/graph` | 3D hyperdimensional graph visualization (HTML). |

**Authentication:** API key via `X-API-Key` header (or `SQAC_API_KEY` env). `/health` is always open.

**Example — curl:**

```bash
# Health check
curl http://localhost:8420/health
# {"status":"ok","simd":true,"lz4":true,"uptime_s":12.3}

# Search
curl -X POST http://localhost:8420/search \
  -H "Content-Type: application/json" -H "X-API-Key: sk-secret" \
  -d '{"query":"deployment target","top_k":3}'
# {"hits":[{"content":"Deploy only to ARM64","confidence":0.72,...}],"count":1}

# Teach
curl -X POST http://localhost:8420/teach \
  -H "Content-Type: application/json" -H "X-API-Key: sk-secret" \
  -d '{"content":"Error budget is 0.1%","key":"sli-slo"}'
# {"ok":true,"entries":5,"path":"./memory/memory.sqac"}
```

**Example — Python:**

```python
import requests

API = "http://localhost:8420"
HEADERS = {"X-API-Key": "sk-secret"}

# Search
r = requests.post(f"{API}/search", headers=HEADERS,
                  json={"query": "deployment target", "top_k": 3})
for hit in r.json()["hits"]:
    print(f"[{hit['confidence']:.3f}] {hit['content']}")

# Teach
r = requests.post(f"{API}/teach", headers=HEADERS,
                  json={"content": "Error budget is 0.1%", "key": "sli-slo"})
print(f"Entries: {r.json()['entries']}")
```

**Server options:**

```
sqac serve --dir ./memory --port 8420 --host 0.0.0.0 \
  --api-key sk-secret --rack --session session.sqac
```

- `--rack` — mount all cartridges as a CartridgeRack (enables `/rack/*` endpoints)
- `--session FILE` — enable session offloader (enables `/session/*` endpoints)
- `--dir` — cartridge directory (default: `.`)

---

## Performance

| Metric | Value |
|---|---|
| Exact lookup | **5 μs**, O(1) at any size |
| Fuzzy scan @ 10K rules (NumPy) | 21–56 ms |
| Fuzzy scan @ 10K rules (Rust SIMD) | **0.7 ms** — **261x speedup** |
| Semantic encode | **~0.1 ms**/query |
| Write (teach) | O(1), ~4 ms/fact |
| Cartridge @ 10K rules | 6.5 MB (semantic vectors) |
| lz4 compression | 15x at scale (payload JSON) |
| Memory (RAM) @ 10K rules | 146 MB |
| Runtime deps | numpy. That's it. |
| Rust SIMD deps | pyo3 + packed_simd2 (optional, auto-detected) |

### Rust SIMD: 261x faster fuzzy scan

SQAC ships an optional Rust extension (`sqac-simd/`) that accelerates the XOR+popcount inner loop:

- **AVX2 + POPCNT** on x86_64 (256-bit lanes, 32 bytes/iteration)
- **NEON** on AArch64 (128-bit lanes, 16 bytes/iteration)
- Falls back to NumPy when Rust isn't installed

```python
# Benchmark at 10K vectors (1024 dims)
# Python (pure XOR+popcount):  181 ms
# NumPy (batched):              21 ms
# Rust SIMD:                     0.7 ms  ← 261x faster
```

At 100K rules (which would be 18 seconds in Python), Rust SIMD brings it to **~70ms**.

Build it:

```bash
pip install maturin
cd sqac-simd && maturin build --release
pip install target/wheels/sqac_simd-*.whl
```

### Whole-system benchmarks

| Gate | Result |
|---|---|
| Exact recall (conf 1.0, correct content) | **445/445** across scales |
| Skill grouped top-1 | **50/50** |
| Back-reference recall (re-opened session) | **40/40** |
| Fail-safe: garbage query → no confident hit | held |
| Graduation | promotes, rerun idempotent |
| Durability | facts + session re-answer from disk after reload |
| Test suite | **97/97 passing** |

---

## What SQAC is honestly *not*

- **Not an unlimited context window.** Unlimited *storage* with constant-cost lookup: proven. Joint reasoning over every stored fact at once: not possible — the model sees what retrieval surfaces.
- **Not magic semantics.** Deep synonym gaps exist per encoder. The system fails safe when it can't bridge them.
- **Not distributed.** Fuzzy tiers are O(n); a Rust SIMD engine exists for 100K+ rules.

We publish our negative results too — they're part of the record.

---

## License

SQAC is licensed under the **Business Source License 1.1 (BSL-1.1)**.

**What this means:**
- ✅ **Free to use** — internal tools, embedded in your products, research, education
- ✅ **Free to modify** — fork it, customize it, contribute back
- ✅ **Free to distribute** — share it with your team, include in open-source projects
- ❌ **No competing SaaS** — you can't offer SQAC as a hosted API, managed database, or cloud service where SQAC is the primary value

**After September 7, 2030**, the license automatically converts to **Apache License 2.0** — fully permissive, forever.

See [LICENSE](LICENSE) for the full text.

---

## Research

SQAC stands on published work. Every link verified; no folklore citations.

**Core HDC / VSA theory:**
- Kanerva, *Binary Spatter-Coding of Ordered K-tuples*, ICANN 1996
- Kanerva, *Hyperdimensional Computing*, Adaptive Behavior 17(3), 2009
- Schlegel et al., *A Comparison of Vector Symbolic Architectures*, arXiv:2001.11797
- Kleyko et al., *Vector Symbolic Architectures as a Computing Framework*, Proc. IEEE 110(10), 2022
- Clarkson et al., *Capacity Analysis of VSA*, arXiv:2301.10352 (JAIR 2026)

**VSA memory & LLM integration:**
- Augeri, *Hypertokens: Holographic Associative Memory in Tokenized LLMs*, arXiv:2507.00002
- Charikar, *Similarity Estimation Techniques from Rounding Algorithms*, STOC 2002

**Neural components:**
- Wang et al., *MiniLM: Deep Self-Attention Distillation*, arXiv:2002.10957 (NeurIPS 2020)
- Minish Lab, [Model2Vec](https://github.com/MinishLab/model2vec) & `potion-base-8M`

**Where the novelty sits:**
1. VSA as an *external* RAG-alternative (Hypertokens works inside latent space; SQAC works outside the model)
2. A cartridge format where the VSA item memory is self-contained and portable
3. An empirical honesty record: all negative results documented in `docs/THESIS.md`

---

**Status:** research-grade, under active development. Core stable and tested (110/110 tests passing).

*Built as an implementation of the SQ thesis — see `docs/THESIS.md` for the full research narrative.*
