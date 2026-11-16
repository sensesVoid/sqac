# SQAC replaces agent-facing `.md` files

![SQAC Banner](sqac.png)

**Give your LLM a memory it can carry in a file.** SQAC (Symbolic Query Addressable Cartridge) turns facts, skills, session context, agent personas, and secrets into portable `.sqac` cartridges that any LLM can query at runtime — no retraining, no vector database, no GPU required.

[![PyPI](https://img.shields.io/pypi/v/sqac)](https://pypi.org/project/sqac/)
[![Python](https://img.shields.io/pypi/pyversions/sqac)](https://pypi.org/project/sqac/)
[![CI](https://github.com/sensesVoid/sqac/actions/workflows/ci.yml/badge.svg)](https://github.com/sensesVoid/sqac/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-BUSL--1.1-blue)](LICENSE)

Your agent already reads `AGENTS.md`, `CLAUDE.md`, skill files, and handoff notes on every session — the same files, rewritten by hand, re-read in full, and never queried. SQAC swaps them for structured memory your model can *search* instead of *reread*.

```text
Ask your LLM: "Who owns the payments service?"

Without SQAC:  "The Federal Reserve Bank of New York maintains the
                payment system."        (confidently wrong — hallucination)
With SQAC:     "Team Atlas maintains the payments service."  (correct, from your data)
```

That gap is the entire product: a frozen, never-trained model answered private questions correctly because we handed it a 2.5 KB memory file. Also, unlike your model's training data, you can actually update it.

## Install

```bash
pip install sqac                  # core
pip install "sqac[mcp]"           # MCP server for agent clients
pip install "sqac[server]"        # HTTP API server
pip install "sqac[cloud]"         # cloud sync (R2/S3/Supabase/MinIO/B2/Spaces)
pip install "sqac[vault]"         # OS keychain for vault secrets
pip install "sqac[all]"           # everything
```

Python 3.10–3.13. Windows, Linux, macOS (x86_64 prebuilt wheels; aarch64/macOS wheels coming post-1.0).

## Quick start (60 seconds)

```bash
# Teach it a fact
sqac teach "We deploy to ARM64 only" --key deploy-target

# Ask it back — try paraphrasing; fuzzy + semantic tiers handle it
sqac search "can we ship x86 images?"
# → "We deploy to ARM64 only"  (confidence 0.72)

# Build memory from a whole project
sqac init .
sqac search "how does authentication work?"
```

If it doesn't know, it says so. SQAC would rather tell you "I don't have that in memory" than invent an answer — a level of humility your model could learn from.

## Pick your path

### I use coding agents (opencode, Claude Code, Codex, Cursor, Zed)

One memory dir, every CLI, every session. The memory follows you when you switch tools:

```bash
sqac mcp setup --host all          # prints wiring snippets for every supported CLI
sqac-mcp                           # runs the MCP server (add [mcp] extra first)
```

Then in any agent session: `mem_bootstrap` loads your working context (project, goal, last handoff), `mem_checkpoint` saves it at task boundaries. Agent personas and working state live in agent cartridges:

```bash
sqac agent init atlas --role "coding assistant"
sqac agent set identity "I am Atlas. I prefer tests before implementation."
sqac agent bootstrap               # the 4 KB "get to work" packet for a fresh session
```

→ Full details: [docs/AGENT.md](docs/AGENT.md), [docs/MCP.md](docs/MCP.md)

### I need to keep secrets out of my prompts

Regular cartridges are plaintext. Secrets and PII belong in the encrypted vault:

```bash
sqac vault init                    # AES-256-GCM, OS keychain or passphrase
sqac vault put openai --type secret
sqac vault exec --env OPENAI_API_KEY=vault:openai -- python script.py
```

MCP vault tools are reference-only: no tool ever returns a plaintext secret. Your model can *use* your API key without ever *seeing* it — like a valet with your car, minus the tipping.

→ Full details: [docs/VAULT.md](docs/VAULT.md)

### I want memory to survive a laptop-bucket-and-coffee incident

Sync cartridges to any S3-compatible store (Cloudflare R2 by default):

```bash
sqac cloud init --provider r2 --bucket my-memory
sqac cloud sync
```

Conflict-aware, explicit allowlist, never silently overwrites. → [docs/CLOUD.md](docs/CLOUD.md)

### I just want the file format

`.sqac` is a self-contained binary format: header + vocab + entries, versioned and forward-compatible. → [docs/FORMAT.md](docs/FORMAT.md)

## What lives where

| Cartridge | Holds | encrypted |
|---|---|---|
| `memory` (default) | facts, skills, docs, session turns | no (plaintext by design) |
| `agent` | persona + working state (identity, goals, decisions) | opt-in (SQAE) |
| `vault` | secrets, PII | yes (AES-256-GCM) |
| `secret` | secret *references* + metadata (never values) | yes (SQSE) |

Every entry is retrievable three ways: **exact** key lookup (O(1)), **lexical/fuzzy** matching (SIMD-accelerated), and optional **semantic** matching for paraphrases. No hit → no answer, never a confident guess.

## The CLI in one glance

```bash
sqac teach "..." --key k            # store one fact
sqac search "query"                 # retrieve it back
sqac init .                         # auto-build memory from a codebase
sqac track .                        # keep it in sync as the code changes
sqac agent ...                      # agent personas + working state
sqac vault ...                      # encrypted secrets/PII
sqac secret ...                     # secret refs/metadata (never values)
sqac cloud ...                      # S3-compatible sync
sqac report "bug title" -d "..."    # file a GitHub issue with diagnostics
sqac mcp setup --host <cli>         # wire memory into your agent
```

→ Full reference in the [CLI section](#the-cli-in-one-glance) above; MCP tool list in [docs/MCP.md](docs/MCP.md); HTTP endpoints in the server module.

## FAQ

**Is this RAG?** Not quite. RAG retrieves passages your model quotes; SQAC retrieves *facts your model uses* and *procedures your model follows* — with exact/lexical/semantic tiers, no vector DB, and a confidence gate that fails safe. If RAG is a search engine for documents, SQAC is a memory your agent can carry in its pocket.

**Does it need a GPU or embeddings service?** No. Retrieval is XOR + popcount on 1024-bit keys; the optional semantic tier runs a small static model in numpy. Storage is one file you can email to a colleague.

**Why not just keep using `AGENTS.md`?** Markdown is great for humans and terrible for retrieval: no keys, no confidence, no dedup, and your agent rereads the whole file every session. SQAC keeps the content, adds structure, and gives your model a 4 KB bootstrap instead of a 40 KB wall of text. Keep the markdown for yourself; give the cartridge to your agent.

**Can I see what it remembers?** Yes — `sqac stats`, `sqac search`, the dashboard (`sqac dashboard`), and the 3D graph view (`sqac viz`) for people who enjoy watching their memory as a glowing particle system.

**Is my data safe?** Memory cartridges are plaintext — treat them like a text file. Secrets live in vault cartridges (AES-256-GCM, keychain or passphrase). Secret references live in the Secret Domain — never values — and secret resolution happens only in the trusted backend. Cloud sync never uploads `agent`/`vault`/`secret` cartridges by default and never silently overwrites conflicts. See [docs/SECURITY.md](docs/SECURITY.md) for the full threat model, including what it does *not* protect against.

**What's the catch?** Fuzzy search is O(n) per scan — sub-100 ms up to ~100K entries with SIMD, beyond that shard your memory (DMS eviction can help). It's memory, not a reasoning engine: it retrieves the right context, your model still has to think.

## Status

- **Version:** 1.2.0 — 478 tests passing, 21 skipped
- **Repository:** https://github.com/sensesVoid/sqac
- **Issues:** https://github.com/sensesVoid/sqac/issues

## Contributing

PRs welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). Bug reports: `sqac report "title" -d "description"` files a GitHub issue with diagnostics attached.

## License

BUSL-1.1 (free to use, modify, distribute — no competing SaaS). Converts to Apache 2.0 in 2030. See [LICENSE](LICENSE).

---

*SQAC stands on hyperdimensional computing and vector-symbolic architecture research — see `docs/THESIS.md` and `experiments/` for the full record, negative results included.*
