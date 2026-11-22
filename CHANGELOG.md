# Changelog

## [1.4.0] — Unreleased (tagged on release)

### Canonical mem_id & mem_get
- **`sqac/memid.py`**: 26-char Crockford base32 ULID (48-bit ms timestamp + 80 bits `os.urandom`, stdlib only).
- `SqacStore`: assigns `meta["mem_id"]` on `add()`, backfills on `load()`, store-level `_id_index` (rebuilt in `load()`/`compact()`), `get(mem_id)` O(1), `add_with_mem_id()`, `delete_entry_by_mem_id()`. Ids are preserved across compact, shard split, save/load, and export.
- **Surfaces**: MCP `mem_get`, CLI `sqac get <mem_id>`, HTTP `GET /entries/{mem_id}` (scope-aware).
- **Every hit carries `mem_id`**: `mem_search`/`_hit_json`, `mem_recall_detailed` + `mem_bootstrap.recent` + `/session/recall` (`mem_ids` arrays), `/rack/search`, `_enriched_hit`, integrations-injected memory lines.

### Tasks & attention block
- **`sqac/tasks.py`** (new): per-project tasks cartridge (`kind=task`, new first-class kind) — `task_add` / `task_done` / `task_list`. `task_done` graduates the task into a durable `kind=fact` memory **preserving its mem_id**; finished tasks never appear in the attention block.
- **Kind=task excluded from ordinary search** (tasks are working memory, not knowledge — Q1); `task_list` and the attention block are the surface.
- **`sqac/attention.py`**: compact `[SQAC attention | turn N]` block with Current + Pending sections (caps 5/10, "+N more", ≤150-token budget).
- **Attention injection (`SQAC_ATTENTION_TURNS`, default 5)** — middleware, MCP `mem_observe`, HTTP `/session/observe`; always available via `GET /attention`.
- **Derived fallback**: with no explicit tasks, Current = continuity goal, Pending = `offloader.recent(5)` — the block never comes up empty.
- **Continuity fixes (D2)**: `_MemoryState` and `server.py` now call `ContinuityStore.load()`; server gains a ContinuityStore.
- Design docs: `docs/ATTENTION.md`, `docs/NORMALIZE.md`, `docs/INTEGRATIONS.md`, `docs/superpowers/specs/`, `docs/superpowers/plans/`.

## [1.3.1] — Unreleased (tagged on release)

### Wheels
- Expanded wheel matrix to include native **Linux aarch64** wheels (Ubuntu 24.04 ARM) alongside `x86_64` and Windows. macOS users install from the sdist (pure Python + optional Rust acceleration).

### Changed
- Bumped `actions/checkout` 4 → 7 across CI and wheel workflows (supersedes dependabot #4).

## [1.3.0] — 2026-11-19

### HTTP parity (`sqac.server`)
- **Agent cartridges over HTTP** (Tier 1A): `GET /agents`, `GET/POST/PUT/DELETE /agents/{name}`. Metadata is read header-only — encrypted (SQAE) agents expose name + `encrypted` flag only, never their contents. `POST /agents` runs `create_agent` (schema-v2 binding); `PUT` applies partial binding updates (`set_binding`); `DELETE` tombstones to `agents/.trash/` (default) or unlinks with `?hard=true`. Agent names are slug-validated and traversal-proof.
- **Rack introspection + reload** (Tier 1B): `GET /rack` (mounted stems with entries/kinds/percent_full/max_entries), `GET /rack/capabilities`, `POST /reload` (re-reads the rack dir without restart).
- **Kind-aware `/teach`** (Tier 1C): with a rack mounted and `kind` set, writes route by kind (`fact`→`facts`, `doc`→`docs`, `skill`→`skills`); otherwise default-memory behavior is preserved.
- **Rack search parity** (Tier 1D): `POST /rack/search` hits now carry `cartridge`/`key`/`kind` at top level plus a top-level `searched` list; new `POST /rack/trace` adds explicit `tier` (`exact`|`fuzzy`|`semantic`) and `latency_ms`. `store._hit` meta gains `key_norm`.
- **Session lifecycle** (Tier 1E): `GET /session/stats` (entries, offloaded exchanges, turns, live buffer, window, kinds), `POST /session/graduate` (auto-creates the target like `write_routed`), `POST /session/forget` (n oldest or all turns; tombstones only, never writes elsewhere; clears the live buffer on `all_turns`). `POST /session/observe` now debounces saves like the MCP layer (`SQAC_SESSION_FLUSH_TURNS`/`SECS`).
- **Auth tiers** (Tier 2G): `--readonly`/`SQAC_READONLY=1` makes every non-GET/HEAD/OPTIONS route return 403; `--scope`/`SQAC_SCOPE=stems` restricts cartridge-scoped calls to the allowed stems (prefix match covers shards and folder artifacts) — search, stats, writes, listings, and exports/imports are all scoped, fail-closed. New `GET /capabilities` reports the active mode.
- **Cartridge lifecycle** (Tier 2F): `POST /cartridges/{name}/export` → base64 tar.gz of the single `.sqac`; `POST /cartridges/{name}/import` → counterpart with traversal guard, 50 MB cap, and an `overwrite=true` guard; rack-aware paths (folder-routed cartridges export correctly).
- **System endpoints** (Tier 2H): `GET /version` (semver + feature flags) and `GET /describe` (machine-readable route index).

### Events (SSE + webhook)
- New **`GET /events`** Server-Sent Events stream on the HTTP server: `mem_write` / `mem_graduate` / `mem_swap` / `mem_checkpoint` (and their HTTP analogs: `/teach`, `/rack/write`, `/session/graduate`) emit **metadata-only** events (never content, never secret values) with an `event:` name and monotonic `seq`. `?once=true` sends the bounded backlog (256 events) and closes — good for probes.
- **Webhooks**: `SQAC_WEBHOOK_URL` (or `SQAC_WEBHOOK`) — every event is POSTed as JSON, best-effort with a 3 s timeout; failures never propagate into write paths.
- Event bus lives in `sqac/events.py` so both the MCP layer (`mem_*`) and the HTTP server emit identically; sandboxed read-only agents can subscribe (`/events` is GET).

### Normalization
- **`sqac.textnorm`** — pluggable pre-encoding normalization for the hybrid tokenizer (opt-in via `SQAC_NORMALIZE`, off by default, fingerprint-stable): `unicode_fold` (NFKD combining-mark strip), `stopwords:<lang>` (built-in `en`; `stopwords:file=...` for any language), `affix:<lang>` (light rule-based suffix stripping; `affix:file=...` for any language). Threaded through `SqacStore`/`SqacStore.load`/`CartridgeRack`/MCP + HTTP server (mounted cartridges included). User-supplied multilingual static embedding tables work today via `SQAC_STATIC_MODEL_DIR` (see [docs/NORMALIZE.md](docs/NORMALIZE.md)).

### Integrations
- **`sqac.integrations`** — drop-in `chat.completions.create` middleware for OpenAI-compatible clients (Ollama, vLLM, LiteLLM, OpenAI): hydrates the system prompt with confidence-gated memories (`threshold=0.60`, `top_k=5`), records both turns into the session offloader (debounced), and passes responses through untouched; anything unmodeled falls through to the wrapped client. `SQAC_MEM_DIR`/`memory_dir` resolution, `SQAC_NORMALIZE` honored.

### Fixed
- **Pre-existing**: `GET /stats`/`POST /teach` (and any default-cartridge call) returned **404** when `memory.sqac` was empty — `SqacStore` implements `__len__`, so an empty store was falsy under `if s:`.
- The HTTP server mounted the rack with **no routes**, so `rack/search` and kind-routed writes could never resolve kinds; it now mirrors sqac_mcp exactly (`routes`, `default="facts"`, `auto_load`, `folder_routing`, session bucket excluded by the rack itself).

### CLI (`sqac export` / `sqac diff`)
- **`sqac export <cartridge> [--format md|json] [--out FILE]`** — serialize a plaintext cartridge to human-readable Markdown or JSON (entries, kind, source, key, type, count). Encrypted cartridges (SQAE/SQSE — vault/secret/agent) refuse to export; payloads stay in-process by design. Console-safe output on legacy (cp1252) terminals.
- **`sqac diff <old.sqac> <new.sqac> [--format md|json]`** — added / removed / updated / **reclassified** (kind-changed) entries between two cartridge versions, matched by `key_norm`. Review memory changes in git PRs; audit agent learning across sessions.

### Notes
- `POST /rack/search` and `/rack/trace` accept an optional `cartridges` scoping via `SQAC_SCOPE`; `/agents` is out of scope by design (documented in `/describe` as applying to cartridges, not agents).
- Deliverable record stored in SQAC at key `proposal:http-parity` (kind `doc`, source `maintainer-proposal`).

## [1.2.2] — 2026-11-17

### Performance
- **Session-bucket save debounce**: `mem_observe` no longer rewrites `session.sqac` on every turn. Saves are now debounced to at most every **10 turns** or **60 seconds** (env `SQAC_SESSION_FLUSH_TURNS` / `SQAC_SESSION_FLUSH_SECS`), whichever comes first, and forced immediately on explicit persist points (`mem_save`, `mem_checkpoint`, `mem_graduate`, `mem_swap`). At 18K entries a per-turn full-session rewrite is O(cartridge size); this removes it from the per-turn hot path entirely. Set `SQAC_SESSION_FLUSH_TURNS=1` / `SQAC_SESSION_FLUSH_SECS=0` to restore v1.2.1 per-turn atomic saves.
- `ContextOffloader` now tracks `_since_flush`/`_last_flush` and exposes `should_flush(turns, secs)`; `save()` resets both counters.

### Behavior change
- A crash of the MCP host now loses at most the unsaved tail of the conversation (≤ 10 turns of recallable exchanges). The cross-CLI continuity packet (`continuity.json`) is unaffected (it is written on every `mem_observe`/`mem_checkpoint`); explicit persist points still flush synchronously. `mem_sparsify` and friends are unchanged.

### Fixed
- MCP stdio tests now spawn the server with an explicit isolated `SQAC_MEM_DIR` (the MCP SDK's default child environment does not inherit it). Previously `tests/test_mcp_continuity.py` and the new debounce test silently used the developer's real `~/.sqacm`; the `recent ≥ 1` assertion previously passed only against accumulated real-home state. It now fills the offload window hermetically (8 alternating turns) before asserting.

## [1.2.1] — 2026-11-17

### Performance
- `SqacStore.stats()` and `__len__` are now **O(1)**: incremental per-kind alive counters maintained at add/delete/compact/load. `rack.stats()`/`session.stats()` cost (~11/7 ms at 18K entries) is gone, including after writes.
- `mem_bootstrap` warm path no longer pays the full kinds-loop; measured ~112–124 ms → ~100 ms on a 18K-entry memory (remaining cost is the Rust similarity scan; SIMD-bound).
- `mem_observe` no longer recomputes a full `stats()` for the save description (~50–80 ms saved per turn at 18K entries).
- `search()` hot loop: inline eligible-entries filter (no per-entry `eligible()` calls), and `heapq.nlargest` instead of full `sort()` for tier results.
- `mem_write` persists **only the written cartridge** (crash-safe, shard-aware) instead of the entire rack — ~2/3 less disk I/O per taught fact at scale.
- Format writer: one joined buffer write (single syscall pass) instead of per-entry `write()` calls; JSON payloads serialized with compact separators (~10 % faster).

### Fixed
- Cartridge serialization of `session` save path no longer forces an all-entries stats recomputation per turn.

## [1.2.0] — 2026-11-16

### Added
- **Secret Domain cartridge** (`cartridge_type: secret`, `sqac secret init/list/put-ref/get-meta/rm/rename/list-domains`): encrypted (SQSE envelope, keychain or Argon2id portable mode) registry of secret *references and metadata only*. Credential-like material is rejected in every field; `secret:` indirection chains are rejected. Backends live under `~/.sqacm/secrets/<name>.sqac`.
- **SecretResolver** (`sqac.secret_resolver.SecretResolver`): backend-only resolution of `vault:NAME`, `vault:VAULT/NAME`, `secret:DOMAIN/NAME`, `provider:1password:*`, `provider:bitwarden:*`, `op://`, `bw://` refs. Fail-closed agent allowlist enforcement (`agent_context.secret_refs`), cycle guard, hash-chained audit events without values.
- **Secret Domain MCP tools** (reference-only): `secret_list`, `secret_meta`. No plaintext tools, ever.
- **Agent schema v2** (additive): `model_profile_id`, `secret_ref`, `secret_refs`, `allowed_tools` header binding + `sync_policy`. Schema 1 cartridges remain readable and upgrade on save. `AgentCartridge.set_binding(...)`, `sqac agent bind`, `sqac agent init --model-profile/--secret-ref/--secret-refs/--allowed-tools`.
- **Per-cartridge sync policy** enforced by the sync engine (authoritative): `extra.sync_policy` = `write_through | never | local_only`, fail-closed on unreadable/encrypted headers; `agent`/`vault`/`secret` default to `never`; `secrets/` is hard-denied; pull refuses to overwrite or create policy-denied cartridges; `sqac cloud status --show-denied` surfaces policy-denied local cartridges; `sqac vault set-policy` for opting in; `sqac agent bind --sync-policy` likewise.
- `sqac cloud sync --allow-local-only-vault` removed (it was never wired up).

### Security
- **Behavior change:** agent/vault/secret cartridges no longer sync by default, even under `--force-remote`/`--force-local`. Operators must set `sync_policy: write_through` explicitly per cartridge. No data loss — denial skips sync; nothing is deleted.
- Encrypted (SQAE/SQSE) cartridges can never have their policy verified remotely; they are treated as `never` on pull.
- Vault create/save always writes `sync_policy: never` (opt-in only via `sqac vault set-policy`).
- OpenAI project key (`sk-proj-*`) detection added to the credential scanner; high-entropy token heuristic added to Secret Domain field validation.

### Changed
- Cloud `SyncState` load/save now honor a module-level path override (testability; no behavior change).

## [1.1.0] — 2026-11-16

### Added
- Multi-vault support: named vaults under `~/.sqacm/vaults/`, `--vault` global flag, `vault init/list-vaults/use` subcommands, `vault:VAULT/NAME` reference syntax (backward compatible with `vault:NAME`).
- External secret providers: 1Password (`op`) and Bitwarden (`bw`) CLI wrappers for vault master keys. Zero new Python dependencies. Provider KDF mode stored in vault header.
- Agent-cartridge encryption: opt-in AES-256-GCM encryption at rest via `--encrypt` flag. `sqac agent encrypt/decrypt/key rotate` commands. Plaintext agents remain first-class.
- Blocking performance gate: `experiments/perf_gate.py` compares against `experiments/PERF_BASELINE.json` (5K-entry baseline), fails CI on >20% regression. Median-of-3 to reduce runner noise.
- Docs site: MkDocs Material on GitHub Pages (`mkdocs.yml`), deployed on release tags.
- Package hygiene: relocated `sqac/RESEARCH.md` → `docs/RESEARCH-unlimited-context.md`, moved benchmark scripts to `experiments/`.

### Security
- Cross-vault AAD isolation: ciphertext copied between vaults fails decryption.
- Provider CLI errors surface install hints and locked-state guidance.

## [1.0.0] — 2026-11-16

### Added
- Format hardening: caps on pb_len (64MB), decompressed payload (64MB), dims (8192), vocab (1M), entries (10M), token length (1024).
- Format fuzz tests (truncation, corruption, version mismatch, huge counts).
- Removed obsolete `format_version` attribute; always writes raw binary vectors (v2+).
- Deleted duplicated `sqac/graph/graph/` package.
- Adversarial credential detection tests (15 tests).
- Adversarial prompt injection detection tests (23 tests).
- Windows lock audit tests (5 tests).
- Cross-CLI integration tests (4 tests).
- `CONTRIBUTING.md` with development setup, testing, code style, PR process.
- `sqac report` CLI command for filing GitHub issues with diagnostics.
- Dedicated docs: `docs/VAULT.md`, `docs/AGENT.md`, `docs/CLOUD.md`, `docs/FORMAT.md`, `docs/MCP.md`, `docs/RELEASE.md`, root `SECURITY.md`.
- README reframed around "SQAC replaces agent-facing `.md` files" with a source-aligned CLI reference.
- Cross-platform capacity benchmark (`experiments/capacity_bench.py` runs on Windows; 100K-entry measurement).

### Security
- Hardened `read_cartridge`: rejects huge counts, caps payload sizes, handles decompression errors.
- Credential detector: adversarial tests for base64, split tokens, homoglyphs, zero-width chars.
- Prompt injection: adversarial tests for RTL overrides, tag chars, HTML comments, code blocks.

### Changed
- `cryptography>=42` is now a core dependency (vault is core feature).

---

## [0.5.0] — 2026-11-16

### Added
- Cloud sync: provider-agnostic S3 storage backend (R2, S3, Supabase, MinIO, B2, Spaces).
- Sync engine: push/pull/status/conflict detection with force flags.
- Cloud CLI: `init`, `status`, `sync` with provider/bucket options and dry-run/force flags.
- MCP tools: `cloud_status`, `cloud_sync`.

---

## [0.4.0] — 2026-11-16

### Added
- Agent cartridge: persona + working state (identity, instruction, preference, goal, decision, note).
- `agent` CLI: init, list, use, show, set, note, done, bootstrap, search.
- MCP tools: agent_bootstrap, agent_update, agent_done, agent_search, agent_list, agent_use, agent_show.
- `mem_bootstrap` prepends active agent packet (SQAC_AGENT=0 to disable).
- Type-based rack isolation: agent/vault cartridges excluded from default search.

### Privacy
- Agent state is plaintext in v1; docs flag it. Encryption can reuse vault crypto later.

---

## [0.3.0] — 2026-11-15

### Added
- Vault cartridge (v5 format): encrypted secret/PII storage with AES-256-GCM.
- `vault` CLI: init, put, list, rm, exec, audit, scan, import-env, rotate.
- MCP tools: vault_list, vault_meta, vault_http (host-allowlisted proxy), vault_audit.
- PII detection (email, phone, SSN, credit card, IP, IBAN) with Luhn validation.
- Hash-chained audit log (vault-audit.jsonl) with tamper evidence.
- Rack/search isolation: vault and agent cartridges excluded from default search.
- Reference-only access: no MCP tool returns plaintext; `vault exec` injects env vars with redaction.

### Security
- Keychain (default) and Argon2id passphrase modes; keychain vaults are local-only.
- Value cap: 64 KB per entry; names validated against PII patterns.
- HTTP proxy: HTTPS only, exact host allowlist, redirect policy, 1 MB response cap.

---

All notable changes to SQAC (Symbolic Query Addressable Cartridge) are documented here.

## [0.2.0] — 2026-11-15

### Added
- Bundled Rust SIMD extension (`sqac._simd`) built into the main wheel with abi3.
- GitHub Actions CI matrix (3 OS × Python 3.10–3.13), NumPy-fallback job,
  cibuildwheel builds, and Trusted Publishing releases.

### Changed
- `numpy<2` pin removed; NumPy 2 and Python 3.13 are supported.
- `sentence-transformers` removed from optional extras (unused).
- Version is single-sourced from `sqac/_version.py`.
- `tree-sitter-language-pack` is now a declared runtime dependency.

### Fixed
- `sqac/autobuild.py` parses on Python 3.10/3.11.
- Removed host-specific `target-cpu=native` from the Rust release profile.
- Windows cartridge writes use a sidecar lock, so atomic replace works.
- `extract_config` handles CRLF files.

## [0.1.10] — 2026-09-09
### Changed
- Semantic tier unblocked by pinning `numpy<2`; README benchmarks refreshed.

## [0.1.9] — 2026-09-09
### Changed
- Renamed `codegraph` to `graph`; improved Windows compatibility and `.gitignore` handling.

## [0.1.8] — 2026-09-09
### Added
- Major retrieval optimizations and a checkpoint recovery system.

## [0.1.7] — 2026-09-08
### Fixed
- Server session cartridge access and graph endpoint.

## [0.1.6] — 2026-09-08
### Security
- Credential detection; removed test secrets from the repository.

## [0.1.5] — 2026-09-08

### Added
- **Credential detection:** SQAC now rejects content matching common secret patterns (AWS keys, GitHub tokens, OpenAI/Anthropic keys, Stripe keys, private keys, passwords, connection strings, JWTs) with a clear `CredentialError`. This prevents accidental storage of credentials in plaintext `.sqac` files.

### Security
- **SQAC is not a secrets manager** — README now documents this boundary prominently. `.sqac` files are plaintext with no encryption at rest. Use environment variables or a secrets manager for credentials.

## [0.1.4] — 2026-09-08

### Added
- **sqac ast** — AST-level code structure intelligence. Parses source files with tree-sitter, extracts symbols (functions, classes, methods) and edges (calls, imports, inheritance). Query API: `ast_explore` (definition + callers + callees), `ast_blast` (blast radius analysis), `ast_callers`, `ast_callees`, `ast_search` (fuzzy name matching). Supports Python, JavaScript/TypeScript, Go, Rust with regex fallback.
- **sqac ast CLI** — `sqac ast init`, `sqac ast explore <symbol>`, `sqac ast blast <symbol>`, `sqac ast callers <symbol>`, `sqac ast callees <symbol>`, `sqac ast search <query>`, `sqac ast stats`, `sqac ast watch`.
- **MCP tools** — `ast_init`, `ast_explore`, `ast_blast`, `ast_callers`, `ast_callees`, `ast_search`, `ast_stats` available via the SQAC MCP server.
- **File watcher** — `sqac ast watch` polls for file changes and rebuilds the graph automatically.

## [0.1.3] — 2026-09-08

### Added
- **Shard compaction:** `rack.compact("facts")` merges all shards of a cartridge back into a single file, drops tombstones, and removes shard files from disk. Keeps search fast and file count manageable after heavy delete workloads.
- **MCP server uses folder routing** (`facts/`, `skills/`, `docs/` subdirectories) and auto-split (`max_entries=25,000`) by default.

### Improved
- **KV cache relief documentation:** README now highlights the 200x–5,000x KV cache reduction as the primary benefit — SQAC recalls only top-k tokens instead of stuffing the entire memory into context.

## [0.1.2] — 2026-09-08

### Changed
- **SQAC acronym formalized:** SQAC now stands for **Symbolic Query Addressable Cartridge** — reflecting the VSA encoding (Symbolic), retrieval tiers (Query Addressable), and portable file format (Cartridge).
- Updated README title, PyPI description, and CLI help text.

## [0.1.1] — 2026-09-08

### Changed
- **README overhaul** for PyPI: install section now shows `pip install sqac` as primary method.
- MCP tools list updated with `mem_bootstrap`, `mem_checkpoint`, `mem_sparsify`.
- Documented server security hardening: HMAC auth, path traversal guard, XSS escaping, threshold isolation, atomic writes, per-cartridge locks.
- Fixed stale test counts (261/261).

## [0.1.0] — 2026-09-08

Initial PyPI release.

### Added
- **Cross-CLI continuity (AgentBridge):** `mem_bootstrap`, `mem_checkpoint`, `mem_sparsify` MCP tools. Seamless handoff between opencode, Claude Code, Codex, Cursor, Zed, and others via shared `continuity.json`.
- **MCP setup CLI:** `sqac mcp setup --host <cli>` generates per-CLI wiring snippets.
- **Dynamic Memory Sparsification (DMS):** Utility-based eviction policy with salience scoring, aging, and access-count promotion.
- **Prompt injection detection:** Trust scoring, risk levels (safe → critical), pattern matching for direct overrides, system impersonation, data exfiltration, unicode tricks, leetspeak bypass.
- **3D graph visualization:** Force-directed layout, bloom post-processing, animated particles, K-means clustering, node dragging, search, focus.
- **Prometheus metrics endpoint** + web dashboard with real-time stats.
- **HTTP API server:** Auth, teach-search pipeline, graph endpoint, compact, rack endpoints, session endpoints.
- **Rust SIMD fuzzy scan:** 261x speedup for XOR+popcount inner loop (AVX2/NEON).
- **Production-grade features:** File locking, compact, query cache, content sanitization.
- **Auto-build & realtime tracking:** `sqac init .` + `sqac track . --interval 5` with audit logging.
- **Skill store:** Multi-key routing, validation, trigger suggestion, thought injection pipeline.
- **Regression test suite** covering search, mutation, rack persistence, and server helpers.

### Fixed
- **Exact-hit search short-circuit:** Previously returned 1 result even with `top_k > 1`; now seeds the results and fuzzy tiers fill the rest.
- **Delete dropping sibling exact mapping:** Deleting one entry of an exact-key pair no longer drops the live sibling's O(1) mapping.
- **Meta/trust alias leaks:** `Hit.as_dict()` and `Hit.trust`/`meta` now return copies, not live references to the store.
- **Dashboard `html` variable shadowing:** Local variable `html` shadowed the `html` module import, causing `NameError` in `html.escape()` calls.
- **Per-request threshold leak:** `/search` threshold changes now restore the original value after the request.
- **`rack.save(None)` vs `save([])`:** Empty list now correctly persists nothing; `None` persists all.
- **Autoload failures silenced:** Cartridge files that fail to load now emit `logging.warning()` instead of being silently dropped.

### Security
- **Timing-safe API key comparison:** `hmac.compare_digest` replaces `==` for API key verification.
- **Path traversal guard:** Cartridge names are validated; `/`, `\`, `..`, `.` are rejected.
- **Per-cartridge write locks:** Concurrent `/teach` requests to the same cartridge are serialized.
- **Thread-safe lazy init:** `_ensure_state()` uses a lock to prevent race conditions.
- **XSS escaping:** Dashboard HTML output escapes all dynamic content with `html.escape()`.
- **Atomic writes:** `state.json` and rack manifests use tmp+replace to prevent corruption on crash.
