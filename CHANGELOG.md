# Changelog

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
