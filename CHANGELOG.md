# Changelog

All notable changes to SQAC (Symbolic Query Addressable Cartridge) are documented here.

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
