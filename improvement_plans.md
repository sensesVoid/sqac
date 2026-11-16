# SQAC Improvement Plans — Road to v1.0

> **Current**: 0.2.0 in progress on `feat/0.2.0-simd-ci` (PR [#1](https://github.com/sensesVoid/sqac/pull/1), CI running)  
> **Target**: v1.0 (stable, documented, agent/vault/cloud-ready, cross-platform)  
> **Milestones**: 0.2.0 (SIMD+CI) → 0.3.0 (vault) → 0.4.0 (agent) → 0.5.0 (R2 sync) → 1.0.0 (hardening)

Roadmap aligned with `docs/superpowers/specs/2026-11-15-sqac-v1-design.md`.

---

## 0.2.0 — Bundled SIMD & CI/CD (in progress)

### 1. ~~Semantic tier / NumPy compatibility~~ ✅ **DONE** (v0.1.10, carried forward)
- Runtime deps now `numpy>=1.24` (NumPy 2 supported) + `tree-sitter-language-pack>=1.0,<1.24` (Python 3.10-safe cap).

### 2. Bundled Rust SIMD + cross-platform wheels ✅ **IMPLEMENTED** (verify in CI/release)
- `sqac._simd` bundled via setuptools-rust/pyo3 0.30 (abi3), `SQAC_SKIP_RUST=1` NumPy fallback, legacy `sqac_simd` fallback.
- cibuildwheel matrix: Linux x86_64, Windows AMD64 (`.github/workflows/wheels.yml`). aarch64 (Linux) and macOS wheels deferred to v1.0.0 (QEMU builds too slow).
- **Pending:** green Wheels/CI on PR #1 and first tagged release artifacts.

### 3. CI/CD, packaging, and release automation ✅ **IMPLEMENTED** (verify in CI)
- CI matrix (3 OS × py3.10–3.13) + NumPy-fallback job, Rust tests/clippy, ruff (`.github/workflows/ci.yml`).
- Trusted Publishing release workflow + Dependabot + `docs/PUBLISHING.md`.
- Single-sourced version (`0.2.0`), metadata fixes, Windows sidecar-lock/CRLF fixes, Python 3.10/3.11 autobuild fix.
- **Pending:** green PR checks, merge, optional TestPyPI dry-run, tag `v0.2.0`.

---

## 0.3.0 — Vault cartridge (next, per v1 design)

- **Reference-only secret vault**: `vault` cartridge type (header v5), AES-256-GCM, OS keychain + Argon2id passphrase, `vault:NAME` references.
- **No plaintext exfiltration**: `vault exec` (env injection) and `vault_http` proxy; no MCP `get`/`resolve`/raw export of values; audit log without values.
- **Rack isolation**: vaults never indexed in semantic/fuzzy tiers or shared search; keychain vaults local-only, `--portable` vaults sync as ciphertext.
- **CLI/MCP**: `sqac vault create/unlock/lock/set/list/rm/rotate/exec/export/import` + proxy tools.

---

## 0.4.0 — Agent cartridge (per v1 design)

- `agent` cartridge type: persona + working state, `set` upserts/singletons/append lists, type-based rack isolation.

## 0.5.0 — Cloudflare R2 sync (per v1 design)

- `StorageBackend` seam, local cache, explicit sync allowlist (memory/agents/vaults; continuity opt-in; never audit/config/session/checkpoints), conflict abort with force flags, ciphertext-only vault sync.

## 1.0.0 — Hardening

- Format/version fuzzing and migration tooling, Windows lock/audit hardening, docs site, security review, scale benchmarks, CI release gates verified on tags.

---

## Backlog (carry-over)

### 4. Cartridge Migration Tooling
**Need**: `sqac migrate <old_cartridge> <new_cartridge> --format-version 2`

**Scenarios**:
- Format version bump (v1 → v2)
- Schema changes (new fields in Entry)
- Compression algorithm change (lz4 → zstd)

**Design**:
```bash
sqac migrate session.sqac session_v2.sqac --from-version 1 --to-version 2
```

---

### 5. Windows Support & CI
**Gaps**:
- `watchdog` uses ReadDirectoryChangesW (different semantics than inotify)
- `mmap` vs `VirtualAlloc` for large cartridges
- Path separators in `.codegraph` / `.sqac-graph`

**Action**: Add Windows runner to CI, test `sqac track --watch`, `sqac graph watch`.

---

### 6. MCP Server Hardening
**Issues**:
- Connection cleanup on client crash (resource leak)
- No auth for remote MCP (currently local-only)
- Tool timeout handling (long searches block)

**Fixes**:
- Add `asyncio.wait_for` with 30s default timeout
- Implement token-based auth for remote MCP
- Add heartbeat/keepalive

---

### 7. Scale Benchmarks & Tiered Storage
**Current Limit**: ~25K entries per cartridge (semantic tier O(n) dot products).

**Targets**:
| Entries | Target Latency | Architecture |
|---------|----------------|--------------|
| 10K | <10ms | Single cartridge (current) |
| 100K | <50ms | CartridgeRack auto-split (10 shards) |
| 1M | <200ms | Tiered: hot (SIMD) + warm (semantic) + cold (disk) |

**Implementation**: Background compaction thread, tier promotion/demotion by access frequency.

---

### 8. Documentation Overhaul
**Missing**:
- `checkpoint` command (hidden easter egg)
- DMS tuning guide (lambda_decay, alpha, demote_threshold)
- Cross-CLI workflow (opencode → claude-code recovery)
- MCP wiring guide per CLI
- Architecture diagram (VSA tiers, DMS, CartridgeRack)

**Format**: `/docs/` with `mkdocs.yml`, auto-deploy to GitHub Pages.

---

## 🟢 Nice to Have (v1.0+)

### 9. Distributed CartridgeRack
- gRPC protocol for remote cartridge access
- Raft consensus for multi-node rack
- `sqac rack join <peer>`

### 10. Semantic Model Swap
- Pluggable encoder interface (MiniLM → E5 → BGE → custom ONNX)
- Model registry with version pinning
- `sqac model pull e5-base-v2`

### 11. Visual Query Builder
- Web UI for graph construction (drag-drop symbols)
- Export to `sqac graph` CLI commands
- Integration with VS Code extension

### 12. Cartridge Signing / Provenance
- `sqac sign <cartridge> --key cosign.key`
- `sqac verify <cartridge> --pubkey cosign.pub`
- Supply chain integrity for shared cartridges

---

## Next actions (0.2.0 completion)

- [ ] **PR #1 CI green**: `CI` (43678897713/latest) and `Wheels` (43678897752/latest) must pass after workflow fixes.
- [ ] **Merge** `feat/0.2.0-simd-ci` → `main` (on approval).
- [ ] **TestPyPI dry-run**: `Release` workflow dispatch with `testpypi=true` (requires Trusted Publishing + `testpypi` environment).
- [ ] **Release**: tag/push `v0.2.0` → PyPI publish + GitHub Release; post-release `pip install --no-cache-dir sqac==0.2.0` smoke.
- [ ] **Plan 0.3.0 (vault)**: break `docs/superpowers/specs/2026-11-15-sqac-v1-design.md` §9 into an implementation plan (cartridge v5, crypto, CLI/MCP, rack isolation, tests).

## Notes

- **v1 sequencing** (approved): vault → agent → R2 → hardening (see design doc). Encryption for secrets lives in the `vault` cartridge, not blanket encryption of memory cartridges.
- **Resolved in 0.2.0**: pyo3 upgraded to 0.30/abi3; `tree-sitter-language-pack` declared as a runtime dep (capped for py3.10).
- **Open debt**: format fuzzing/migration, docs site, security review of `credentials.py`/`injection.py`, tiered storage/scale, MCP remote auth/hardening, distributed rack, model swap, signing — tracked for 0.3.0–1.0.0/backlog above.

*Updated 2026-11-15 — reflects PR #1 / 0.2.0 progress*