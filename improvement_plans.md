# SQAC Improvement Plans — Road to v1.0

> **Current**: v0.1.8 (production-usable with caveats)  
> **Target**: v1.0 (stable, documented, encrypted, cross-platform)  
> **Milestones**: v0.2.0 → v0.3.0 → v1.0

---

## 🔴 Blockers (Must Fix Before v0.2.0)

### 1. Semantic Tier Broken — numpy 2.x / scipy Incompatibility
**Impact**: `pip install sqac[all]` installs but semantic encoder crashes at runtime (`ModuleNotFoundError: scipy`).

**Root Cause**: `scipy` wheels compiled against numpy 1.x; current env has numpy 2.5.3.

**Fix Options**:
| Option | Effort | Trade-off |
|--------|--------|-----------|
| Pin `numpy<2` in `pyproject.toml` | 5 min | Works now; defers upgrade |
| Upgrade scipy/scikit-learn to numpy 2 compatible | 1-2 hrs | Future-proof; may break other deps |
| Vendor minimal scipy subset | 1 day | Zero deps; maintenance burden |

**Recommended**: Pin `numpy<2` for v0.2.0, schedule upgrade for v0.3.0.

---

### 2. No ARM64 SIMD Wheels
**Impact**: `pip install sqac[simd]` fails on Apple Silicon / ARM servers — falls back to NumPy (500ms vs 10ms at 10K).

**Fix**:
- Set up `cibuildwheel` in GitHub Actions (macOS + Linux ARM64)
- Cross-compile `sqac-simd` with `cargo build --target aarch64-unknown-linux-gnu`
- Test on macOS runner (`macos-14`)

**Files**: `.github/workflows/build.yml`, `sqac-simd/Cargo.toml`

---

### 3. No Cartridge Encryption
**Impact**: All cartridges (including session memory with secrets) stored as plaintext `.sqac` files.

**Threat Model**:
- Laptop theft → full conversation history exposed
- Shared CI runners → cross-project memory leak
- Backup leakage → PII in cartridges

**Fix Options**:
| Approach | Library | Overhead |
|----------|---------|----------|
| `age` (Rust) | `age` crate | ~5ms encrypt/decrypt |
| `cryptography` (Fernet) | Python | ~2ms |
| `libsodium` (secretbox) | `pynacl` | ~1ms |

**Recommended**: `age` — single binary, modern, streaming, integrates with Rust SIMD crate.

**Implementation**:
```python
# store.py: add encrypt/decrypt to save/load
def save(self, path, ..., passphrase=None):
    if passphrase:
        # age encrypt stream
    else:
        # current plaintext
```

---

## 🟡 High Priority (v0.3.0)

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

## 📋 Sprint Plan

### Sprint 1 (v0.2.0 — 1 week)
- [ ] Pin `numpy<2` in `pyproject.toml`
- [ ] Add ARM64 wheel build workflow
- [ ] Implement `age` encryption for `save/load`
- [ ] Fix MCP connection cleanup

### Sprint 2 (v0.3.0 — 2 weeks)
- [ ] `sqac migrate` command
- [ ] Windows CI + watchdog testing
- [ ] MCP auth + timeout handling
- [ ] Tiered storage prototype (background compaction)

### Sprint 3 (v1.0 — 3 weeks)
- [ ] Comprehensive docs (mkdocs + GitHub Pages)
- [ ] Scale benchmarks (100K, 1M entries)
- [ ] Cartridge signing (cosign)
- [ ] Cross-CLI integration tests (opencode ↔ claude-code ↔ codex)

---

## 🎯 Success Criteria for v1.0

| Metric | Target |
|--------|--------|
| `pip install sqac[all]` | Works on Linux/macOS/Windows, x86_64/ARM64 |
| Cold start latency | <2s (including model download) |
| Fuzzy search 100K entries | <50ms (SIMD) |
| Semantic search 25K entries | <27ms |
| Encryption overhead | <10ms per save/load |
| Migration success rate | 100% (automated tests) |
| Documentation coverage | 100% public API + CLI |
| Test matrix | 3 OS × 2 arch × 3 Python versions |

---

## 📝 Notes

- **Priority order** is flexible — adjust based on user feedback
- **Technical debt**: `sqac-simd` crate needs `pyo3` 0.22 upgrade (currently 0.21)
- **Dependencies**: Consider vendoring `tree-sitter-language-pack` to avoid system deps
- **Security**: Audit `credentials.py` / `injection.py` for bypasses

---

*Generated 2026-09-09 — update as work progresses*