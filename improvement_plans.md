# SQAC Roadmap — Current State

Aligned with `docs/superpowers/specs/2026-11-15-sqac-v1-design.md`.

## Released milestones

| Version | Scope | Status |
|---------|-------|--------|
| **0.1.x** | Core store, SIMD, HTTP server, MCP, graph | Released to PyPI |
| **0.2.0** | Bundled Rust SIMD (`sqac._simd`, abi3), CI/CD, NumPy 2 + Python 3.13, version single-source | Released |
| **0.3.0** | Vault cartridge (AES-256-GCM, keychain/Argon2id, `vault exec`, reference-only MCP tools, PII scan, hash-chained audit) | Released |
| **0.4.0** | Agent cartridge (persona + working state, bootstrap, type-based rack isolation) | Released |
| **0.5.0** | Cloud sync (provider-agnostic S3: R2/S3/Supabase/MinIO/B2/Spaces, push/pull/status, conflict handling) | Released |
| **0.5.1** | Hotfix: `cryptography>=42` as core dependency | Released |

## In flight — 1.0.0 hardening

Branch `feat/1.0.0-hardening` (PR #11). Scope:

- **Format hardening:** size caps in `read_cartridge` (dims ≤ 8192, vocab ≤ 1M,
  entries ≤ 10M, token ≤ 1024, payload/binvec/decompressed ≤ 64 MB) + fuzz tests.
- **Obsolescence cleanup:** removed `SqacStore.format_version`; always writes raw
  binvec (v2+). Deleted duplicated `sqac/graph/graph/` package.
- **Security review:** adversarial tests for credential detection (15) and prompt
  injection (23); Windows lock audit (sidecar lock, concurrent writers, compact
  re-entrancy); cross-CLI integration tests.
- **Docs:** README reframe ("SQAC replaces agent-facing `.md` files"),
  `CONTRIBUTING.md`, root `SECURITY.md`, dedicated docs (`VAULT/AGENT/CLOUD/
  FORMAT/MCP/RELEASE`), updated roadmap (this file).
- **Bug reporting:** `sqac report` CLI command (diagnostics + `gh issue create`).
- **Release gate:** full test suite green on the hardening branch; clean-venv
  install verification and `pip-audit` before tagging `v1.0.0`.

### Release-gate status (1.0.0)

| Gate | Status |
|------|--------|
| Full test suite | 400 passed, 21 skipped |
| PR #11 CI matrix (3 OS × py3.10–3.13 + numpy-fallback) | All checks SUCCESS |
| Clean-venv install (Windows, `pip install C:\path\to\sqac`) | Verified — `import ok 1.0.0` |
| `pip-audit` on runtime deps (fresh venv) | **No known vulnerabilities found** (sqac itself skipped: not yet on PyPI) |
| 100K-entry capacity benchmark | Measured — see `experiments/CAPACITY_BENCHMARK.md` |

Dev-environment note: the **global** dev Python environment audits dirty
(pip/setuptools/urllib3/filelock/pillow/transformers/accelerate/werkzeug carry
known CVEs from unrelated dev packages). Those are not SQAC runtime
dependencies; upgrade them at your convenience
(`py -3.12 -m pip install -U pip setuptools urllib3 filelock pillow`).

## Post-1.0 candidates (deferred, not scheduled)

- **aarch64 (Linux) + macOS wheels** — deferred from the v0.2.0 wheel matrix;
  re-enable when CI time allows (macos-14/macos-15-intel + QEMU aarch64).
- **Docs site** — optional MkDocs/GitHub Pages if README + `docs/` are no longer
  sufficient.
- **Agent-cartridge encryption** — reuse vault crypto for agent state if
  requested (v1 keeps agent state plaintext).
- **Multi-vault support** — `--vault` flag and `vault:VAULT/NAME` references;
  format already carries the vault name in AAD.
- **Performance gate** — benchmark thresholds become blocking (regression > 20%)
  once the 1.0 benchmark baseline is recorded.

## Known limitations (honest record)

- Memory/agent cartridges are plaintext by design; secrets belong in vaults.
- Fuzzy search is O(n) per scan; SIMD mitigates but does not eliminate scaling
  limits (see `experiments/CAPACITY_BENCHMARK.md` for measured ceilings).
- Cloud sync is explicit allowlist-based; continuity/session state is opt-in.
