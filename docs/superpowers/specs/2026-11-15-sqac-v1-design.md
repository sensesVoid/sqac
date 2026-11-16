# SQAC v1 — Production Design

**Date:** 2026-11-15
**Status:** Draft for maintainer approval
**Repo:** `github.com/sensesVoid/sqac` (cloned at `C:\Users\geral\sqac`, `main` @ `a92bcbc`)
**Current release:** 0.1.10 (PyPI)
**Target:** 1.0.0

---

## 1. Intent

SQAC is a research-grade memory cartridge with a working MCP server and a real user
(the maintainer). Four changes move it from "impressive prototype" to a product that
can be installed without caveats:

1. **SIMD ships in the main package.** `pip install sqac` must include the Rust
   acceleration on supported platforms. No `sqac[simd]` extra, no manual
   `maturin build`, no second distribution. A pure-NumPy fallback remains for
   platforms without a wheel.
2. **New cartridge types.** `agent` cartridges carry one agent's persona and working
   state. `vault` cartridges store API keys and PII encrypted, and expose only
   references — the plaintext never enters model context.
3. **Cloudflare R2 storage.** Cartridges can sync to R2 with a local cache, so memory
   survives machine changes and vaults can travel as ciphertext only.
4. **Production readiness + CI/CD.** Fix known breakage, enforce a supported
   Python/platform matrix, build wheels in CI, publish releases automatically, and
   close the correctness, security, and documentation gaps before 1.0.

**Why now:** the package is already on PyPI but `pip install sqac[simd]` fails
(`sqac-simd` is not published), Python 3.10/3.11 cannot even import the package
(syntax error), and vault-grade secrets currently have no safe home. Each of these
undermines the product story.

---

## 2. Goals

| # | Goal | Evidence of done |
|---|------|------------------|
| G1 | Bundled SIMD | Fresh `pip install sqac` on Windows/Linux/macOS x64 imports `sqac._simd`; `stats()["simd"]` is true; NumPy fallback still works |
| G2 | Vault type | Secrets/PII stored encrypted; no MCP tool returns plaintext; shell exec and HTTP proxy resolve references; audit log written |
| G3 | Agent type | Per-agent persona/working state; bootstrap packet auto-included; CLI + MCP tools; isolated from normal rack search |
| G4 | R2 sync | `sqac cloud sync` push/pull/status; offline cache; conflict detection; passphrase-mode vaults upload as ciphertext (keychain-mode vaults stay local-only) |
| G5 | CI/CD | CI matrix green (3 OS × 3.10–3.13); wheels + sdist published on tag via trusted publishing |
| G6 | v1 hardening | Known bugs fixed; version single-sourced; format reader hardened; docs cover every public API/CLI command |

## 3. Non-goals

- Multi-tenant server auth, hosted SaaS, distributed rack/Raft.
- Web UI for graph construction.
- Integrations with 1Password/Bitwarden/AWS Secrets Manager (v1 supports import from
  `.env`; external providers are post-1.0).
- Provider-specific SDKs. The vault HTTP proxy is generic.
- Encrypting ordinary memory by default. Only `vault` cartridges are encrypted in v1.
- Windows ARM64 and musllinux as hard release gates (best-effort jobs).
- Migrating the codebase off NumPy or rewriting `SqacStore` internals.

## 4. Anti-goals

- Any MCP tool that returns a decrypted secret value. There must be no
  `vault_get`/`vault_resolve` tool.
- Storing key material in a cartridge, repo, R2, or audit log.
- Silently indexing vault content in the semantic/fuzzy tiers or the shared rack.
- Breaking read compatibility with 0.1.x cartridges.
- Shipping a wheel that crashes on a CPU without AVX2 (`target-cpu=native` must go).

---

## 5. Constraints and discovered facts

- **Runtime:** Python 3.10+; `numpy<2` is pinned in `pyproject.toml`, which blocks
  Python 3.13/3.14 (no NumPy 1.x wheels). The code imports only `transformers` and
  `torch`, not `sentence-transformers` or `scipy`; NumPy 2.4.6 works on 3.12 today.
  The pin and the `sentence-transformers` extra are stale and must be removed.
- **Rust:** 1.97.1 and Cargo are installed on the maintainer machine. `pyo3` is pinned
  to 0.22; current stable is 0.30.1.
- **PyPI:** latest `sqac` is 0.1.10; `sqac-simd` does not exist (404). The `simd`
  extra is therefore broken.
- **Format:** `MAGIC=b"SQAC"`, `VERSION=4`, entry flags carry tombstone/binvec/
  compression/kind. Header has a forward-compatible `extra` JSON dict.
- **Storage:** every layer is path-based. `SqacStore.save/load`, `CartridgeRack`,
  `ContextOffloader`, and `CheckpointCartridge` all take filesystem paths.
- **MCP:** `sqac_mcp.py` holds a process-global `_MemoryState` with a rack over
  `~/.sqacm`, a session offloader, and a continuity store. Tool handlers are
  serialized by an RLock.
- **Version drift:** `sqac.__version__ = "0.1.6"`, `pyproject` = 0.1.10, MCP server
  declared 0.1.0, CHANGELOG ends at 0.1.5.
- **Known defects:** `sqac/autobuild.py:350` uses an f-string with a backslash
  (SyntaxError before 3.12); `sqac/graph/graph/` duplicates the graph package;
  `rack.create()` sets `store.format_version = 3` although the header version is
  always `VERSION`; no CI; duplicate headings in `improvement_plans.md`.

---

## 6. Approaches considered

| Decision | Options | Chosen | Why |
|----------|---------|--------|-----|
| SIMD packaging | setuptools-rust optional extension / maturin backend / keep separate | **setuptools-rust, `optional=True`** | Keeps the current build system and the pure-Python fallback; wheels built in CI; source installs without Rust still work; smallest change |
| Cartridge types | header-level type / entry-kind only / separate file extension | **Header-level `cartridge_type`** | Vault needs a cartridge-wide refusal gate and encryption policy; entry kinds alone would let a vault be indexed by normal code; a separate extension duplicates the format |
| Old-reader safety | bump all files to v5 / version-gate only vaults / no gate | **Vault files are header v5; memory/agent stay v4** | Old readers refuse v5 via their existing version check, but can still read normal cartridges. No silent vault reads |
| Secret usage | MCP get / shell env injection / server-side proxy | **Both: `vault exec` env injection first, then `vault_http` proxy** | Matches "called via var"; covers shell agents and MCP-only clients; plaintext stays inside a child process or the MCP server |
| Unlock | OS keychain / passphrase / key file | **Keychain default, Argon2id passphrase fallback** | Zero friction on the maintainer's Windows machine; headless fallback; key never syncs |
| Agent cartridge | registry / isolated namespace / persona+state | **Persona + working state** | Matches the product need; reuses `SqacStore`; avoids duplicating the memory model |
| Cloudflare | R2+cache / Worker API / KV/D1 | **R2 + local cache + explicit/auto sync** | No service to operate; offline-first; works with existing path-based code; ciphertext-only vaults |
| CI/CD | Actions + cibuildwheel + trusted publishing | **Chosen** | Standard for binary Python wheels; no long-lived PyPI token |

---

## 7. Architecture overview

```
                    ┌──────────────────────────────┐
                    │  Agent CLI / MCP client      │
                    └──────────────┬───────────────┘
                                   │ references only
             ┌─────────────────────┼──────────────────────┐
             │                     │                      │
      sqac_mcp.py            sqac CLI                 sqac-server
             │                     │                      │
      ┌──────┴─────────────────────┴──────────────────────┴─────┐
      │                     sqac package                        │
      │  rack.py ── facts/docs/skills (plaintext memory)        │
      │  agent.py ─ agents/<name>.sqac (persona + state)        │
      │  vault.py ─ vaults/<name>.sqac (AES-256-GCM, refs only) │
      │  cloud.py ─ sync engine ── storage.py ── R2            │
      │  store.py ─ SqacStore (exact/lexical/semantic tiers)    │
      │  _simd (Rust, bundled) ── NumPy fallback                │
      │  format.py ─ .sqac v4 (memory/agent), v5 (vault)        │
      └─────────────────────────────────────────────────────────┘
```

---

## 8. Workstream 1 — Bundled SIMD + CI/CD (v0.2.0)

### 8.1 Rust crate

- Rename the Python module to `sqac._simd`: Cargo `[lib] name = "_simd"`,
  `#[pymodule] fn _simd`.
- Upgrade `pyo3` to `0.30` with features `["extension-module", "abi3-py310"]`.
- Remove `target-cpu = "native"` from `[profile.release]`. Runtime dispatch already
  selects AVX2/POPCNT on x86_64 and NEON on aarch64.
- Release profile: `opt-level = 3`, `lto = "thin"`, `codegen-units = 1`,
  `strip = true`.
- Keep `bulk_similarity` (used by tests) and `pack_and_similarity` (store hot path).
- Commit `Cargo.lock`; add `cargo test` and `cargo clippy -- -D warnings` to CI.

### 8.2 Packaging

- Add `setup.py` with:

```python
from setuptools import setup
from setuptools_rust import RustExtension

setup(rust_extensions=[
    RustExtension(
        "sqac._simd",
        path="sqac-simd/Cargo.toml",
        binding="pyo3",
        optional=True,
        py_limited_api=True,
    )
])
```

- `build-system.requires`: `setuptools>=77`, `wheel`, `setuptools-rust>=1.13`.
- `SQAC_SKIP_RUST=1` makes `setup.py` omit the extension (Rust-less sdist installs
  and the fallback CI job).
- Add `MANIFEST.in` to include `sqac-simd/Cargo.toml`, `sqac-simd/Cargo.lock`,
  `sqac-simd/src/*.rs`.
- Remove the `simd` extra and the `sqac-simd` dependency. `all` no longer references
  it. Update README install instructions.
- Add `sqac/simd.py`:

```python
try:
    from . import _simd as backend
except ImportError:
    try:
        import sqac_simd as backend   # legacy installs
    except ImportError:
        backend = None
HAS_SIMD = backend is not None
```

  `store.py` imports `from .simd import backend as _simd, HAS_SIMD`.

### 8.3 CI — `.github/workflows/ci.yml`

- Matrix: `ubuntu-latest`, `windows-latest`, `macos-14` × Python 3.10, 3.11, 3.12,
  3.13. (3.14 is best-effort until NumPy/Transformers wheels settle.)
- Steps: checkout, setup Python, `dtolnay/rust-toolchain@stable`,
  `Swatinem/rust-cache@v2`, `pip install -e ".[mcp,server,compress]"`,
  `pytest -q`, `python -c "import sqac.simd as s; assert s.HAS_SIMD"` (except the
  fallback job), `cargo test`, `ruff check`.
- Separate fallback job: `SQAC_SKIP_RUST=1 pip install -e .` then run tests with
  `stats()["simd"] is False`.
- Add `ruff` config to `pyproject.toml`; fix violations once.
- `pip-audit` runs non-blocking initially; becomes blocking at 1.0.

### 8.4 Wheels — `.github/workflows/wheels.yml`

- `python -m build --sdist` plus `cibuildwheel`:
  - Linux x86_64 + aarch64 (QEMU), macOS x86_64 + arm64, Windows AMD64.
  - `CIBW_BUILD=cp310-*` with abi3 (`py_limited_api=True`), producing one wheel per
    platform. Fall back to `cp310 cp311 cp312 cp313` if abi3 causes problems.
  - `CIBW_TEST_COMMAND` imports `sqac`, asserts `HAS_SIMD`, and checks
    `pack_and_similarity` parity against NumPy on random vectors.
- Artifacts uploaded and retained 14 days.

### 8.5 Release — `.github/workflows/release.yml`

- Trigger: tag `v*` (e.g. `v0.2.0`) or manual dispatch for TestPyPI. The repo has no
  tags or GitHub Releases today; `v0.2.0` establishes the convention.
- Publishing uses **PyPI Trusted Publishing** (OIDC), not an API token:
  `permissions: id-token: write`, `environment: pypi`, and
  `pypa/gh-action-pypi-publish`. No secret is stored in GitHub.
- One-time setup by the maintainer on PyPI: project `sqac` → Manage → Publishing →
  add a GitHub publisher with owner `sensesVoid`, repository `sqac`, workflow
  `release.yml`, environment `pypi`. Create the `pypi` environment in GitHub
  settings (optionally with a required reviewer). Until that exists, the publish job
  fails authentication, which is the expected signal.
- Jobs: sdist → wheels → `twine check` → publish → GitHub Release with artifacts and
  the CHANGELOG section. A manual `workflow_dispatch` input can target TestPyPI with
  a separate publisher.
- `dependabot.yml` for `pip`, `github-actions`, and `cargo`.
- Repository settings: require CI on `main`; protect release tags.

### 8.6 Acceptance

- Fresh install on clean Windows 11 x64, Ubuntu 24.04 x64:
  `sqac stats` reports `"simd": true`.
- macOS and aarch64 (Linux) wheels are deferred to v1.0; source builds with
  Rust installed still produce native wheels for those platforms.
- `pip install --no-binary sqac sqac` on a machine without Rust still imports and
  falls back to NumPy.
- Benchmark does not regress: 25K fuzzy scan ≤ 20 ms with SIMD.

---

## 9. Workstream 2 — Vault cartridge (v0.3.0)

### 9.1 Threat model

**Protects against:** disk theft, backup leakage, R2 object exposure, accidental
context inclusion, model seeing a secret by default.

**Does not protect against:** a compromised OS account (keychain unlocks), a malicious
command that prints an injected env var (redaction is best-effort, not a sandbox), or
an attacker who can call the MCP proxy with an allowed host. Document these limits in
`docs/SECURITY.md`.

### 9.2 Format

- Header `extra["cartridge_type"] = "vault"`; header version **5**. `format.VERSION`
  becomes 5 (maximum supported), while `CartridgeHeader.version` defaults to
  `WRITE_VERSION = 4`; only vault creation writes 5. Readers accept ≤5; 0.1.x
  readers refuse v5 before parsing entries, so a vault can never be read as plain
  memory by an old client.
- Header `extra["vault"]`:

```json
{
  "schema": 1,
  "kdf": "keychain" | "argon2id",
  "salt": "b64",              // argon2id only
  "time_cost": 3,
  "memory_cost": 65536,
  "parallelism": 4,
  "wrap_nonce": "b64",
  "wrapped_dek": "b64",
  "keychain_service": "sqac-vault",
  "created": "ISO-8601"
}
```

- Entry payload (metadata is plaintext; the value is not):

```json
{
  "name": "openai",
  "entry_type": "secret" | "pii",
  "label": "OpenAI API key",
  "tags": ["llm"],
  "policy": {"allowed_hosts": ["api.openai.com"], "auth": "bearer", "header_name": null},
  "created": "...", "updated": "...",
  "nonce": "b64",
  "ct": "b64"
}
```

- `key_bits` encodes the name for exact lookup; no `binvec`, no semantic vectors.
- `FLAG_ENCRYPTED = 0x0008` (new flag bit) is set on every vault entry.
- AAD for each entry: `sqac-vault-v1|<vault>|<name>|<entry_type>`. v1 always uses
  the vault name `default`; the field is included now so a future multi-vault
  release does not need a format change. Moving an entry between vaults or renaming
  it invalidates decryption.
- Crypto: AES-256-GCM with a random 96-bit nonce per entry; 32-byte DEK per vault;
  DEK wrapped by the keychain master key or an Argon2id-derived KEK.
- Value length cap: 64 KB per entry; names are opaque identifiers (warn if a name
  looks like an email/phone; PII belongs in the value).

### 9.3 Key management

- v1 has exactly **one vault**, stored at `~/.sqacm/vaults/default.sqac` and named
  `default` internally. `vault put/rm/list` address entries, not vaults. A `--vault`
  flag and `vault:VAULT/NAME` references are reserved for a later release; the
  header/AAD/cloud layout already carry the vault name so that change is additive.
- `sqac vault init [--portable]` creates the cartridge and generates a DEK.
- Default: random 32-byte master key in the OS keychain (`keyring`, service
  `sqac-vault`, username `default`). The wrapped DEK lives in the header.
- `--portable` (alias `--passphrase`): Argon2id (`time_cost=3`, `memory_cost=64 MiB`,
  `parallelism=4`) over a 16-byte salt stored in the header; prompted with `getpass`.
- Headless: `SQAC_VAULT_PASSPHRASE` env var, with a warning in docs.
- `sqac vault rotate [--portable|--keychain]` generates a new DEK/KEK and
  re-encrypts; the old key material is removed from the keychain.
- Keychain failures fall back to a passphrase prompt; never silently store the key
  in a file.
- **Portability contract:** keychain-mode vaults are local to the machine. R2 sync
  warns and refuses to upload them unless `--allow-local-only-vault` is passed.
  Cross-machine use requires `vault init --portable` or `vault rotate --portable`
  before syncing. No key export endpoint exists in v1.

### 9.4 API and reference resolution

```python
class Vault:
    @classmethod
    def create(cls, path, mode="keychain") -> "Vault": ...
    def put(self, name, value, entry_type="secret", label="", tags=None,
            policy=None) -> dict: ...
    def get_meta(self, name) -> dict: ...
    def list(self) -> list[dict]: ...
    def remove(self, name) -> bool: ...
    def rotate(self, mode=None) -> None: ...
    # internal, never exposed through MCP:
    def resolve(self, name) -> str: ...
    def exec_env(self, mapping: dict[str, str], cmd: list[str],
                 redact=True) -> ExecResult: ...
    def http(self, name, method, url, headers=None, body=None,
             timeout=30) -> HttpResponse: ...
```

- Reference grammar: `vault:NAME`, `${vault:NAME}`, `vault://NAME`; canonical form is
  `vault:NAME`. `vault exec --env VAR=vault:NAME` resolves only the listed variables.
- Unlock lifecycle: the keychain fetch or Argon2id derivation happens once per
  process; the unwrapped DEK is cached in memory keyed by vault path + header mtime
  and dropped on rotation or process exit. The stdio MCP server never prompts; if
  the keychain has no key and `SQAC_VAULT_PASSPHRASE` is unset, value operations
  fail with a structured "vault locked" error while metadata operations still work.
- `exec_env` copies the current environment, removes `SQAC_VAULT_*` variables from
  the child, runs with `subprocess.run(capture_output=True)`, and by default replaces
  any occurrence of a resolved value in stdout/stderr with
  `[REDACTED:vault:NAME]`. `--no-redact` is opt-in.
- `http` is stdlib `urllib.request`:
  - requires an `allowed_hosts` policy on the entry (exact host match; no wildcards
    by default);
  - HTTPS only; redirects are followed only to allowlisted hosts, max 5;
  - auth modes: `bearer` (default), `header:<name>`, `query:<param>`;
  - response body capped at 1 MB; timeout 30 s; request headers never returned.
- Audit log `~/.sqacm/vault-audit.jsonl`: timestamp, vault, name, op
  (`put|meta|exec|http|rotate|remove`), host, host CLI/session, pid, success, bytes.
  Each record carries `prev` (SHA-256 of the previous record) for tamper evidence.
  No values, ever.

### 9.5 CLI

```
sqac vault init [--portable]
sqac vault put NAME --type secret|pii --label "..." [--allowed-host H]... [--from-env VAR | --from-file PATH]
sqac vault list [--json]
sqac vault rm NAME
sqac vault rotate [--portable|--keychain]
sqac vault exec --env VAR=vault:NAME [--env ...] [--no-redact] -- CMD [ARGS...]
sqac vault audit [--tail N]
sqac vault scan [CARTRIDGE...]      # report secrets/PII in plaintext cartridges
sqac vault import-env .env [--prefix]
```

`put` reads the value from stdin (hidden prompt on a TTY), never from argv, so it
does not enter shell history or model context. `import-env` parses `KEY=VALUE` lines
and stores them as secret entries. `--vault` is not part of the v1 CLI; adding it
later selects among named vaults without changing any reference syntax.

### 9.6 MCP surface

- `vault_list()` → metadata only.
- `vault_meta(name)` → label/type/tags/policy/updated; no value.
- `vault_http(name, method, url, body=None, headers=None)` → status + response body
  (redacted); rejects non-allowlisted hosts. If the vault is locked (no keychain key
  and no `SQAC_VAULT_PASSPHRASE` in the server environment), returns
  `{"ok": false, "error": "vault_locked", "hint": "..."}` without attempting a prompt.
- `vault_audit(limit=20)`.
- **Deliberately absent:** `vault_get`, `vault_resolve`, `vault_put`, `vault_exec`.
  Writes happen through the CLI so the model never supplies or receives plaintext.
- Server instructions mention this boundary explicitly.

### 9.7 PII handling

- New `sqac/pii.py`: email, phone (E.164-ish), US SSN, credit card (Luhn), IBAN,
  IP address, passport-like patterns. Returns spans + types; no blocking.
- `mem_write` warns when content looks like PII but stores it (memory is plaintext by
  design). Vault `put --type pii` encrypts it.
- `vault scan` walks normal cartridges, reports matches, and suggests
  `sqac vault put`. No automatic move in v1.

### 9.8 Acceptance tests

- Cartridge bytes contain no plaintext value (grep/entropy test).
- Wrong passphrase and tampered ciphertext/AAD fail closed.
- `keyring` mocked; keychain-missing path falls back to prompt.
- Search/rack never returns vault entries.
- `exec` injects env, strips `SQAC_VAULT_*`, redacts output.
- `http` rejects disallowed hosts and redirects; succeeds against a local HTTPS test
  server on an allowlisted host.
- Audit chain verifies; rotation invalidates old ciphertext.

---

## 10. Workstream 3 — Agent cartridge (v0.4.0)

- Header `extra["cartridge_type"] = "agent"`; header version 4 (old readers see a
  normal memory cartridge). `extra["agent"] = {schema: 1, name, role, created,
  updated}`.
- Entries use `meta.section` ∈ `identity`, `instruction`, `preference`, `goal`,
  `decision`, `note`; `meta.priority` 0–3; `meta.status` `open|done`; optional tags.
  Entry `kind` stays `fact`/`generic` so existing search works.
- `sqac/agent.py`:

```python
class AgentCartridge:
    def __init__(self, path, name, role="", semantic=False): ...
    @classmethod
    def create(cls, path, name, role="") -> "AgentCartridge": ...
    def set(self, section, content, key=None, priority=1, tags=None) -> int: ...
    def note(self, content, tags=None) -> int: ...
    def complete(self, key_or_query) -> bool: ...
    def search(self, query, top_k=3) -> list[Hit]: ...
    def bootstrap(self, max_chars=4000) -> str: ...
    def stats(self) -> dict: ...
```

- `set(section, content, key=None)` semantics:
  - With `key`: upsert by normalized key, replacing that entry's content/metadata.
  - Without `key` on a singleton section (`identity`, `instruction`, `preference`):
    replace the single existing entry for that section.
  - Without `key` on a list section (`goal`, `decision`, `note`): append and return a
    generated stable key (`goal-<8 hex>`, `decision-<8 hex>`, `note-<8 hex>`), stored
    in `meta.key`.
- `complete(key_or_query)` resolves an exact `meta.key` first, then a unique fuzzy
  search hit; sets `meta.status="done"` and returns false if ambiguous. `agent set`
  and `agent_update` accept an optional key for the same upsert behavior.
- `bootstrap()` order: identity + instructions + preferences (all), open goals by
  priority then recency, last 5 decisions, then notes only if budget remains. Closed
  goals and superseded singletons never duplicate: upsert replaces them.
- Storage: `~/.sqacm/agents/<name>.sqac`; active agent in `~/.sqacm/agent_active`.
- Rack: auto-load skips `agent` and `vault` types. `register()` may mount an agent
  cartridge for type-aware APIs, but the default search-target collector excludes
  `agent`/`vault` types by header type (not by name), so `mem_search` never returns
  them. `agent_search` is the explicit path. A normal cartridge named `agent` is
  unaffected because isolation is type-based.
- CLI: `sqac agent init|list|use|show|set|note|done|bootstrap|search`.
- MCP: `agent_bootstrap(name=None)`, `agent_update(section, content, key=None,
  priority=1)`, `agent_done(key_or_query)`, `agent_search(query, name=None)`,
  `agent_list()`, `agent_use(name)`, `agent_show(name=None)`.
- `mem_bootstrap` prepends the active agent packet when present; `SQAC_AGENT=0`
  disables. Token budget defaults to 4000 characters.
- Privacy: agent state is plaintext in v1; docs flag it. Encryption can reuse the
  vault crypto in a later release if requested.

### Acceptance

- Creating two agents and switching restores the right persona in a new MCP session.
- Agent entries do not appear in `mem_search` results unless explicitly requested.
- Bootstrap stays under the character budget; goals/decisions sort correctly.

---

## 11. Workstream 4 — Cloudflare R2 sync (v0.5.0)

### 11.1 Storage seam

`sqac/storage.py`:

```python
class StorageBackend(Protocol):
    def get(self, key: str) -> bytes: ...
    def put(self, key: str, data: bytes, *, if_match: str | None = None,
            if_none_match: bool = False) -> str: ...   # returns etag
    def head(self, key: str) -> ObjectInfo | None: ...
    def delete(self, key: str) -> None: ...
    def list(self, prefix: str = "") -> list[ObjectInfo]: ...
```

- `LocalBackend` (tests + future), `S3Backend` (boto3 S3 API, optional extra
  `cloud = ["boto3>=1.34"]`), `FakeBackend` (tests, supports conditional put).
- S3Backend is provider-agnostic: accepts `endpoint_url`, `region`, `access_key_id`,
  `secret_access_key`, and `bucket`. Works with any S3-compatible provider.
- Built-in provider presets (auto-fill endpoint/region):
  - `r2`: Cloudflare R2 — `https://<account_id>.r2.cloudflarestorage.com`, region `auto`
  - `s3`: AWS S3 — `https://s3.<region>.amazonaws.com`, region from config
  - `supabase`: Supabase Storage — `https://<project_ref>.supabase.co/storage/v1/s3`, region `auto`
  - `minio`: self-hosted MinIO — user-provided endpoint, region `us-east-1`
  - `b2`: Backblaze B2 — `https://s3.<region>.backblazeb2.com`, region from config
  - `spaces`: DigitalOcean Spaces — `https://<region>.digitaloceanspaces.com`, region from config
- All providers use signature v4 and path-style addressing. Provider-specific
  quirks (e.g., R2 has no egress fees, Supabase S3 gateway rate limits) are
  documented per-provider in docs/CLOUD.md.
- `sqac cloud init --provider <name>` walks through provider-specific setup and
  verifies a `head_bucket`. Auto-detects provider from env vars when possible
  (e.g., `SQAC_R2_ACCOUNT_ID` → r2, `SUPABASE_URL` → supabase).

### 11.2 Config and credentials

- `~/.sqacm/cloud.json`: `{provider, endpoint_url, bucket, prefix, region, sync_mode, exclude}`.
  The `provider` field selects the preset; `endpoint_url` and `region` are auto-filled
  from the preset but can be overridden (e.g., for MinIO or custom S3-compatible stores).
- Credentials in the OS keychain (service `sqac-cloud`) or env vars:
  - Generic: `SQAC_CLOUD_ACCESS_KEY_ID`, `SQAC_CLOUD_SECRET_ACCESS_KEY`
  - Provider-specific aliases also accepted: `SQAC_R2_*`, `SQAC_S3_*`, `SUPABASE_*`,
    `AWS_*` (checked in priority order; first match wins).
  - `SQAC_CLOUD_ACCOUNT_ID` for R2 (used to build endpoint).
- `sqac cloud init [--provider r2|s3|supabase|minio|b2|spaces]` walks through
  provider-specific setup and verifies a `head_bucket`; `sqac cloud status`
  shows connectivity, object count, and last sync.
- Never write credentials into `cloud.json`. Only non-secret config (provider, bucket,
  prefix, region, sync_mode, exclude) is persisted.

### 11.3 Sync engine

- Sync set is an explicit allowlist, not a directory walk:
  - always: `facts/`, `skills/`, `docs/` (memory), `agents/`, `vaults/`, including
    shard files;
  - opt-in: `continuity.json` (`sync_continuity`, default false; paths are
    machine-specific) and `archives/` (default false);
  - never: `session*.sqac`, `checkpoint*.sqac`, `vault-audit.jsonl`, `cloud.json`,
    `.sync-state.json`, `cloud.log`, `agent_active`, `*.tmp`, `*.lock`.
- Remote keys mirror the local relative path: `v1/facts/facts.sqac`,
  `v1/vaults/default.sqac`, `v1/agents/atlas.sqac`.
- In `sqac cloud sync [NAME...]`, `NAME` is a cartridge base name and includes all
  of its shards (same convention as the rack), not a filesystem path.
- A keychain-mode vault is local-only: push warns and skips it unless
  `--allow-local-only-vault` is given. Passphrase-mode vaults upload as ciphertext.
- Local state `~/.sqacm/.sync-state.json`: per object
  `{sha256, etag, synced_at}`.
- Push: upload changed objects with `If-Match` (update) or `If-None-Match: *`
  (create). Pull: download when the remote etag differs from local state. Status:
  compare local sha, synced sha, and remote etag → `in_sync|local_ahead|remote_ahead|
  conflict`.
- Conflict policy: abort that object with a clear message; `--force-local` or
  `--force-remote` resolves. Never silently overwrite.
- Modes: `SQAC_CLOUD_SYNC=off|pull|push|both` (default `off`). In MCP, `pull` runs at
  state init; `push` runs after `mem_save`/`mem_checkpoint` with a 3-second debounce
  on a daemon thread. Failures log to `~/.sqacm/cloud.log` and never block local
  writes.
- Vaults: sync treats files as opaque bytes; no decryption, no key upload.
- CLI: `sqac cloud sync [--push|--pull|--both] [--dry-run] [--force-local|--force-remote] [NAME...]`.
- MCP: `cloud_status()`, `cloud_sync(direction="both")`.
- R2 settings: private bucket, scoped API token, object versioning recommended.
  Document in `docs/CLOUD.md`.

### 11.4 Acceptance

- Round-trip push/pull against a fake backend and against R2 (gated by env vars).
- Conflict detection when both sides change; force flags resolve deterministically.
- Offline mode leaves local memory fully functional.
- Vault object bytes on R2 are ciphertext (same no-plaintext test).

---

## 12. Workstream 5 — v1 hardening (v1.0.0)

### 12.1 Defects to fix

1. `sqac/autobuild.py:350` f-string backslash → SyntaxError on 3.10/3.11. Precompute
   the stripped value; add 3.10/3.11 to CI.
2. Delete the duplicated `sqac/graph/graph/` package after confirming no imports
   reference it.
3. Single-source the version: `sqac.__version__` from `importlib.metadata`, MCP
   server version from the package, CHANGELOG catches up 0.1.6–0.1.10 and each new
   release.
4. Remove `numpy<2`; allow `numpy>=1.24`; drop `sentence-transformers` from extras.
   Test on NumPy 1.26 and 2.x.
5. Remove `target-cpu=native`; upgrade `pyo3` to 0.30.
6. Clarify/remove `SqacStore.format_version` and the `rack.create()` assignment of 3;
   the header version is the format version.
7. Harden `read_cartridge`: cap `pb_len` and decompressed payload size (64 MB), reject
   negative/huge counts, and fuzz truncation/corruption.
8. Audit Windows file locking: `msvcrt.locking` on a zero-byte file, concurrent
   writers, and `compact_cartridge` re-entrancy. Add a sidecar lock fallback if needed.
9. `pyproject` metadata: PEP 639 license (`setuptools>=77`), `license-files`,
   `project.urls`, classifiers, `[tool.pytest.ini_options]`, `[tool.ruff]`.
10. Correct the MCP dependency pin to `mcp>=2.0,<3`: `sqac_mcp.py` imports
    `mcp.server.mcpserver.MCPServer`, which does not exist in the 1.x SDK. Align
    `requirements-mcp.txt` or delete it in favor of extras.
11. Update README: bundled SIMD, removed `[simd]`, vault/agent/cloud sections,
    architecture, threat model. Rewrite `improvement_plans.md` (duplicate headings,
    stale status) or replace it with a roadmap that points at this design.
12. Add `SECURITY.md`, `CONTRIBUTING.md`, and release notes process.

### 12.2 Security review

- `credentials.py` and `injection.py` bypass audit; add adversarial tests (base64,
  split tokens, homoglyphs, zero-width characters).
- Vault boundary tests: no MCP tool returns plaintext; rack/auto-load skip vaults;
  audit chain; SSRF allowlist; redirect handling.
- Constant-time API key comparison in `server.py`; auth required for non-loopback.
- Path traversal in cartridge names (`rack.create("../x")`) and cloud keys.
- Decompression bomb and oversized-payload limits.

### 12.3 Documentation

- `docs/VAULT.md`, `docs/AGENT.md`, `docs/CLOUD.md`, `docs/SECURITY.md`,
  `docs/FORMAT.md` (v4/v5), `docs/MCP.md` (tool list per client).
- Every CLI command in `--help` has a README example; every MCP tool has a one-line
  purpose and an example.
- MkDocs site is optional for 1.0; README + `docs/` must be complete.

### 12.4 Release gate

- CI matrix green: 3 OS × 3.10–3.13; wheels import and pass parity tests.
- `pip install sqac`, `pip install sqac[mcp]`, `pip install sqac[all]` verified in a
  clean venv on Windows and Linux.
- No open P0/P1 issues; all known defects above closed.
- Benchmarks within 10% of the 0.1.10 baseline (SIMD path).
- `pip-audit` clean for runtime dependencies.

---

## 13. Version and release plan

| Version | Scope |
|---------|-------|
| **0.2.0** | Bundled SIMD, abi3 wheels, CI/CD, 3.10/3.11 syntax fix, version single-source, NumPy 2 / Python 3.13 support |
| **0.3.0** | Vault type: encryption, keychain/passphrase, `vault exec`, MCP `vault_http`, audit, PII detection |
| **0.4.0** | Agent type: persona/working state, bootstrap, CLI + MCP tools |
| **0.5.0** | Cloudflare R2 sync: storage seam, push/pull/status, conflict handling, MCP tools |
| **1.0.0** | Hardening, security audit, docs, compatibility guarantees, release gate |

Each release: CHANGELOG entry, version bump, tag `vX.Y.Z`, CI publishes wheels/sdist,
GitHub Release notes.

---

## 14. Test strategy

- Keep `pytest` (it already runs the `unittest` suites).
- New suites: `test_simd.py` (parity + availability), `test_vault.py` (crypto,
  redaction, exec, HTTP allowlist, audit), `test_agent.py`, `test_cloud.py`
  (FakeBackend + optional moto), `test_format_fuzz.py`.
- Crypto tests use fixed keys/test vectors; keyring is mocked. No real secrets in the
  repo (the repo already has a "no test secrets" convention).
- CI: unit + integration on the full matrix; wheel smoke tests; a Rust job for
  `cargo test` + `clippy`.
- Performance: a lightweight benchmark job records SIMD throughput; thresholds are
  advisory until 1.0, then blocking at 20% regression.

---

## 15. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| abi3/pyo3/setuptools-rust wheel failures | Keep per-version wheel fallback behind a matrix switch; fallback CI job proves NumPy path |
| Rust build breaks sdist installs | `optional=True` + `SQAC_SKIP_RUST=1`; document that prebuilt wheels are the supported path |
| Vault crypto misuse | AEAD with AAD, random nonces, authenticated wraps, tamper tests, external review before 1.0 |
| Secret leakage through command output | Redact by default; document limits; never expose `exec` over MCP |
| R2 API differences (conditional writes) | FakeBackend logic tests + conditional-write probe; fall back to read-before-write with conflict warning |
| MCP SDK drift | Pin `<2`; CI covers the installed version; bump deliberately |
| Windows file locking edge cases | Dedicated Windows CI job; sidecar lock fallback |
| Scope creep across five workstreams | Ship each version independently; no workstream blocks another's release |

---

## 16. Decisions made by the builder (check these)

- Cartridge type lives in `header.extra["cartridge_type"]`; only vaults bump the
  file version to 5.
- v1 has a single vault named `default` at `~/.sqacm/vaults/default.sqac`;
  multi-vault support is additive later (`--vault`, `vault:VAULT/NAME`), and the
  header/AAD/cloud paths already carry the name.
- Vault writes are CLI-only; MCP exposes list/meta/http/audit and no getter.
- Vault HTTP requires an explicit host allowlist; wildcards are refused.
- `vault exec` redacts captured output by default.
- Keychain-mode vaults are local-only; R2 push warns/skips them. Portable
  (passphrase) vaults are the cross-machine path.
- The stdio MCP server reads `SQAC_VAULT_PASSPHRASE`, caches the unwrapped DEK in
  process memory, never prompts, and returns `vault_locked` for value operations.
- Agent cartridges stay plaintext in v1; vault encryption is not reused there yet.
- Agent `set` upserts when given a key, replaces singleton sections, and appends with
  a generated key for list sections. `agent_done` accepts a key or unique query.
- Rack isolation is type-based: `agent`/`vault` cartridges are excluded from default
  search even if explicitly registered; `agent_search` is the explicit path.
- Cloud sync is an explicit allowlist (memory, agents, vaults; opt-in continuity),
  not a directory walk; audit/cloud/config/session/checkpoint files never sync.
- Wheel strategy starts with abi3; per-version wheels are the fallback.
- R2 sync is explicit by default (`off`), with opt-in auto pull/push.
- PyPI publishing uses Trusted Publishing (OIDC); no API token is stored.
- CI is hard-gated on Python 3.10–3.13 × Linux/macOS/Windows; 3.14 is best-effort.

## 17. Left to the builder

- Exact wording of CLI help, MCP tool descriptions, and error messages.
- PII regex set and confidence thresholds.
- R2 bucket/account configuration and token scoping (needs maintainer credentials).
- MkDocs vs README-only docs site.
- Whether `continuity.json` sync should default on after dogfooding (v1 default is
  off).
