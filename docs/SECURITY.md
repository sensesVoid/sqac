# SQAC Security Model

## Vault Cartridge (v0.3.0)

### Threat Model

**Protects against:**
- Disk theft / backup leakage: vault entries are AES-256-GCM ciphertext; plaintext values never touch disk.
- R2 object exposure: passphrase-mode vaults upload as ciphertext; keychain-mode vaults are local-only.
- Accidental context inclusion: no MCP tool returns plaintext; `vault exec` redacts resolved values from output.
- Model seeing a secret by default: reference-only (`vault:NAME`), resolved only in child processes or HTTP proxy.

**Does NOT protect against:**
- A compromised OS account (keychain unlocks automatically).
- A malicious command that prints an injected env var (redaction is best-effort, not a sandbox).
- An attacker who can call the MCP proxy with an allowed host.
- Physical access to an unlocked machine.

### Crypto

- **Cipher:** AES-256-GCM, random 96-bit nonce per entry.
- **DEK:** 32 bytes per vault, generated with `secrets.token_bytes`.
- **KEK:** OS keychain master key (default) or Argon2id (`time_cost=3`, `memory_cost=64 MiB`, `parallelism=4`, 16-byte salt).
- **AAD:** `sqac-vault-v1|<vault>|<name>|<entry_type>` — binds entry to vault/name/type; moving or renaming invalidates decryption.
- **DEK wrapping:** AES-256-GCM with `sqac-vault-dek-wrap` as AAD.
- **Tamper detection:** AES-GCM authentication tag; any modification to ciphertext/nonce/AAD causes decryption failure.

### Key Management

- **Keychain mode (default):** random 32-byte master key stored in OS keychain (service `sqac-vault`, username `default`). Wrapped DEK in header.
- **Portable mode (`--portable`):** Argon2id over a user-provided passphrase. Salt stored in header. Prompted with `getpass` (hidden on TTY) or `SQAC_VAULT_PASSPHRASE` env var.
- **Headless:** `SQAC_VAULT_PASSPHRASE` env var (documented; prefer keychain in production).
- **Rotation:** `sqac vault rotate` generates new DEK/KEK and re-encrypts all entries; old key material removed.
- **Keychain failures:** fall back to passphrase prompt; never silently store keys in files.

### Portability Contract

- Keychain-mode vaults are **local to the machine**. R2 sync warns and refuses to upload them unless `--allow-local-only-vault` is passed.
- Cross-machine use requires `vault init --portable` or `vault rotate --portable` before syncing.
- No key export endpoint exists in v1.

### MCP Boundary

- **Exposed tools:** `vault_list`, `vault_meta`, `vault_http`, `vault_audit`.
- **Deliberately absent:** `vault_get`, `vault_resolve`, `vault_put`, `vault_exec`. Writes happen through the CLI so the model never supplies or receives plaintext.
- **Locked behavior:** if no keychain key and no `SQAC_VAULT_PASSPHRASE`, metadata ops succeed; value ops return `{"ok": false, "error": "vault_locked"}` without prompting or hanging.

### HTTP Proxy

- HTTPS only; redirects followed only to allowlisted hosts (max 5).
- Exact host match required (no wildcards by default).
- Auth modes: `bearer` (default), `header:<name>`, `query:<param>`.
- Response body capped at 1 MB; timeout 30 s.
- Request headers never returned in response.

### Audit Log

- Location: `~/.sqacm/vault-audit.jsonl`.
- Fields: timestamp, vault, name, op, host, session, pid, success, bytes, error, prev (SHA-256 chain).
- **No values, ever.** Chain provides tamper evidence.

### PII Handling

- `sqac/pii.py` detects email, phone, SSN, credit card (Luhn-validated), IBAN, IP address.
- `mem_write` warns when content looks like PII but stores it (memory is plaintext by design).
- Vault `put --type pii` encrypts PII values.
- `vault scan` reports PII in plaintext cartridges and suggests migration. No automatic move.

### Rack Isolation

- Vault and agent cartridges are excluded from default search/rack by `cartridge_type` header (not by name).
- A normal memory cartridge named "vault" is unaffected.
- Vault entries never appear in `mem_search` results.

### Recommendations

1. Use keychain mode on personal machines; portable mode for CI/shared environments.
2. Set `SQAC_VAULT_PASSPHRASE` only in trusted CI environments; prefer keychain.
3. Review `vault audit` periodically for unexpected access.
4. Run `vault scan` before sharing cartridges.
5. Enable R2 object versioning for vault recovery.
