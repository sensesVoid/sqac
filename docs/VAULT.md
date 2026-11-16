# Vault — encrypted secrets and PII

The vault is SQAC's encrypted storage layer. Secrets and PII must live in a
`vault` cartridge, never in a plaintext memory cartridge.

## Quick start

```bash
sqac vault init                     # keychain mode (default)
sqac vault put openai --type secret # value read from hidden prompt / stdin
sqac vault list                     # metadata only (no values)
sqac vault exec --env OPENAI_API_KEY=vault:openai -- python -c "import os; print(bool(os.getenv('OPENAI_API_KEY')))"
sqac vault audit --tail 20
sqac vault scan                     # find PII/secrets that leaked into plaintext cartridges
```

## Modes

| Mode | Keying | Portability |
|------|--------|-------------|
| `keychain` (default) | random 32-byte master key in the OS keychain (service `sqac-vault`) | **Local-only.** Not uploaded to cloud sync unless `--allow-local-only-vault` is passed |
| `portable` (`--portable`) | Argon2id over a passphrase (`time_cost=3`, `memory_cost=64 MiB`, `parallelism=4`, 16-byte salt) | Cross-machine; uploads as ciphertext |

- Passphrase input: hidden `getpass` prompt on a TTY, or `SQAC_VAULT_PASSPHRASE`
  env var for headless/CI use (prefer keychain on personal machines).
- `sqac vault rotate [--portable|--keychain]` re-encrypts all entries with a new
  DEK/KEK and removes the old key material.
- Keychain failures fall back to a passphrase prompt; keys are never silently
  written to files.

## What is protected

- Values are AES-256-GCM ciphertext at rest (random 96-bit nonce per entry,
  32-byte per-vault DEK wrapped by the keychain key or Argon2id KEK).
- AAD (`sqac-vault-v1|<vault>|<name>|<entry_type>`) binds entries to their
  vault/name/type; renaming or moving invalidates decryption.
- Tampered ciphertext/nonce/AAD fails closed (GCM authentication).
- Keychain-mode vault bytes stay local; portable-mode vaults sync as ciphertext.

## What is NOT protected

- A compromised OS account (keychain unlocks automatically).
- A child process that prints an injected env var — redaction in `vault exec`
  is best-effort, not a sandbox.
- Physical access to an unlocked machine.
- A caller that is allowed to use `vault_http` against an allowlisted host.

## Reference grammar and value access

- `vault:NAME` (canonical), `${vault:NAME}`, and `vault://NAME` are parsed as
  references. References never contain values.
- `sqac vault put` reads the value from a hidden prompt or stdin — never argv,
  so values stay out of shell history and model context.
- `vault exec --env VAR=vault:NAME -- CMD` injects values into the child process
  environment only, strips `SQAC_VAULT_*` variables from the child, and redacts
  resolved values from captured stdout/stderr by default (`--no-redact` opts out).

## MCP boundary (reference-only)

Exposed MCP tools: `vault_list`, `vault_meta`, `vault_http`, `vault_audit`.

Deliberately absent: `vault_get`, `vault_resolve`, `vault_put`, `vault_exec`.
Writes happen through the CLI; the model never supplies or receives plaintext
values. If the vault is locked (no keychain key, no `SQAC_VAULT_PASSPHRASE`),
metadata operations still work but value operations return a structured
`vault_locked` error without prompting.

## `vault_http` proxy

- HTTPS only; redirects followed only to allowlisted hosts (max 5).
- Exact host match required (no wildcards by default) — hosts must be configured
  on the entry with `--allowed-host`.
- Auth modes: `bearer` (default), `header:<name>`, `query:<param>`.
- Response body capped at 1 MB; 30 s timeout; request headers never returned.

## Audit log

- Location: `~/.sqacm/vault-audit.jsonl`.
- Fields: timestamp, vault, name, op (`put|meta|exec|http|rotate|remove`), host,
  session, pid, success, bytes, error, `prev` (SHA-256 chain for tamper evidence).
- **No values, ever.**

## PII handling

- `sqac/pii.py` detects email, phone, SSN, credit card (Luhn-validated), IBAN,
  and IP addresses.
- `mem_write` warns when content looks like PII but stores it (memory is
  plaintext by design).
- `vault put --type pii` encrypts the value.
- `sqac vault scan [CARTRIDGE...]` reports PII/secrets in plaintext cartridges
  and suggests migration; nothing is moved automatically.

## Recommendations

1. Keychain mode on personal machines; portable mode for CI/shared environments.
2. Set `SQAC_VAULT_PASSPHRASE` only in trusted CI environments.
3. Review `vault audit` periodically for unexpected access.
4. Run `vault scan` before sharing cartridges.
5. Enable object versioning on your cloud bucket for vault recovery.
