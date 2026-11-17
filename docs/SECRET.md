# Secret Domain — references and metadata, never values

A Secret Domain is a portable, encrypted-capable registry of secret
*references* an application or agent is authorized to use. It stores
**pointers and metadata only** — raw secret material lives in Vault
cartridges or external providers and is resolved exclusively inside the
trusted backend process.

## Quick start

```bash
sqac secret init                       # create the "default" domain (keychain-encrypted)
sqac secret init prod --portable       # passphrase-encrypted portable domain
sqac secret put-ref openrouter vault:prod/openrouter \
    --description "OpenRouter key for the inference service" --tag llm
sqac secret put-ref anthropic op://Private/Anthropic/api_key --tag llm
sqac secret list                       # metadata only — no values, ever
sqac secret get-meta openrouter
sqac secret rename openrouter openrouter_key
sqac secret rm openrouter_key
sqac secret list-domains
```

All commands print **metadata only**; no command can display a secret value.

## What a domain stores

| Field | Meaning |
|-------|---------|
| `name` | Stable key within the domain (`[A-Za-z0-9._-]{1,128}`) |
| `ref` | Resolution target — `vault:NAME`, `vault:VAULT/NAME`, `vault://...`, `${vault:...}`, `provider:1password:<op://...>`, `provider:bitwarden:<id>`, or aliases `op://...` / `bw://...` |
| `ref_kind` | Derived: `vault` or `provider` |
| `description` | Human label (non-secret) |
| `tags` | e.g. `["llm", "openrouter"]` |
| `scopes` | Optional allowed-use hints (hosts/tools) |
| `created` / `updated` | ISO-8601 UTC |

Invariants:

- `secret:DOMAIN/NAME` refs are **rejected** — no indirection chains.
- Credential-like material (pattern- and entropy-based) is rejected in every
  field, including descriptions and tags.
- `name` is unique within a domain; `put-ref` upserts atomically.
- The domain file is an SQSE envelope (AES-256-GCM; keychain-held key by
  default, or Argon2id + passphrase with `--portable` /
  `SQAC_SECRET_PASSPHRASE`). `--unencrypted` exists for development only.
- Header: `cartridge_type: secret`, `sync_policy: never` — the domain can
  never sync to the cloud (enforced by the sync engine).

## Storage

Domains live at `~/.sqacm/secrets/<name>.sqac`. The active/default domain is
`~/.sqacm/secrets/default.sqac`; `--domain NAME` selects another.
`SQAC_SECRET_DOMAIN` is honored by the MCP tools.

## SecretResolver (backend-only)

`sqac.secret_resolver.SecretResolver` resolves refs to plaintext **inside the
trusted backend process only**:

```python
from sqac.secret_resolver import SecretResolver, SecretResolutionError

r = SecretResolver()                      # sqac_dir=~/.sqacm by default
try:
    value = r.resolve("secret:default/openrouter",
                      agent_context={"secret_refs": ["secret:default/openrouter"]})
    provider_call(api_key=value)          # plaintext never leaves the process
except SecretResolutionError:
    ...                                    # fail closed
```

Rules the resolver enforces:

- **Agent allowlist (fail closed).** When `agent_context` carries
  `secret_refs`/`secret_ref`, refs outside the allowlist are denied and the
  denial is audited.
- **One level of indirection.** A `secret:` ref resolves to the target ref,
  which must not itself be a `secret:` ref (no chains, no cycles).
- **Backend-only.** Plaintext is never returned by MCP or HTTP surfaces.
- **Audit, never values.** Every resolution (ok / denied / error) appends a
  record to `~/.sqacm/vault-audit.jsonl` (hash-chained) with the ref, kind,
  outcome and latency — never the resolved value.
- **Best-effort zero retention.** Values are returned for immediate use; the
  resolver does not cache them.

Ref forms:

- `vault:NAME` → active vault entry `NAME`
- `vault:VAULT/NAME` → named vault (multi-vault) entry `NAME`
- `secret:DOMAIN/NAME` → domain metadata → target `ref` → resolve
- `provider:1password:op://...` → `op read` (1Password CLI)
- `provider:bitwarden:<id>` → `bw get password <id>` (Bitwarden CLI)
- `op://...`, `bw://...` → convenience aliases

## MCP tools (reference-only)

- `secret_list(domain=None)` — list refs (names, kinds, tags) — no values
- `secret_meta(name, domain=None)` — one ref's metadata — no values

There is deliberately **no** `secret_resolve`/`secret_get`/`secret_put` MCP
tool. Backend services (Genos or your own) import `SecretResolver` directly.

## Sync policy

`secret` cartridges default to `sync_policy: never` and the sync engine
enforces it: they are excluded from push collection and refused on pull
(even under force flags). See `docs/CLOUD.md`.
