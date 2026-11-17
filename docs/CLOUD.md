# Cloud sync — S3-compatible storage

SQAC syncs cartridges to any S3-compatible object store. Cloudflare R2 is the
default preset, but the backend is provider-agnostic.

## Providers

| Provider | Endpoint | Region |
|----------|----------|--------|
| `r2` (default) | `https://<account_id>.r2.cloudflarestorage.com` | `auto` |
| `s3` | `https://s3.<region>.amazonaws.com` | from config |
| `supabase` | `https://<project_ref>.supabase.co/storage/v1/s3` | `auto` |
| `minio` | user-provided endpoint | `us-east-1` |
| `b2` | `https://s3.<region>.backblazeb2.com` | from config |
| `spaces` | `https://<region>.digitaloceanspaces.com` | from config |
| `custom` | user-provided endpoint | user-provided |

All providers use signature v4 with path-style addressing. Provider-specific
quirks are documented per provider in provider guides.

## Setup

```bash
sqac cloud init --provider r2 --bucket my-bucket --prefix v1
sqac cloud status
sqac cloud sync --dry-run
sqac cloud sync
```

- Config lives at `~/.sqacm/cloud.json` (provider, endpoint_url, bucket, prefix,
  region, sync_mode, exclude). **Never contains credentials.**
- Credentials come from the OS keychain (service `sqac-cloud`) or env vars:
  - Generic: `SQAC_CLOUD_ACCESS_KEY_ID`, `SQAC_CLOUD_SECRET_ACCESS_KEY`
  - Provider aliases also accepted: `SQAC_R2_*`, `SQAC_S3_*`, `SUPABASE_*`,
    `AWS_*` (first match wins)
  - `SQAC_CLOUD_ACCOUNT_ID` for R2 endpoint construction
- Sync mode: `SQAC_CLOUD_SYNC=off|pull|push|both` (default `off` — sync is
  explicit; auto pull/push is opt-in).

## What syncs (explicit allowlist)

- **Always:** `facts/`, `skills/`, `docs/` (memory), `agents/`, `vaults/`
  (including shard files)
- **Opt-in:** `continuity.json` (`sync_continuity`, default **false** — paths
  are machine-specific), `archives/`
- **Never:** `session*.sqac`, `checkpoint*.sqac`, `vault-audit.jsonl`,
  `cloud.json`, `.sync-state.json`, `cloud.log`, `agent_active`, `*.tmp`,
  `*.lock`, and anything under `secrets/`

## Per-cartridge sync policy (authoritative)

The path allowlist is only an *eligibility* filter. Every `.sqac` cartridge
carries its own `extra.sync_policy`:

| Value | Meaning |
|-------|---------|
| `write_through` | Eligible for sync (subject to the allowlist + vault/keychain rules) |
| `never` | Hard deny — never pushed/pulled/created from remote |
| `local_only` | Alias for `never` in v1.2 (reserved semantics) |

When the field is absent, defaults apply: **`agent`, `vault` and `secret` →
`never`**; memory/facts/skills/docs → `write_through`. Enforcement lives in
the sync engine and is fail-closed:

- Unreadable/corrupt/encrypted (SQAE/SQSE) headers resolve to `never`.
- Push skips policy-denied cartridges entirely (even under `--force-local`).
- Pull refuses to overwrite a policy-denied local cartridge, and refuses to
  write a policy-denied *remote* cartridge (verified from downloaded bytes) —
  `--force-remote` does not bypass either rule.
- `secrets/` is hard-denied for new remote material; policy-verified
  `write_through` `.sqac` cartridges are the only pullable exception, and
  non-`.sqac` material in sensitive directories (`agents/`, `vaults/`,
  `secrets/`) is refused by default.
- `sqac cloud status --show-denied` lists local cartridges excluded by
  policy (never synced).

Operators who explicitly want an agent cartridge synced must opt in per
cartridge: set `sync_policy: write_through` in the header (e.g. via
`sqac agent bind --sync-policy write_through`). Note that *encrypted* agent
cartridges can never have their policy verified remotely, so they are always
treated as `never` on pull.

Remote keys mirror the local relative path (`v1/facts/facts.sqac`,
`v1/vaults/default.sqac`, `v1/agents/atlas.sqac`). Names given to
`sqac cloud sync [NAME...]` are cartridge base names including their shards.

## Conflict policy

- Local state (`~/.sqacm/.sync-state.json`) tracks `{sha256, etag, synced_at}`
  per object.
- Status compares local sha, synced sha, and remote etag →
  `in_sync | local_ahead | remote_ahead | conflict`.
- Push uses conditional writes (`If-Match` / `If-None-Match: *`).
- **Conflicts abort that object with a clear message.** `--force-local` or
  `--force-remote` resolves deterministically. Nothing is silently overwritten.

```bash
sqac cloud sync --push           # push only
sqac cloud sync --pull           # pull only
sqac cloud sync --force-local    # resolve conflicts with local content
sqac cloud sync --force-remote   # resolve conflicts with remote content
```

## Vault behavior

- Sync treats vault files as opaque bytes — no decryption, no key upload.
- **Vault cartridges default to `sync_policy: never`** and never sync unless
  the operator opts in: `sqac vault set-policy write_through` (or sets the
  header manually). Keychain-mode vaults remain effectively local-only.
- Encrypted agent cartridges can never have their policy verified remotely;
  they are treated as `never` on pull regardless of header claims.
- There is no force flag that overrides an effective `never` policy.

## MCP tools

- `cloud_status()` — connectivity, provider/bucket, per-object sync status
- `cloud_sync(direction="both")` — trigger push/pull/both

Failures log to `~/.sqacm/cloud.log` and never block local writes. Offline mode
leaves local memory fully functional.

## Recommendations

1. Private bucket + scoped, least-privilege API token.
2. Enable object versioning for recovery.
3. Keep `sync_continuity` off unless you know the continuity paths are portable.
