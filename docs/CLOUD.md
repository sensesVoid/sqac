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
  `*.lock`

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
- **Keychain-mode vaults are local-only:** push warns and skips them unless
  `--allow-local-only-vault` is passed.
- Portable (passphrase) vaults upload as ciphertext.

## MCP tools

- `cloud_status()` — connectivity, provider/bucket, per-object sync status
- `cloud_sync(direction="both")` — trigger push/pull/both

Failures log to `~/.sqacm/cloud.log` and never block local writes. Offline mode
leaves local memory fully functional.

## Recommendations

1. Private bucket + scoped, least-privilege API token.
2. Enable object versioning for recovery.
3. Keep `sync_continuity` off unless you know the continuity paths are portable.
