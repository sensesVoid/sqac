# `.sqac` format reference

A `.sqac` file is a self-contained, portable, hot-swappable memory cartridge.

```text
+-----------------------------+
| header (fixed, JSON)        |  magic, version, dims, vocab fingerprint
| vocab    (string -> index)  |  the encoder's atom table
| entries  (n records)        |  packed key bits + payload
| ext      (optional block)   |  forward-compatible extensions
+-----------------------------+
```

Design rules:

- Keys are packed BSC bit-vectors (D bits → D/8 bytes) so lookups are
  XOR + popcount on raw bytes — no float math in the hot path.
- Payloads are plaintext for memory/agent cartridges: the consumer sees text +
  confidence, never a vector. The vector is an address, not a message.
- The vocabulary is the ABI: the header carries a fingerprint so a runtime can
  refuse cartridges encoded with an incompatible vocab.
- Unknown extension blocks MUST be skipped by readers (forward compatibility).

## Versions

| Version | Change |
|---------|--------|
| v1 | auxiliary vectors (content/semantic keys) hex-encoded in payload JSON |
| v2 | auxiliary vectors moved into a per-entry raw binary block (`FLAG_BINVEC`) |
| v3 | knowledge kind added to entry flags (bits 4–11, uint8: generic/fact/skill/doc/turn) |
| v4 | lz4 compression on binvec blocks (`FLAG_COMPRESSED`) |
| v5 | vault cartridges (`FLAG_ENCRYPTED` entries, header type `vault`, encrypted payloads) |

Constants (from `sqac/format.py`):

- `MAGIC = b"SQAC"`
- `VERSION = 5` — maximum supported reader version
- `WRITE_VERSION = 4` — default for memory/agent cartridges
- `VAULT_WRITE_VERSION = 5` — vault cartridges

Old readers refuse newer files via the version check **before** parsing
entries, so a v5 vault can never be read as plain memory by an old client.

## Header

Fixed JSON with: `magic`, `version`, `dims`, `vocab_fingerprint`,
`encoder_name`, `created`, `name`, `description`, plus an optional `extra`
dict for forward-compatible metadata:

- `extra["cartridge_type"]`: `"memory"` (default) | `"agent"` | `"vault"`
- `extra["vault"]`: vault crypto metadata (schema, kdf, salt, wrap nonce,
  wrapped DEK, keychain service, created)
- `extra["agent"]`: agent metadata (schema, name, role, created, updated)

The vocab fingerprint is a SHA-256 over the encoder identity (`encoder_name`
+ `dims`) and the sorted vocab; two cartridges with the same fingerprint are
query-compatible.

## Entries

Per record: `flags (u16)` + `payload length (u32)` + `key_bits (dims/8 bytes)`
+ optional `binvec` (`u32` length + raw bytes) + payload.

Flag bits:

| Bit | Name | Meaning |
|-----|------|---------|
| 0x0001 | `FLAG_DELETED` | tombstone — ignored by reads |
| 0x0002 | `FLAG_BINVEC` | raw binary auxiliary-vector block present (v2+) |
| 0x0004 | `FLAG_COMPRESSED` | payload/binvec is lz4-compressed (v4+) |
| 0x0008 | `FLAG_ENCRYPTED` | payload is ciphertext (v5 vault+) |
| bits 4–11 | kind | knowledge kind, uint8 (v3+): 0=generic, then fact/skill/doc/turn |

Payloads are JSON (UTF-8). At 1024 dims a binvec block is 384 bytes
(128 key + 128 sem_key + 128 sem_ckeys); lz4 typically reduces that 40–80%
depending on data.

## Read-side limits (v1.0.0 hardening)

`read_cartridge` rejects malformed/hostile files with `FormatError`:

| Limit | Value |
|-------|-------|
| `dims` | 8 ≤ dims ≤ 8192, multiple of 8 |
| entries | ≤ 10,000,000 |
| vocab tokens | ≤ 1,000,000 |
| vocab token length | ≤ 1024 bytes |
| payload length (`pb_len`) | ≤ 64 MB |
| binvec length | ≤ 64 MB |
| decompressed payload | ≤ 64 MB |

Truncated files, bad magic, oversized counts, and decompression failures raise
`FormatError` instead of allocating unbounded memory. Fuzz tests live in
`tests/test_format_fuzz.py`.

## Concurrency and durability

- Writes are atomic: write to `<path>.tmp`, then rename over the target.
- A file lock (fcntl on Unix, `msvcrt.locking` on Windows with a `.lock`
  sidecar) serializes concurrent writers; the lock is released before the
  tmp→target swap so readers never block on a held lock.
- Windows atomic replace retries on `PermissionError` (Defender/indexer
  briefly holding the file) with exponential backoff.
