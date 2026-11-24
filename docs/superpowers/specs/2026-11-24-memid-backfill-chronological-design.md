# mem_id backfill + chronological ordering — design

- Date: 2026-11-24
- Status: proposed (awaiting approval)
- Scope: sqac (store / memid / cli / sqac_mcp)

## Intent & why

`mem_id` is the stable, immutable handle for an entry (used by `get`,
`delete_entry_by_mem_id`, `/entries/{mem_id}`, task graduation, S5
tombstoning). Entries created before mem_id existed have no id, and the
current backfill has two gaps that make "newest-first by mem_id" unreliable:

1. **Backfill is in-memory only.** `SqacStore.load()` calls
   `_backfill_mem_ids()`, but the minted ids are not written back unless
   something later calls `save()`. A cartridge that is only ever read gets
   *different* ids on every load → ids for legacy entries are not stable
   across restarts.
2. **Legacy order is scrambled.** Every backfilled entry is stamped "now,"
   so sorting by mem_id descending yields "all tied at backfill time, ties
   broken by random bits" — not the real newest-first order.

New entries are already fine: `add()` mints the id at creation, and a ULID
embeds its own millisecond timestamp, so descending mem_id = newest first
for anything created going forward.

## Goals

- Every live entry has a **stable, globally-unique** mem_id that survives
  restarts.
- mem_id descending order reflects **true newest-first** for both new and
  legacy entries.
- A read-only surface to list entries newest-first.
- Same-millisecond creations (batch `pack` / `teach` loops) preserve
  insertion order.

## Non-goals

- No importance ranking. mem_id is permanent; importance is dynamic
  (search/DMS) and must not influence an entry's sort position.
- No recovery of legacy entries' true wall-clock birthdate — it no longer
  exists. We recover *relative* order only.
- No `.sqac` format bump. mem_id already lives in entry `meta`; old readers
  ignore the key. Additive and downgrade-safe (standing backward-compat gate).

## Approaches considered

| Approach | Verdict | Why |
|---|---|---|
| **Monotonic ULID backfill in file order + explicit persist command** (picked) | ✅ | Recovers true relative order (entries are appended in creation order), ids stay globally unique, persistence is deliberate. |
| Deterministic-from-content ids (stable without writing) | ❌ | Ids collide across stores that share content; mem_id is a *global* handle (tombstones, `/entries`, `get_by_mem_id`). Global uniqueness wins. |
| Auto-save on load | ❌ | Side-effectful reads; churns mtimes and trips cloud sync on every load. |
| Importance-ordered legacy entries | ❌ | Sort position would drift as salience/access_count change — breaks the "permanent handle" promise. |

## The how (decisions made)

### 1. Monotonic ULID source (`sqac/memid.py`)

Add a monotonic mint alongside the existing `new_mem_id()`:

- Keep `new_mem_id()` as-is (random 80-bit suffix) for the one-off case.
- Add a monotonic generator that, within the same millisecond, increments
  the previous random component by 1 instead of drawing fresh randomness
  (classic ULID monotonicity). Strictly increasing ids → lexicographic sort
  of equal-length Crockford-base32 ids equals numeric/chronological order
  (the Crockford alphabet is in ASCII order, so string sort == value sort).
- First call after a gap (new ms) draws fresh randomness as today.

### 2. `add()` uses the monotonic source (call #1: include same-ms ordering)

`SqacStore.add()` mints via the monotonic generator so batch creates inside
one millisecond keep insertion order. The store holds the generator's last
(timestamp, random) state. This is the same code path the backfill uses, so
there is one ordering rule everywhere.

**Seed from existing ids on load (call #5).** `load()` scans existing ids,
decodes the max `(last_ms, last_random)`, and seeds the per-store generator.
Without this, a cartridge whose ids were minted later than "now" (clock skew,
restored backup, backfill-at-T then add-at-T' with the clock gone backwards)
would mint a new id that sorts *before* existing ones — breaking the
newest-first guarantee this change exists to provide. Cost: one O(n) scan at
load (the entry loop already runs).

### 3. Backfill writes ids in file order, monotonically (`sqac/store.py`)

`_backfill_mem_ids()` already walks entries in order. Change it to mint via
the monotonic source so successive legacy entries get strictly increasing
ids that match their file (= creation) position. Idempotent: entries that
already have an id are skipped; re-running is a no-op.

### 4. Explicit persist command (call #2: scope = both)

- `sqac backfill [--db PATH]` — loads the cartridge(s), runs the monotonic
  backfill, saves. **Default: every eligible cartridge in the memory dir**
  (`SQAC_MEM_DIR` / `~/.sqacm` if set, else `*.sqac` in cwd), so one command
  fixes the whole memory. `--db PATH` scopes to a single cartridge.
- Idempotent and safe to re-run.

**Skip non-memory cartridges (builder finding #1).** The memory dir holds
cartridges that are not SqacStore memory and must never be opened for
backfill: v5 encrypted `vaults/*.sqac` (load crashes — no `["content"]`),
`agents/*.sqac` (type `agent`), `secrets/*.sqac` (type `secret`),
`session.sqac`, `archives/*`, and checkpoint cartridges. Enumerate by
cartridge_type / path convention and skip them. Reuse `CartridgeRack`'s
directory filter (it already skips session/vault/agent) and extend it to also
skip `secret`.

**Save-if-changed (builder finding #3).** Backfill rewrites only cartridges
where it actually minted ≥1 id. A no-op run touches nothing — this preserves
the anti-churn rationale that killed auto-save-on-load. `_backfill_mem_ids()`
returns the count minted; the command saves iff count > 0.

**Preserve header on save (builder finding #2).** `SqacStore.save()` today
rebuilds the header with `name=""`, `description=""`, and no `extra`, so a
naive load→save **wipes the cartridge's name, description, and
cartridge_type**. Fix `save()` to preserve the loaded header's name /
description / cartridge_type (store them on the instance at `load()`, reuse
them as defaults when args are empty). Backfill depends on this; it also
fixes every other load→save roundtrip.

- (Optional, same PR) `mem_backfill` MCP tool wrapping the same operation for
  agents that can't shell out.
- Deliberately **not** auto-saving on load (see rejected approaches). Ids
  also persist naturally after any `mem_write` / `mem_save`, so the command
  is for making ids deterministic up front.

### 5. Read-only recency listing (Gap 3 surface)

- `SqacStore.list_by_recency(limit=None)` → list of
  `{"mem_id", "created_ms", "key_norm", "kind"}` for live entries, sorted by
  mem_id **descending** (newest first). `created_ms` decoded from the ULID's
  top 48 bits for display.
- **Inclusion (builder finding #4):** excludes tombstoned (`deleted`) entries
  AND `kind=task` entries — consistent with `search()`, which treats tasks as
  working memory with their own surface (`task_list` / attention block). A
  recency view that listed tasks would contradict the rest of the read path.
- `sqac ids [--db PATH] [--limit N]` prints it. Read-only, O(n), no writes.

## Backward compatibility

- No format/version bump. mem_id stays in entry `meta`; older readers ignore
  the key; backfill-on-load already tolerated missing ids.
- Cartridges written by this change load unchanged in older sqac (extra meta
  key is ignored). Downgrade-safe.
- CI fixture test both directions (old reader reads new cartridge, new reader
  reads old cartridge and backfills) — follows the standing rule.

## Left to the builder

- Exact placement of the monotonic state — must be per-store so two stores
  don't interfere, but the generator itself is stateless given
  `(last_ms, last_random)`. Seeded from max existing id at load (decided).
- Whether `mem_backfill` ships in this PR or a fast-follow (recommend same PR;
  it's a thin wrapper).
- Test matrix: same-ms monotonicity, file-order recovery on a synthetic
  legacy cartridge, idempotent re-run, save-if-changed no-op leaves mtime
  untouched, header preserved across backfill, clock-skew seed (load a
  cartridge with a future-dated id, add, assert new id sorts last),
  cross-reader downgrade fixture.

## Test plan

- Unit: monotonic ids strictly increase within a ms; legacy file order
  reproduced by descending sort.
- Integration: `backfill` then reload → ids stable across two loads.
- CLI: `sqac backfill`, `sqac ids --limit`, `--db` scoping.
- Fixture: v4 cartridge without mem_ids → backfill → reload in "old reader"
  mode (mem_id ignored) still loads.