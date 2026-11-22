# SQAC Attention & Canonical mem_id - Design

Date: 2026-11-22
Status: Draft for approval (builder-check settled items incorporated)
Target: v1.4.0 (three PRs, sequenced)

## Intent

Two problems, one design:

1. **Agents fail to retrieve memory that exists.** They must guess keys or
   re-describe content at search time; fuzzy/semantic tiers miss; there is
   no way to reference a memory precisely once found.
2. **Agents lose the thread of their own work.** Task state lives in
   free-text checkpoints and an 8-turn window; nothing tracks current vs
   pending work, and nothing re-injects it as the session runs.

The answer to both is the same primitive: **a canonical, stable `mem_id`
on every entry, plus one O(1) lookup by id** - and on top of it, an
**attention block** (Current / Pending tasks) that SQAC re-injects every
few turns, with finished tasks graduated to durable `kind=fact` memory so
the block stays flat forever.

## Goals

- Every entry (all kinds, all cartridges) carries a durable, globally
  unique `mem_id` (ULID, stored in entry `meta`).
- One lightweight lookup primitive: `mem_id -> entry` (content, kind,
  source, cartridge, meta). No search, no threshold, no fuzzy.
- Every search/recall hit carries its `mem_id`, on every surface.
- Attention block: Current Tasks + Pending Tasks, compact (<= ~150
  tokens), re-injected every N turns (default 5, `SQAC_ATTENTION_TURNS`),
  on the task tool's own response after any task change, and at session
  start.
- Finished tasks **graduate** to durable `kind=fact` memories (same
  mem_id preserved); they never appear in the block again.
- Task state is **hybrid**: explicit task tools are the primary path; a
  derived fallback (from continuity + turns) renders when no explicit
  tasks exist, so the block is never empty.
- All surfaces: integrations middleware first, then MCP, then HTTP.
- A parallel **retrieval diagnosis** workstream (recall benchmark per
  tier) whose findings feed cheap gate/ranking fixes.

## Non-goals

- No on-disk format change (v6). mem_id rides in entry `meta`; readers
  of older cartridges are unaffected.
- No new runtime dependencies (ULID implemented in ~30 lines of stdlib).
- No auto-graduation: only `task_done` graduates.
- No finished-tasks list in the block, by design.
- `mem_get` does not search; not-found is an honest miss and the agent
  falls back to search.
- No entry-level merge import; cartridge import stays a file replace.

## Anti-goals

- The block must never exceed ~150 tokens; caps: 5 current, 10 pending
  (overflow shows "+N more").
- mem_ids never change once assigned - not on compact, shard split,
  save/load, export/import, or graduation.
- No silent task expiry or deletion; only explicit `task_done` removes
  from current/pending.
- No polling loops: cadence is turn-counted; no cross-process push
  channel is built.
- `mem_recall`'s plain-text format does not change.

## Background: what exists today

- **Retrieval tiers**: exact (key_norm, O(1)), lexical/fuzzy (char
  trigram BSC + SIMD popcount), optional semantic (static embeddings).
  Confidence gate fails safe. Entries are addressed by key or by an
  internal list index that shifts on save/load/compact - there is **no
  stable public identifier today**.
- **`graduate`** (rack.py): a batch pass over a whole session store with
  a threshold and grader. No per-entry graduation exists.
- **`mem_checkpoint` / continuity.json**: free-text goal/summary at task
  boundaries. No task structure. (Pre-existing bug found by builder
  check: `_MemoryState` never calls `ContinuityStore.load()`, so the
  goal is always empty in fresh MCP processes; `server.py` has no
  continuity store at all. PR 2 fixes both - see D2.)
- **`ContextOffloader`**: 8-turn rolling window, debounced saves
  (10 turns / 60 s), DMS eviction. Session bucket is ephemeral recall
  cache. One exchange writes 2-3 store entries (one per key).
- **Integrations middleware** (`sqac.integrations`): wraps
  OpenAI-compatible clients, hydrates the system prompt per call,
  observes turns. It sees every turn - the natural first home for
  periodic attention injection.
- **MCP server**: `mem_bootstrap` (one-shot context packet),
  `memory://context` resource, `mem_observe` (called every turn by the
  memory-keeper protocol). Tool results are how MCP agents see anything.
- **Events bus** (`sqac/events.py` + `GET /events` SSE + webhooks):
  mem_write / mem_graduate / mem_swap / mem_checkpoint, metadata only.

## Approaches (decided with the maintainer)

1. **Task state source** - chosen: *hybrid*. Explicit task tools
   primary; derived fallback when no explicit tasks exist.
2. **Finished tasks** - chosen: *graduate to `kind=fact`*, same mem_id
   preserved; the block replaces the finished section with the mem_id
   handle.
3. **Surfaces** - chosen: *all three*, integrations middleware first,
   then MCP, then HTTP.
4. **Cadence** - chosen: *every 5 turns*, env `SQAC_ATTENTION_TURNS`,
   plus session start and task-change refresh.
5. **mem_id** - chosen: *global ULID in entry meta*, store-level index.
6. **Retrieval quality** - chosen: *diagnose first, in parallel*.

Rejected: agent-managed-only task list (forgetting failure mode),
derived-only (misclassification), per-cartridge sequential ids, v6
format field.

## Settled by builder check (my calls, flagged for maintainer veto)

- **S1 - Index placement.** Store-level, not rack-only:
  `SqacStore._id_index: dict[str, int]`, rebuilt in `load()`/`compact()`,
  maintained in `add()`/`delete()`. The rack composes per-cartridge
  indexes. Surfaces without a rack (`sqac get`, HTTP default store,
  offloader recall) use the store index directly - one O(1) path.
- **S2 - Exchange ids.** One store entry = one mem_id (an exchange's
  2-3 entries each have their own; they are distinct memories).
  `mem_recall` text format unchanged; `mem_recall_detailed`,
  `mem_bootstrap.recent` and `/session/recall` gain a `mem_ids` array
  aligned with their entries.
- **S3 - "Immediate" re-injection.** Defined as: the task tool's own
  response (`task_add`/`task_done` result) carries the refreshed block.
  Other surfaces converge on their next turn. No cross-process push.
- **S4 - Import collisions.** Cartridge import is a file replace;
  per-entry collision detection is N/A in PR 1. Ids travel inside the
  replaced file; the file-level overwrite guard applies as today.
- **S5 - Rewrite identity.** mem_id identifies an immutable entry:
  upsert (`mem_write` same key) appends a new entry with a NEW id and
  tombstones the old (old id = honest miss). `sparsify` relocation and
  compact-dedupe survivors KEEP their id. Autobuild of changed content
  = new id. Two live entries never share an id; `mem_get` skips
  tombstones.
- **S6 - Task text trust.** Task text is agent-authored, trusted by
  construction (same posture as existing middleware injection);
  documented, no new scanning in PR 2.
- **S7 - PR 3 shape.** Report-first: synthetic multilingual corpora at
  build time, no pass/fail bar yet; maintainer's real failed queries
  requested when PR 3 starts; fixes are follow-up PRs.

## Design

### A. mem_id + mem_get (PR 1)

**Generation.** New module `sqac/memid.py`: `new_mem_id() -> str` -
ULID: 48-bit ms timestamp + 80 bits `os.urandom`, Crockford base32,
26 chars, lexicographically time-sortable. No dependencies.

**Assignment.** In `SqacStore.add()` and every entry-creating path
(rack writes, `/teach`, offloader observe, import): if `meta` lacks
`mem_id`, assign one. Stored as `meta["mem_id"]`.

**Backfill.** On `SqacStore.load()`, entries without `mem_id` get one
in memory; the next save persists them. One-time migration cost.

**Index.** See S1.

**Lifecycle guarantees.**

- *Compact / shard split / save-load*: entries move with their meta -
  ids survive. Verified by test.
- *Export/import*: ids travel inside the file (S4).
- *Graduation* (PR 2): the fact receives the **same mem_id** as the
  task it came from; the task entry is tombstoned; the index is
  re-pointed (S5 semantics).
- *Upsert / sparsify / dedupe*: per S5.

**Lookup surfaces.**

- MCP tool `mem_get(mem_id)` -> `{found, content, kind, source,
  cartridge, meta}` or honest not-found.
- CLI `sqac get <mem_id>`.
- HTTP `GET /entries/{mem_id}` (respects SQAC_SCOPE like search).
- All hit-bearing responses carry mem_id: `mem_search`,
  `mem_recall`/`mem_recall_detailed` (per S2), `/rack/search` (+
  `/rack/trace`), `/session/recall`, integrations-injected lines.

### B. Attention block + task state (PR 2)

**Storage.** Tasks live in the rack as a `tasks` cartridge
(`kind=task` entries, auto-created like facts/docs/skills). Task entry
meta: `status: "current" | "pending"`, `created`, `updated`,
`project` (see Q2 below).

**Tools (MCP).**

- `task_add(content, status="pending")` -> entry + mem_id.
- `task_done(mem_id)` -> graduate: write content as `kind=fact` into
  the default facts cartridge **with the same mem_id**, tombstone the
  task entry, re-point the index. Emits `mem_graduate` event.
- `task_list(status=None)` -> entries with mem_ids.

**Derived fallback.** When the tasks cartridge has no current/pending
entries for the active project, the block renders: Current = active
goal from continuity.json (requires D2 fixes); Pending = session turns
after the latest checkpoint's turn marker. The marker is recorded by
`mem_checkpoint` into the offloader state (persisted; survives save and
restart; `mem_swap` resets it as today).

**Rendering.** Compact text:

```
[SQAC attention | turn 12]
Current:
- <content> (mem_id 01J8...)
Pending:
- <content> (mem_id 01J8...)
```

Caps 5/10, "+N more" overflow, <= ~150 tokens enforced by test.

**Cadence & triggers.** Turn counting per surface: middleware counts
`create()` calls; MCP counts `mem_observe` calls; HTTP counts
`/session/observe` calls (per-process, keyed by memory dir). Inject
when `turns % SQAC_ATTENTION_TURNS == 0` (default 5), on session start
(mem_bootstrap includes the block), and on the task tool's own response
after add/done (S3).

**Delivery per surface.**

- MCP: block rides `mem_bootstrap`, `mem_observe` responses (when due),
  the `memory://context` resource (always current), and task tool
  responses (S3).
- Integrations middleware: true system-prompt injection (extend
  `_augment`); block appended when due.
- HTTP: `/session/observe` response carries it when due; new
  `GET /attention` always returns the current block.

**D2 - continuity fixes riding PR 2** (pre-existing bugs, required by
the fallback): `_MemoryState.__init__` calls `ContinuityStore.load()`;
`server.py` gains a `ContinuityStore` with save lifecycle on
checkpoint/observe.

### C. Retrieval diagnosis (PR 3, parallel)

- `experiments/recall_bench.py`: seeded multilingual corpora,
  paraphrase pairs, code identifiers; per-tier recall + confidence
  distributions vs ground truth.
- Maintainer's 3-5 real failed queries added as cases (requested at
  PR 3 start).
- Output: triage table (which tier fails which query class) + follow-up
  fix PRs (gate thresholds, ranking blend, query expansion) informed by
  data. No pass/fail bar until the table exists (S7).

## Backward compatibility (hard requirement)

- **No format bump.** Format stays v4/v5. `mem_id` rides inside the
  per-entry payload JSON as `meta["mem_id"]`; payload is `json.loads`-ed
  as an opaque dict, so 1.3.x readers parse new cartridges without
  error and ignore the key.
- **Old cartridges, new code.** Backfill on load (A); `mem_get` on a
  not-yet-backfilled id = honest not-found; after first save, all
  entries are addressable.
- **Downgrade.** New cartridges remain readable by 1.3.x; only the new
  tools/endpoints are unavailable there.
- **Version skew.** New MCP tools are additive. The attention block is
  a NEW OPTIONAL field on `mem_observe`/`mem_bootstrap` responses -
  never a renamed/removed field. SSE event names unchanged; task events
  reuse `mem_graduate`/`mem_write` with additive metadata. New HTTP
  routes only; existing routes gain fields only.
- **Config.** New env vars optional with defaults; unset = 1.3.x
  behavior plus the block when tasks exist.
- **No state migration** of continuity.json, session cartridges, or
  rack layouts.
- **CI gate.** Per release candidate: a 1.3.x-built cartridge fixture
  loaded by new code (backfill path), and a new-code-written cartridge
  parsed by the old reader path (payload JSON round-trip without the
  mem_id key).

## Decided by maintainer (post builder-check)

- **Q1 - Tasks are EXCLUDED from ordinary retrieval.** `kind=task`
  becomes a new first-class kind (VALID_KINDS + kind-exclusion across
  `mem_search`, `rack.search`, offloader, integrations injection,
  dashboard); `task_list` and the attention block are their only
  surface. Tasks do NOT get a rack route beyond their own cartridge.
- **Q2 - Tasks are PER-PROJECT.** `task_add` stamps `project` (same
  `_project_name` derivation as mem_checkpoint: cwd basename or
  SQAC_PROJECT); `task_list` and the attention block filter by the
  active project; `task_done` rejects cross-project ids; HTTP shows
  all tasks with an optional `?project=` filter.

## Risks & mitigations

- **Agents forget task tools** -> derived fallback keeps the block
  truthful; `task_add` is one call.
- **Index staleness** (crash between write and index update) -> index
  rebuildable from entries on load; serve misses by rebuild + retry.
- **Block staleness in MCP** (no system prompt) -> bootstrap + observe-
  when-due + resource + task-response delivery (S3).
- **Diagnosis scope creep** -> time-boxed benchmark; fixes are
  follow-up PRs.

## Rollout

1. **PR 1** - `sqac/memid.py`, assignment + backfill + store index,
   `mem_get` on all three surfaces, mem_id on all hits. Tests: id
   stability (compact/split/save/load/export-import per S5), O(1)
   lookup, backfill.
2. **PR 2** - tasks cartridge + Q1/Q2 resolutions, task tools,
   graduation-preserving-id, attention renderer + cadence + injection
   on all surfaces, derived fallback, D2 continuity fixes. Tests: caps,
   cadence triggers, graduation identity, fallback, BC fixtures.
3. **PR 3** - recall benchmark + triage report; fix PRs follow.

Version 1.4.0. CHANGELOG + README + docs/ATTENTION.md at PR 2 merge.

## Left to the builder

- ULID encoding details (Crockford alphabet, overflow handling).
- `GET /attention` auth (lean: same `_verify_key`).
- Recall bench corpus composition.