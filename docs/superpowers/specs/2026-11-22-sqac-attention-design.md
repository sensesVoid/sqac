# SQAC Attention & Canonical mem_id — Design

Date: 2026-11-22
Status: Draft for approval
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
on every entry, plus one O(1) lookup by id** — and on top of it, an
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
  immediately on task-state change, and at session start.
- Finished tasks **graduate** to durable `kind=fact` memories (same
  mem_id preserved); they never appear in the block again.
- Task state is **hybrid**: explicit task tools are the primary path; a
  derived fallback (from checkpoints/turns) renders when no explicit
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

## Anti-goals

- The block must never exceed ~150 tokens; caps: 5 current, 10 pending
  (overflow shows "+N more").
- mem_ids never change once assigned — not on compact, shard split,
  save/load, export/import, or graduation.
- No silent task expiry or deletion; only explicit `task_done` removes
  from current/pending.
- No polling loops: cadence is turn-counted, push-on-change.

## Background: what exists today

- **Retrieval tiers**: exact (key_norm, O(1)), lexical/fuzzy (char
  trigram BSC + SIMD popcount), optional semantic (static embeddings).
  Confidence gate fails safe. Entries are addressed by key or by an
  internal list index that shifts on save/load/compact — there is **no
  stable public identifier today**.
- **`graduate`** (rack.py): a batch pass over a whole session store with
  a threshold and grader. No per-entry graduation exists.
- **`mem_checkpoint` / continuity.json**: free-text goal/summary at task
  boundaries. No task structure.
- **`ContextOffloader`**: 8-turn rolling window, debounced saves
  (10 turns / 60 s), DMS eviction. Session bucket is ephemeral recall
  cache.
- **Integrations middleware** (`sqac.integrations`): wraps
  OpenAI-compatible clients, hydrates the system prompt per call,
  observes turns. It sees every turn — the natural first home for
  periodic attention injection.
- **MCP server**: `mem_bootstrap` (one-shot context packet),
  `memory://context` resource, `mem_observe` (called every turn by the
  memory-keeper protocol). Tool results are how MCP agents see anything.
- **Events bus** (`sqac/events.py` + `GET /events` SSE + webhooks):
  mem_write / mem_graduate / mem_swap / mem_checkpoint, metadata only.

## Approaches (decided with the maintainer)

1. **Task state source** — chosen: *hybrid*. Explicit task tools
   primary; derived fallback (active goal from continuity/checkpoint +
   recent unreviewed turns) when no explicit tasks exist.
2. **Finished tasks** — chosen: *graduate to `kind=fact`*, same mem_id
   preserved; the block replaces the finished section with the mem_id
   handle (ids on current/pending entries make them exactly
   referenceable).
3. **Surfaces** — chosen: *all three*, integrations middleware first
   (zero agent behavior change), then MCP (block rides mem_bootstrap +
   mem_observe responses + memory://context), then HTTP (observe
   response + GET /attention + /entries/{mem_id}).
4. **Cadence** — chosen: *every 5 turns*, env `SQAC_ATTENTION_TURNS`,
   plus immediate re-injection on task-state change and session start.
5. **mem_id** — chosen: *global ULID in entry meta* (time-sortable,
   collision-free, no coordination), rack-maintained
   id -> (cartridge, entry) index, ids survive compact / shard split /
   save/load / export+import.
6. **Retrieval quality** — chosen: *diagnose first, in parallel*
   (recall benchmark), findings become follow-up gate/ranking PRs.

Rejected: agent-managed-only task list (forgetting failure mode),
derived-only (misclassification), per-cartridge sequential ids (agents
must track cartridge context), format-level id field (unnecessary v6
bump).

## Design

### A. mem_id + mem_get (PR 1)

**Generation.** New module `sqac/memid.py`: `new_mem_id() -> str` —
ULID: 48-bit ms timestamp + 80 bits `os.urandom`, Crockford base32,
26 chars, lexicographically time-sortable. No dependencies.

**Assignment.** In `SqacStore.add()` and every entry-creating path
(rack writes, `/teach`, offloader observe, cartridge import): if
`meta` lacks `mem_id`, assign one. Stored as `meta["mem_id"]`.

**Backfill.** On `SqacStore.load()`, entries without `mem_id` get one
assigned in memory; the rack's next save persists them. Documented
one-time migration cost; ids stable from first save onward.

**Index.** `CartridgeRack` maintains `_id_index: dict[str, tuple[str,
int]]` (mem_id -> (cartridge name, entry list index)). Rebuilt on
mount/load; updated on add, delete, compact, shard split, swap. In
memory only — the source of truth is the entries' meta.

**Lifecycle guarantees.**

- *Compact*: entries (and meta) are rewritten as-is — ids survive.
  Verified by test.
- *Shard split*: entries move with their meta — ids survive.
- *Export/import*: meta travels; on import, id collision with an
  existing entry triggers the existing overwrite guard semantics
  (documented; default refuse).
- *Graduation* (PR 2): the fact receives the **same mem_id** as the
  task it came from; the task entry is tombstoned. Identity is
  preserved; the index entry is re-pointed.

**Lookup surfaces.**

- MCP tool `mem_get(mem_id)` -> `{found, content, kind, source,
  cartridge, meta}` or honest not-found.
- CLI `sqac get <mem_id>`.
- HTTP `GET /entries/{mem_id}` (respects SQAC_SCOPE like search).
- All hit-bearing responses carry mem_id: `mem_search`,
  `mem_recall`/`mem_recall_detailed`, `/rack/search` (+ `/rack/trace`),
  `/session/recall`, integrations-injected memory lines.

### B. Attention block + task state (PR 2)

**Storage.** Tasks live in the rack as a `tasks` cartridge
(`kind=task` entries, auto-created like facts/docs/skills).
Task entry meta: `status: "current" | "pending"`, `created`,
`updated`. Content is the task text.

**Tools (MCP).**

- `task_add(content, status="pending")` -> returns entry + mem_id.
- `task_done(mem_id)` -> graduates: write content as `kind=fact` into
  the default facts cartridge **with the same mem_id**, tombstone the
  task entry, re-point the index. Emits `mem_graduate` event.
- `task_list(status=None)` -> entries with mem_ids.

**Derived fallback.** When the tasks cartridge has no current/pending
entries, the block renders: Current = active goal from
continuity.json / latest checkpoint; Pending = count + first lines of
unreviewed session turns since the last checkpoint. Honest, cheap,
never empty. Explicit `task_add` immediately takes over.

**Rendering.** Compact text:

```
[SQAC attention | turn 12]
Current:
- <content> (mem_id 01J8...)
Pending:
- <content> (mem_id 01J8...)
```

Caps 5/10, "+N more" overflow. <= ~150 tokens enforced by test.

**Cadence & triggers.**

- Turn counting per surface: middleware counts `create()` calls; MCP
  counts `mem_observe` calls; HTTP counts `/session/observe` calls.
  Counter is per-process, keyed by memory dir.
- Inject when `turns % SQAC_ATTENTION_TURNS == 0` (default 5), on
  task-state change (add/done), and at session start
  (mem_bootstrap includes the block).
- MCP delivery: block rides `mem_bootstrap`, `mem_observe` responses
  (when due), and the `memory://context` resource (always current).
- HTTP: `/session/observe` response carries it when due; new
  `GET /attention` always returns the current block.

### C. Retrieval diagnosis (PR 3, parallel)

- `experiments/recall_bench.py`: seeded corpora (multilingual,
  paraphrase pairs, code identifiers), measures per-tier recall and
  confidence distributions against ground truth.
- Collect 3-5 real-world failed queries from the maintainer's usage
  as additional cases.
- Output: a triage table — which tier fails which query class — and
  follow-up fixes (gate thresholds, ranking blend, query expansion)
  as separate small PRs informed by data.

## Backward compatibility (hard requirement)

- **No format bump.** Cartridge format stays v4/v5. `mem_id` rides inside
  the per-entry payload JSON as `meta["mem_id"]`; payload is
  `json.loads`-ed as an opaque dict, so 1.3.x readers parse cartridges
  written by this release without error and simply ignore the key.
- **Old cartridges, new code.** Entries lacking `mem_id` are backfilled
  in memory on load and persisted on the rack's next save. `mem_get` on
  a not-yet-backfilled id returns an honest not-found (never a wrong
  hit); after first save, all entries are addressable.
- **New code, old readers (downgrade).** Cartridges written with
  mem_ids remain readable by sqac 1.3.x (unknown meta key ignored);
  only `mem_get`/task tools are unavailable there. Documented as
  one-way feature availability, not data incompatibility.
- **Version skew between agents/servers.** New MCP tools (`mem_get`,
  `task_add`, `task_done`, `task_list`) are purely additive — old
  clients never call them, old servers never see them. The attention
  block is delivered as a NEW OPTIONAL field on `mem_observe` /
  `mem_bootstrap` responses (never a renamed/removed field), so strict
  clients that ignore unknown fields keep working; agents that read it
  gain the block.
- **Wire formats.** SSE event names unchanged (`mem_write`,
  `mem_graduate`, ...); task events reuse existing names with additive
  metadata. `GET /attention` and `GET /entries/{mem_id}` are new
  routes; existing routes' response shapes only gain fields.
- **Config.** All new env vars (`SQAC_ATTENTION_TURNS`, task caps) are
  optional with defaults; with none set, behavior is identical to
  1.3.x except for the presence of the block when tasks exist.
- **No state migration.** No rewrite of `continuity.json`, session
  cartridges, or rack layouts; existing memories keep working with or
  without ids.
- **Test gate.** CI must include: a 1.3.x-built cartridge fixture
  loaded by new code (backfill path), and a new-code-written cartridge
  parsed by the old reader path (payload JSON round-trip without the
  mem_id key) for every release candidate.

## Risks & mitigations

- **Agents forget task tools** -> derived fallback keeps the block
  truthful; task_add is one call.
- **Index staleness** (crash between entry write and index update) ->
  index is rebuildable from entries on load; serve misses by rebuild +
  retry.
- **mem_id collision on import** -> ULIDs make this practically
  impossible; overwrite guard covers the theoretical case.
- **Block staleness in MCP** (no system prompt) -> three delivery
  vehicles (bootstrap, observe-when-due, resource) plus the HTTP/middle-
  ware paths where injection is real.
- **Diagnosis scope creep** -> time-boxed benchmark; fixes are
  follow-up PRs, not part of PR 1/2.

## Rollout

1. **PR 1** — `sqac/memid.py`, assignment + backfill + rack index,
  `mem_get` on all three surfaces, mem_id on all hits. Tests: id
  stability across compact/split/save/load/export-import; O(1) lookup.
2. **PR 2** — tasks cartridge, task_add/task_done/task_list,
  graduation-preserving-id, attention renderer + cadence +
  injection on all surfaces, derived fallback. Tests: block caps,
  cadence triggers, graduation identity, fallback rendering.
3. **PR 3** — recall benchmark + triage report; fix PRs follow.

Version: 1.4.0. CHANGELOG + README (features: attention, mem_id) at
merge of PR 2; docs/ATTENTION.md.

## Left to the builder

- Exact ULID encoding details (Crockford alphabet, overflow
  handling) — sensible defaults fine.
- Whether `GET /attention` needs auth-tier treatment (lean: same
  _verify_key as everything).
- Test corpus composition for the recall bench.