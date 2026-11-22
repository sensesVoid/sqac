# SQAC Attention & Canonical mem_id — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every SQAC memory entry a stable, canonical `mem_id` and an O(1) lookup; add an attention block (Current/Pending tasks, finished tasks graduate to `kind=fact`) re-injected every 5 turns; run a retrieval diagnosis in parallel.

**Architecture:** ULID ids live in entry `meta` (no format bump). `SqacStore` gains a store-level id index rebuilt on load/compact and maintained on add/delete. Tasks are a new first-class kind in a per-project `tasks` cartridge; `task_done` graduates an entry to `kind=fact` preserving its id. The attention block is rendered from task entries + a continuity-derived fallback and injected by the integrations middleware (primary), MCP responses, and HTTP. Retrieval diagnosis is a standalone benchmark experiment.

**Tech Stack:** Python 3.10+, stdlib only (ULID hand-rolled), existing sqac store/rack/offloader/server/mcp/integrations. No new deps.

**Spec:** `docs/superpowers/specs/2026-11-22-sqac-attention-design.md`

## Global Constraints

- No new runtime dependencies; backward compatible (no format bump, additive wire fields only).
- mem_ids never change once assigned (not on compact, shard split, save/load, export/import, graduation).
- All new env vars optional with defaults; unset = 1.3.x behavior.
- `mem_recall` plain-text format unchanged.
- Kind=task is first-class: excluded from ordinary search by default.
- Tasks are per-project (stamp `project` via `_project_name`).
- Backward-compat CI gate: 1.3.x cartridges load in new code; new cartridges parse in old readers.

## Review Focus

- **Old cartridges on new code:** a 1.3.x-built cartridge must load, get ids backfilled in memory, and serve honest not-found for old ids until first save. (Backfill + index rebuild.)
- **New cartridges on old readers:** `meta["mem_id"]` must be ignored by 1.3.x (payload is opaque JSON dict) — verified by a round-trip test reading the new file with the old reader path.
- **Id stability across the lifecycle:** compact, shard split, save/load, export/import, graduation — all must preserve ids; no two live entries share an id.
- **Attention block token budget:** renderer must enforce <= 150 tokens with 5/10 caps and "+N more" overflow; verified by test.
- **Task tool forgetting:** derived fallback must render when no explicit tasks exist, so the block is never empty.
- **Turn counting correctness:** middleware counts `create()` calls; MCP counts `mem_observe`; HTTP counts `/session/observe`. Must not double-count or skip.

---

## Phase 1 — mem_id + mem_get (PR 1)

### Task 1: mem_id generation module

**Files:**
- Create: `sqac/memid.py`
- Create: `tests/test_memid.py`

**Interfaces:**
- Produces: `new_mem_id() -> str` (26-char Crockford base32 ULID, time-sortable)

- [ ] **Step 1: Write the failing test**

```python
import time
from sqac.memid import new_mem_id

def test_format_and_sortability():
    a = new_mem_id()
    time.sleep(0.002)
    b = new_mem_id()
    assert len(a) == 26 and len(b) == 26
    assert a < b  # time-sortable
    assert all(c in "0123456789ABCDEFGHJKMNPQRSTVWXYZ" for c in a)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_memid.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement `new_mem_id` in `sqac/memid.py`**

48-bit ms timestamp + 80 bits `os.urandom`, Crockford base32, 26 chars.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_memid.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sqac/memid.py tests/test_memid.py
git commit -m "feat(memid): ULID generator (stdlib only)"
```

### Task 2: Store assignment + index + backfill

**Files:**
- Modify: `sqac/store.py` (SqacStore.__init__, add, load, compact, delete)
- Test: `tests/test_memid_store.py`

**Interfaces:**
- Consumes: `new_mem_id()` from `sqac.memid`
- Produces: `SqacStore._id_index: dict[str, int]`, `SqacStore.get(mem_id) -> Optional[dict]`, `meta["mem_id"]` on every entry

- [ ] **Step 1: Write the failing tests** (assign on add, backfill on load, O(1) get, index maintained on delete)

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Implement** assignment in `add()`, backfill in `load()`, index in `__init__`/`load`/`compact`/`delete`, `get(mem_id)` method.

- [ ] **Step 4: Run tests to verify they pass**

- [ ] **Step 5: Commit**

### Task 3: Compact / shard split preserve ids

**Files:**
- Modify: `sqac/rack.py` (`compact`, shard split paths)
- Test: `tests/test_memid_store.py` (extend)

- [ ] **Step 1: Write failing tests** (compact preserves ids; shard split preserves ids)

- [ ] **Step 2-4: Implement + verify**

Compact already rewrites entries as-is; ensure meta (and thus mem_id) survives. Shard split moves entries with meta. Add assertions.

- [ ] **Step 5: Commit**

### Task 4: Export/import id preservation

**Files:**
- Modify: `sqac/export.py` (export includes mem_id in JSON; import preserves)
- Test: `tests/test_export_diff.py` (extend) or `tests/test_memid_store.py`

- [ ] **Step 1-5: Implement + verify + commit**

Export JSON entries already include meta? Check — export `_entries` currently does NOT include meta. Add `meta` to exported entries so mem_id travels. Import path (server.py cartridge import) restores payload as-is (meta included) — verify.

### Task 5: mem_get on all surfaces

**Files:**
- Modify: `sqac_mcp.py` (add `mem_get` tool)
- Modify: `sqac/cli.py` (add `sqac get <mem_id>` command)
- Modify: `sqac/server.py` (add `GET /entries/{mem_id}`)
- Test: `tests/test_mem_get.py`

- [ ] **Step 1: Write failing tests** (MCP mem_get returns entry; CLI sqac get prints content; HTTP GET /entries/{id} returns entry; not-found returns honest miss)

- [ ] **Step 2-4: Implement + verify**

MCP: `mem_get(mem_id)` tool — rack-level lookup (search all in-scope cartridges' indexes); fallback to linear scan if index miss (rebuild). CLI: `sqac get <mem_id>` (loads the cartridge or rack dir). HTTP: `GET /entries/{mem_id}` (scope-aware).

- [ ] **Step 5: Commit**

### Task 6: mem_id on all hit responses

**Files:**
- Modify: `sqac/store.py` (search results carry mem_id in hit meta or top-level)
- Modify: `sqac/rack.py` (rack search/trace hits carry mem_id)
- Modify: `sqac/offloader.py` (recall results carry mem_id)
- Modify: `sqac/integrations.py` (injected memory lines include mem_id)
- Modify: `sqac/server.py` (rack/search, rack/trace, session/recall responses)
- Modify: `sqac_mcp.py` (mem_search, mem_recall, mem_recall_detailed, mem_bootstrap.recent)
- Test: extend existing test suites

- [ ] **Step 1: Write failing tests** (each surface's hits include mem_id)

- [ ] **Step 2-4: Implement + verify**

Hit shape: add `mem_id` to the hit dict/meta. For recall text format: UNCHANGED (per settled S2); recall_detailed gains `mem_ids` array.

- [ ] **Step 5: Commit**

### Task 7: Phase 1 integration + backward-compat gate

**Files:**
- Test: `tests/test_memid_compat.py` (BC fixtures)
- Create fixture: `tests/fixtures/cartridge_1.3.x.sqac` (build by checking out sqac 1.3.1? Or synthesize via code)

- [ ] **Step 1: Write BC tests**
  - Load a 1.3.x-built cartridge (entries without mem_id) → backfill assigns ids in memory; search works; save persists ids.
  - New-code-written cartridge parsed by old reader path: payload JSON round-trips, mem_id key ignored.
- [ ] **Step 2: Run full suite**
- [ ] **Step 3: Commit + PR 1**

---

## Phase 2 — Attention + tasks (PR 2, depends on Phase 1)

### Task 8: Task cartridge + tools + graduation

**Files:**
- Create: `sqac/tasks.py` (task_add, task_done, task_list logic — or integrate into rack/store)
- Modify: `sqac_mcp.py` (add `task_add`, `task_done`, `task_list` tools; add `kind=task` to VALID_KINDS)
- Modify: `sqac/store.py` (add KIND_TASK = 5, KIND_NAMES entry)
- Modify: `sqac/rack.py` (_RACK_ROUTES: add "task": "tasks"; auto-create)
- Modify: `sqac/cli.py` (add task subcommands? optional — tools first)
- Test: `tests/test_tasks.py`

- [ ] **Step 1: Write failing tests** (task_add returns mem_id; task_done graduates to fact preserving id; task_list filters status; cross-project rejection; derived fallback when empty)

- [ ] **Step 2-4: Implement + verify**

`task_done(mem_id)`: find entry by id (index), read content, write as `kind=fact` via rack to facts with same mem_id, tombstone task entry, re-point index. Emit `mem_graduate` event.

- [ ] **Step 5: Commit**

### Task 9: Attention renderer + cadence

**Files:**
- Create: `sqac/attention.py` (render_block(turn, current, pending) -> str; caps; overflow)
- Modify: `sqac/offloader.py` (turn counter for attention cadence; last_checkpoint_turn marker)
- Test: `tests/test_attention.py`

- [ ] **Step 1: Write failing tests** (caps 5/10, overflow "+N more", <=150 tokens, cadence trigger, immediate on task change)

- [ ] **Step 2-4: Implement + verify**

- [ ] **Step 5: Commit**

### Task 10: Injection per surface

**Files:**
- Modify: `sqac/integrations.py` (_Completions.create: inject attention block when due; turn counting)
- Modify: `sqac_mcp.py` (mem_bootstrap includes block; mem_observe returns block when due; memory://context resource)
- Modify: `sqac/server.py` (GET /attention; /session/observe carries block when due; ContinuityStore load fix; server continuity store)
- Test: `tests/test_attention_surfaces.py`

- [ ] **Step 1: Write failing tests** (middleware injects block; MCP mem_observe response carries block when due; HTTP /attention returns block)

- [ ] **Step 2-4: Implement + verify**

Continuity fixes (D2): `_MemoryState.__init__` calls `ContinuityStore.load()`; `server.py` gains `ContinuityStore` with save lifecycle on checkpoint/observe.

- [ ] **Step 5: Commit**

### Task 11: Derived fallback

**Files:**
- Modify: `sqac/attention.py` (render with fallback: current = continuity goal; pending = turns after last checkpoint marker)
- Modify: `sqac/offloader.py` (record last_checkpoint_turn in state, persist)
- Test: extend `tests/test_attention.py`

- [ ] **Step 1-5: Implement + verify + commit**

### Task 12: Phase 2 integration + tests

**Files:**
- Test: `tests/test_tasks_integration.py`
- Modify: `docs/ATTENTION.md` (new)

- [ ] **Step 1-4: Verify + commit + PR 2**

---

## Phase 3 — Retrieval diagnosis (PR 3, parallel from PR 1)

### Task 13: Recall benchmark

**Files:**
- Create: `experiments/recall_bench.py`
- Create: `experiments/recall_corpora/` (seeded multilingual + paraphrase + code id corpora)

- [ ] **Step 1: Write benchmark harness** (per-tier recall + confidence distributions vs ground truth)

- [ ] **Step 2: Run on current codebase, produce triage table**

- [ ] **Step 3: Commit + report findings** (follow-up fix PRs informed by data)

---

## Self-Review

1. **Spec coverage:** mem_id generation ✓ (Task 1), assignment+index+backfill ✓ (Task 2), lifecycle guarantees ✓ (Tasks 3-4), mem_get surfaces ✓ (Task 5), mem_id on hits ✓ (Task 6), BC gate ✓ (Task 7), task cartridge+tools ✓ (Task 8), attention renderer+cadence ✓ (Task 9), injection per surface ✓ (Task 10), derived fallback ✓ (Task 11), diagnosis ✓ (Task 13). All spec sections mapped.

2. **Step scan:** Each step has test/code/run/commit. No "handle edge cases" vagueness. Task 2 and 6 are the biggest; their tests pin exact behavior.

3. **Type consistency:** `new_mem_id() -> str`, `SqacStore.get(mem_id) -> Optional[dict]`, `task_add(content, status="pending") -> dict` (with mem_id), `task_done(mem_id) -> dict`, `render_block(turn, current, pending) -> str`. Consistent throughout.

4. **Review Focus:** All six input classes have tests — old-cartridge backfill (Task 7), id stability (Tasks 3-4), token budget (Task 9), derived fallback (Task 11), turn counting (Task 10), backward-compat (Task 7).

5. **Proportion:** Plan is ~200 lines for a 3-PR feature with ~15 tasks. Reasonable; each task is one test cycle + commit.
