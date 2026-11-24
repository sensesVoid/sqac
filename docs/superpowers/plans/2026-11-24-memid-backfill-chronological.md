# mem_id Backfill + Chronological Ordering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every sqac entry a stable, globally-unique mem_id whose descending sort order is true newest-first, for both new and legacy entries, with an explicit idempotent `sqac backfill` to persist legacy ids and a read-only `sqac ids` recency listing.

**Architecture:** A monotonic ULID generator (same-millisecond increments instead of fresh randomness) becomes the single minting path. `SqacStore` holds the generator, seeds it from the max existing id on load (so clock skew can't mint a past-sorting id), and routes `add()`/ `compact()`/ `_backfill_mem_ids()` through it. Backfill now returns a minted count so the CLI can save-if-changed; `save()` is fixed to preserve the cartridge header. `list_by_recency()` surfaces live non-task entries newest-first.

**Tech Stack:** Python 3.10+, stdlib only. pytest. Existing `sqac` modules (`memid.py`, `store.py`, `cli.py`, `sqac_mcp.py`).

**Spec:** `docs/superpowers/specs/2026-11-24-memid-backfill-chronological-design.md`

## Global Constraints

- No new runtime dependencies (stdlib only).
- No `.sqac` format or version bump. mem_id stays in entry `meta`; older readers ignore the key; downgrade-safe.
- Windows/macOS/Linux parity (no POSIX-only path tricks).
- mem_id format unchanged: 26-char Crockford base32 ULID (`0123456789ABCDEFGHJKMNPQRSTVWXYZ`).
- Backward compatible: a cartridge written by this change loads unchanged in older sqac.
- ruff clean; full suite green before each commit.

## Review Focus

- **Legacy relative order** — backfilling a cartridge whose entries were appended A,B,C must yield ids where C sorts newest; descending order reproduces file-reverse. (Task 2 test.)
- **Clock-skew / restored backup** — a cartridge with a future-dated id, then `add()`, must mint an id that still sorts *after* the existing one. (Task 2 test.)
- **Non-memory cartridge safety** — `sqac backfill` must never open vault(v5)/agent/secret/session/archive/checkpoint cartridges (load crashes on encrypted payload). (Task 4 test.)
- **Header preservation** — load→backfill→save must not wipe name/description/cartridge_type. (Task 3 test.)
- **No-op runs are truly no-op** — a second `backfill` on an already-backfilled cartridge rewrites nothing (mtime unchanged). (Task 4 test.)

---

### Task 1: Monotonic ULID generator + ms decoder

**Files:**
- Modify: `sqac/memid.py`
- Test: `tests/test_memid_monotonic.py`

**Interfaces:**
- Consumes: nothing (leaf module).
- Produces:
  - `class MonotonicMemId` with:
    - `.mint() -> str` — 26-char ULID, strictly increasing per instance.
    - `.seed(timestamp_ms: int, randomness: int) -> None` — resume from a prior high-water mark.
    - `.seed_from(mem_id: str) -> None` — decode an existing id (Crockford-aware) and seed from it.
  - `decode_mem_id(mem_id: str) -> tuple[int, int]` — returns `(timestamp_ms, randomness)` from a 26-char ULID.
  - `decode_mem_id_ms(mem_id: str) -> int` — convenience wrapper returning just the ms.

**CRITICAL decoding note:** mem_id chars are **Crockford base32** (`0123456789ABCDEFGHJKMNPQRSTVWXYZ`, skips I/L/O/U, so J=18, Z=31). This is **not** Python's `int(s, 32)` (which uses 0-9a-v and rejects 'Z'). All randomness decoding must go through the Crockford alphabet — never `int(x, 32)`. Use `seed_from(mem_id)` so callers never hand-decode randomness.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_memid_monotonic.py
from sqac.memid import MonotonicMemId, new_mem_id, decode_mem_id, decode_mem_id_ms

def test_same_ms_is_strictly_increasing():
    g = MonotonicMemId()
    ids = [g.mint() for _ in range(1000)]
    assert ids == sorted(ids)
    assert len(set(ids)) == 1000

def test_decode_roundtrips_timestamp():
    mid = new_mem_id()
    ms, rand = decode_mem_id(mid)
    assert isinstance(ms, int) and ms > 1_600_000_000_000  # plausible epoch ms
    assert decode_mem_id_ms(mid) == ms

def test_seed_from_resumes_above_highwater():
    g1 = MonotonicMemId()
    last = g1.mint()
    g2 = MonotonicMemId()
    g2.seed_from(last)
    assert g2.mint() > last

def test_crockford_letters_decode_without_crash():
    # 'Z' (Crockford value 31) would raise under Python int(s, 32)
    mid = new_mem_id()
    ms, _ = decode_mem_id(mid)
    assert ms > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_memid_monotonic.py -v`
Expected: FAIL (ImportError: cannot import name 'MonotonicMemId')

- [ ] **Step 3: Implement `MonotonicMemId` and decoders in `sqac/memid.py`**

Reuse the existing `_CROCKFORD` alphabet and the `(timestamp_ms << 80) | randomness` layout (128-bit value, 26×5=130 bits, top 2 bits zero). Add a private `_crockford_decode(s: str) -> int` that folds each char via `_CROCKFORD.index(ch)` as `value = value * 32 + idx`. `decode_mem_id`: `value = _crockford_decode(mem_id)`, return `(value >> 80, value & ((1 << 80) - 1))`. `MonotonicMemId.mint()`: if current ms > `_last_ms`, draw fresh `os.urandom(10)` randomness and set `_last_ms`; else reuse `_last_rand + 1` (carry into `_last_ms` if it overflows 80 bits). `seed(ms, rand)`: set `_last_ms=ms`, `_last_rand=rand`. `seed_from(mem_id)`: `ms, rand = decode_mem_id(mem_id); self.seed(ms, rand)`. Build the id string by emitting 26 Crockford chars from the 128-bit value, most-significant first (reverse of the existing `new_mem_id` loop).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_memid_monotonic.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add sqac/memid.py tests/test_memid_monotonic.py
git commit -m "feat(memid): monotonic ULID generator + ms decoder"
```

---

### Task 2: Wire the store to the monotonic source (seed on load, mint everywhere)

**Files:**
- Modify: `sqac/store.py:28` (import), `sqac/store.py:226-238` (init state), `sqac/store.py:362-363` (add), `sqac/store.py:456-457` (compact), `sqac/store.py:533-540` (backfill), `sqac/store.py:822` (after `_rebuild_id_index` in `load`)
- Test: `tests/test_store_memid_order.py`

**Interfaces:**
- Consumes: `MonotonicMemId` (`.mint()`, `.seed_from(mem_id)`) and `decode_mem_id_ms` from Task 1.
- Produces:
  - `SqacStore._memid_gen: MonotonicMemId` (instance state).
  - `_backfill_mem_ids(self) -> int` now **returns the count minted** (was `-> None`).
  - All three mint sites call `self._memid_gen.mint()`.
  - `load()` seeds via `store._memid_gen.seed_from(max_id)` (Crockford-safe; no manual randomness decode).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_store_memid_order.py
import time
from sqac.store import SqacStore
from sqac.memid import decode_mem_id_ms

def test_batch_add_sorts_in_creation_order():
    st = SqacStore(semantic=False)
    ids = [st._entries[st.add(f"fact {i}", key=f"k{i}", validate=False)]["meta"]["mem_id"]
           for i in range(200)]
    assert ids == sorted(ids)  # same-ms batch stays monotonic

def test_descending_recap_is_newest_first():
    st = SqacStore(semantic=False)
    for i in range(50):
        st.add(f"entry {i}", key=f"key-{i}", validate=False)
    ids = [e["meta"]["mem_id"] for e in st._entries]
    assert ids == sorted(ids)  # creation order == ascending id order

def test_load_seeds_above_existing_future_id():
    import tempfile, os
    st = SqacStore(semantic=False)
    st.add("anchored", key="anchor", validate=False)
    future_ms = int(time.time() * 1000) + 10_000_000  # 10s in the future
    # forge a future-dated id by seeding a gen and stamping the entry
    from sqac.memid import MonotonicMemId
    g = MonotonicMemId(); g.seed(future_ms, 0)
    st._entries[0]["meta"]["mem_id"] = g.mint()
    d = tempfile.mkdtemp()
    p = os.path.join(d, "c.sqac"); st.save(p)
    st2 = SqacStore.load(p)
    idx = st2.add("later entry", key="later", validate=False)
    assert decode_mem_id_ms(st2._entries[idx]["meta"]["mem_id"]) >= future_ms
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_store_memid_order.py -v`
Expected: FAIL (no `_memid_gen`)

- [ ] **Step 3: Implement**

In `__init__` add `self._memid_gen = MonotonicMemId()` near line 237. Add `from .memid import new_mem_id, MonotonicMemId` at line 28. Replace the three `new_mem_id()` mints (lines ~363, ~457, ~540) with `self._memid_gen.mint()`. In `load()`, after `store._rebuild_id_index()` (line ~822), scan live entries, find the max id (string compare — fixed-width Crockford sorts lexicographically), and call `store._memid_gen.seed_from(max_id)` (Crockford-safe; do NOT use `int(x, 32)`). Change `_backfill_mem_ids` to count and return mints. `list_by_recency` lands in Task 4.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_store_memid_order.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add sqac/store.py tests/test_store_memid_order.py
git commit -m "feat(store): mint monotonic mem_ids seeded from load high-water mark"
```

---

### Task 3: Preserve cartridge header across save

**Files:**
- Modify: `sqac/store.py:762` (store header in load), `sqac/store.py:699-737` (save), `sqac/store.py:191` (init defaults)
- Test: `tests/test_store_header_preserved.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `SqacStore._header_name: str`, `_header_description: str`, `_header_extra: dict` (default `""`, `""`, `{}`).
  - `save()` uses args when given, else the stored header values; passes `extra=self._header_extra`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_store_header_preserved.py
import os, tempfile
from sqac.store import SqacStore
from sqac.format import read_cartridge

def test_save_preserves_name_description_type():
    st = SqacStore(semantic=False)
    st.add("hello world", key="greeting", validate=False)
    d = tempfile.mkdtemp(); p = os.path.join(d, "c.sqac")
    st.save(p, name="My Cart", description="a desc")
    # simulate a roundtrip via load then save with no args
    st2 = SqacStore.load(p)
    p2 = os.path.join(d, "c2.sqac")
    st2.save(p2)  # no name/description -> must reuse header
    h = read_cartridge(p2).header
    assert h.name == "My Cart"
    assert h.description == "a desc"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_store_header_preserved.py -v`
Expected: FAIL (h.name == "")

- [ ] **Step 3: Implement**

In `__init__` (line ~191) add `self._header_name = ""`, `self._header_description = ""`, `self._header_extra: dict = {}`. In `load()` after reading `h` (line ~749), set `store._header_name = h.name`, `store._header_description = h.description`, `store._header_extra = dict(h.extra)`. In `save()`, change the `CartridgeHeader(...)` call to `name=name or self._header_name`, `description=description or self._header_description`, `extra=dict(self._header_extra)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_store_header_preserved.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sqac/store.py tests/test_store_header_preserved.py
git commit -m "fix(store): save() preserves cartridge name/description/type"
```

---

### Task 4: `list_by_recency` + `sqac backfill` + `sqac ids`

**Files:**
- Modify: `sqac/store.py` (new `list_by_recency`), `sqac/cli.py:62-72` (subparsers), `sqac/cli.py` dispatch (new branches near line 800)
- Test: `tests/test_recency_and_backfill_cli.py`

**Interfaces:**
- Consumes: `_backfill_mem_ids() -> int` (Task 2), `decode_mem_id_ms` (Task 1), header preservation (Task 3).
- Produces:
  - `SqacStore.list_by_recency(self, limit: int | None = None) -> list[dict]` → `{"mem_id","created_ms","key_norm","kind"}` for live, non-task entries, mem_id descending.
  - CLI `sqac backfill [--db PATH]` and `sqac ids [--db PATH] [--limit N]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_recency_and_backfill_cli.py
import os, tempfile, time
from sqac.store import SqacStore, KIND_TASK
from sqac.cli import main

def _legacy(path):  # cartridge with NO mem_ids, appended A,B,C
    st = SqacStore(semantic=False)
    for k in ("a", "b", "c"):
        st.add(f"content {k}", key=k, validate=False)
    for e in st._entries: e["meta"].pop("mem_id", None)
    st.save(path)

def test_backfill_recovers_file_order_newest_first():
    d = tempfile.mkdtemp(); p = os.path.join(d, "c.sqac"); _legacy(p)
    rc = main(["backfill", "--db", p]); assert rc == 0
    st = SqacStore.load(p)
    order = [e["key_norm"] for e in st.list_by_recency()]
    assert order == ["c", "b", "a"]  # newest (last appended) first

def test_backfill_is_idempotent_noop():
    d = tempfile.mkdtemp(); p = os.path.join(d, "c.sqac"); _legacy(p)
    main(["backfill", "--db", p]); m1 = os.path.getmtime(p)
    time.sleep(0.01)
    main(["backfill", "--db", p]); m2 = os.path.getmtime(p)
    assert m1 == m2  # second run rewrote nothing

def test_ids_lists_newest_first_and_skips_tasks():
    d = tempfile.mkdtemp(); p = os.path.join(d, "c.sqac")
    st = SqacStore(semantic=False)
    st.add("one", key="k1", validate=False)
    st.add("two", key="k2", validate=False)
    st.add("taskrow", key="kt", kind=KIND_TASK, validate=False)
    st.save(p)
    rc = main(["ids", "--db", p, "--limit", "10"]); assert rc == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_recency_and_backfill_cli.py -v`
Expected: FAIL (list_by_recency missing / CLI rejects 'backfill')

- [ ] **Step 3: Implement**

`list_by_recency`: iterate `self._entries`, skip `deleted` and `kind == KIND_TASK`, collect dicts `{"mem_id", "created_ms": decode_mem_id_ms(mem_id), "key_norm", "kind"}`, sort by `mem_id` reverse, slice `limit`. CLI: add `p_backfill = sub.add_parser("backfill", ...)` with `add_db(p_backfill)`; `p_ids = sub.add_parser("ids", ...)` with `add_db(p_ids)` + `--limit`. Dispatch `backfill`: resolve target(s) — `--db` given → just that file; else `$SQAC_MEM_DIR` (or `~/.sqacm`) else cwd; enumerate `*.sqac` recursively, **skipping** any whose path matches `vault`/`agent`/`secret`/`session`/`archives`/`checkpoint`. For each candidate: open the header cheaply with `read_cartridge(p).header` inside a `try/except` (a v5 encrypted vault has no readable header and must be skipped, never crash the run); skip unless `header.cartridge_type == "memory"`; then `st = SqacStore.load(p)`, `n = st._backfill_mem_ids()`, `if n: st.save(p)`. Print JSON summary `{"cartridges": [...], "minted": total}`. Dispatch `ids`: `st = _open(args.db)`, print JSON of `st.list_by_recency(limit=args.limit)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_recency_and_backfill_cli.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add sqac/store.py sqac/cli.py tests/test_recency_and_backfill_cli.py
git commit -m "feat(cli): sqac backfill + sqac ids + list_by_recency"
```

---

### Task 5 (optional, same PR): `mem_backfill` MCP tool

**Files:**
- Modify: `sqac_mcp.py` (new tool near the other `mem_*` tools, ~line 960)
- Test: `tests/test_mcp_backfill.py`

**Interfaces:**
- Consumes: `SqacStore.load`, `_backfill_mem_ids`, `save`.
- Produces: `mem_backfill() -> str` (JSON `{"cartridges", "minted"}`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mcp_backfill.py
import os, tempfile
def test_mem_backfill_tool(monkeypatch):
    d = tempfile.mkdtemp()
    monkeypatch.setenv("SQAC_MEM_DIR", d); monkeypatch.setenv("SQAC_SEMANTIC", "0")
    import importlib, sqac_mcp; importlib.reload(sqc_mcp)
    # seed a legacy fact cartridge via the rack then call the tool
    assert isinstance(sqc_mcp.mem_backfill(), str)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mcp_backfill.py -v`
Expected: FAIL (no attribute 'mem_backfill')

- [ ] **Step 3: Implement `mem_backfill`**

Thin wrapper over `_state().rack`: for each cartridge, `n = store._backfill_mem_ids()`, save-if-changed; return `_ok({"cartridges": [...], "minted": total})`. Follow the existing `_ok`/`_err` + `@mcp.tool` decorator pattern used by neighbors.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mcp_backfill.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sqac_mcp.py tests/test_mcp_backfill.py
git commit -m "feat(mcp): mem_backfill tool"
```

---

## Final verification

- [ ] `pytest -q` — full suite green (586+ existing + new).
- [ ] `ruff check sqac sqac_mcp.py tests` — clean.
- [ ] Manual: `sqac backfill` on a copy of `~/.sqacm` completes without touching vault/agent/secret cartridges.
- [ ] Manual: `sqac ids --limit 5` prints newest-first with plausible timestamps.