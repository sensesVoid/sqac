# Attention Blocks

Re-injectable compact memory: the `[SQAC attention | turn N]` block.

## Where the block is injected

| Surface | Mechanism | Notes |
|---|---|---|
| Integrations middleware | `SqacClient.chat.completions.create` → injects into system prompt | plus own turn counter |
| MCP | `mem_bootstrap`, `memory://context` resource | always current |
| MCP | `mem_observe` response | when `attention_due(ATTENTION_TURNS)` |
| HTTP | `GET /attention` | always returns the current block |
| HTTP | `/session/observe` response | when due |

## Cadence

- `SQAC_ATTENTION_TURNS` env (default `5`) — refresh every N turns
- Immediate refresh on task-state change (`mark_attention_dirty()`)

## Sources (render_block / render_block_with_fallback)

- **Explicit tasks first**: `Current:` from `TaskManager.list(status="current")`,
  `Pending:` from `list(status="pending")` — mem_ids for exact lookup
- **Derived fallback** (when no explicit tasks):
  - Current = continuity goal for the project (`continuity.record(project).get("goal")`)
  - Pending = recent session exchanges (`offloader.recent(5)`)

## Caps

- `CURRENT_CAP = 5`, `PENDING_CAP = 10`, overflow `+N more` lines
- `count_tokens` proxy: whitespace words × 4/3
- Hard budget: `MAX_TOKENS = 150` (enforced in tests)

## TaskManager ↔ block

`sqac/tasks.py` (kind=task entries, per-project): `task_add` / `task_done` / `task_list`.
Finished tasks graduate to durable `kind=fact` memories preserving mem_id.
