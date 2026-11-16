# Agent cartridges — persona + working state

Agent cartridges replace per-agent persona and working-state markdown files
(`AGENTS.md` / `CLAUDE.md` style notes) with structured, queryable entries.

## Quick start

```bash
sqac agent init atlas --role "coding assistant"
sqac agent use atlas                       # set active agent
sqac agent set identity "I am Atlas, a pragmatic coding agent."
sqac agent set instruction "Prefer tests before implementation."
sqac agent set preference "Answer concisely."
sqac agent set goal "Ship SQAC v1 docs" --key docs-v1 --priority 2
sqac agent decision "Use ruff with line length 100"
sqac agent note "CI matrix reduced to Linux x86_64 + Windows AMD64"
sqac agent bootstrap                       # print the session-start packet
sqac agent done docs-v1                    # mark goal/decision done
sqac agent search "docs"                   # search this agent's memory
sqac agent list
```

## Sections

| Section | Behavior |
|---------|----------|
| `identity` | Singleton — `set` replaces the existing entry |
| `instruction` | Singleton |
| `preference` | Singleton |
| `goal` | List — `set` appends (or upserts when `--key` is given) with a generated stable key (`goal-<8 hex>`) |
| `decision` | List — same key behavior (`decision-<8 hex>`) |
| `note` | List — same key behavior (`note-<8 hex>`) |

`set --key KEY` upserts by normalized key, replacing that entry's
content/metadata. `--priority` (0–3) controls bootstrap ordering for goals.
`sqac agent done KEY_OR_QUERY` marks an entry done (exact `meta.key` first,
then a unique fuzzy match; ambiguous queries fail closed).

## Bootstrap packet

`bootstrap(max_chars=4000)` composes the "get to work" packet for a fresh
session, in order:

1. identity + instructions + preferences (all entries)
2. open goals, by priority then recency
3. last 5 decisions
4. notes, only if character budget remains

Closed goals and superseded singletons never duplicate — upsert replaces them.

## Storage and isolation

- Agents live at `~/.sqacm/agents/<name>.sqac`; the active agent pointer is
  `~/.sqacm/agent_active`.
- Rack auto-load skips `agent` and `vault` cartridges **by header type** (not by
  name), so a normal memory cartridge named `agent` is unaffected and agent
  entries never leak into default `mem_search` results.
- Agent state is plaintext in v1 (docs flag this); encryption can reuse vault
  crypto later if requested.

## MCP tools

- `agent_list()` — available agents + active name
- `agent_use(name)` — set active agent
- `agent_show(name=None)` — bootstrap packet (default: active agent)
- `agent_bootstrap(name=None, max_chars=4000)` — bootstrap packet
- `agent_update(section, content, key=None, priority=1)` — set a section
- `agent_done(key_or_query)` — mark done
- `agent_search(query, name=None)` — search agent memory

`mem_bootstrap` prepends the active agent's packet when one is set; set
`SQAC_AGENT=0` to disable that.
