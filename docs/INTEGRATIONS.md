# OpenAI-Compatible Client Middleware (`sqac.integrations`)

Wrap any client with the `chat.completions.create` shape — **Ollama,
vLLM, LiteLLM, OpenAI** — and get SQAC memory automatically:

1. **Hydrate** — before the call, SQAC searches your rack for memories
   relevant to the last user message, keeps those with confidence
   ≥ `threshold`, and injects them into (or prepends) the system prompt.
2. **Record** — after the call, the user turn and the assistant reply are
   observed into the session offloader (debounced like the MCP layer:
   save at most every 10 turns / 60 s, or on your explicit flush).

```python
from openai import OpenAI
from sqac.integrations import wrap

client = wrap(
    OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"),
    memory_dir="~/.sqacm",   # default: SQAC_MEM_DIR or ~/.sqacm
    threshold=0.60,          # confidence gate
    top_k=5,
)

resp = client.chat.completions.create(
    model="llama3.1",
    messages=[{"role": "user", "content": "can we ship x86 images?"}],
)
# $ model answers using your memory; the exchange is recorded
```

## Behavior notes

- **Passthrough** — response objects are returned untouched; any
  attribute/method not modeled by the wrapper falls through to the
  wrapped client (`wrapped.models`, etc.).
- **Injection** — if the messages list already has a system message it is
  extended in place; otherwise a system message is prepended.
- **Gate** — hits below `threshold` are not injected (default 0.60).
- **Recording** — observes `role=user` (last user message) and
  `role=assistant` (response content). Uses `ContextOffloader`; a crash
  loses at most the unsaved tail (≤ 10 turns), same semantics as
  `mem_observe`.
- **Memory dir** — resolved as: explicit `memory_dir=` arg →
  `SQAC_MEM_DIR` → `~/.sqacm`.
- **Normalization** — `SQAC_NORMALIZE` is honored for the rack, like the
  MCP and HTTP racks.
