"""sqac.integrations — OpenAI-compatible client middleware.

Wraps any client with the `chat.completions.create` shape (Ollama, vLLM,
LiteLLM, OpenAI) to automatically:

1. **Hydrate** the system prompt with confidence-gated SQAC memories
   relevant to the last user message, and
2. **Record** the user/assistant turns into the SQAC session offloader
   (debounced like the MCP layer), so the next turn can recall them.

    from openai import OpenAI
    from sqac.integrations import wrap

    client = wrap(OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"))
    client.chat.completions.create(model="llama3.1", messages=[...])

Everything else is identical to the unwrapped client — response objects
are passed through untouched.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from .rack import CartridgeRack
from .offloader import ContextOffloader
from .dms import DMS
from . import textnorm

_RACK_ROUTES = {"fact": "facts", "doc": "docs", "skill": "skills"}
_SESSION_BUDGET = 25_000
_FLUSH_TURNS = int(os.environ.get("SQAC_SESSION_FLUSH_TURNS", "10"))
_FLUSH_SECS = float(os.environ.get("SQAC_SESSION_FLUSH_SECS", "60"))


def _memory_dir(memory_dir: Optional[str | Path]) -> Path:
    if memory_dir is not None:
        return Path(memory_dir).expanduser()
    return Path(os.environ.get("SQAC_MEM_DIR", Path.home() / ".sqacm")).expanduser()


def _make_rack(d: Path) -> CartridgeRack:
    return CartridgeRack(
        directory=d, routes=dict(_RACK_ROUTES), default="facts",
        semantic=True, min_confidence=0.60, auto_load=True,
        max_entries=25_000, folder_routing=True,
        normalizer=textnorm.from_env(),
    )


def _format_memories(hits: list) -> str:
    lines = ["Relevant memories (SQAC, confidence-gated):"]
    for h in hits:
        mid = getattr(h, "mem_id", "") or (h.meta.get("mem_id", "") if isinstance(getattr(h, "meta", None), dict) else "")
        tag = f" mem_id={mid}" if mid else ""
        lines.append(f"- [{h.confidence:.2f}]{tag} {h.content}")
    return "\n".join(lines)


def _augment(messages: list[dict], injection: str) -> list[dict]:
    msgs = [dict(m) for m in messages]
    for m in msgs:
        if m.get("role") == "system":
            m["content"] = (m.get("content") or "") + "\n\n" + injection
            return msgs
    return [{"role": "system", "content": injection}] + msgs


class _Completions:
    def __init__(self, client: "SqacClient", upstream, rack, offloader):
        self._client = client
        self._upstream = upstream
        self._rack = rack
        self._offloader = offloader

    def create(self, model: str, messages: list[dict], **kwargs: Any):
        rack = self._rack
        offloader = self._offloader

        query = next((m.get("content") or "" for m in reversed(messages)
                      if m.get("role") == "user"), "")
        hits = rack.search(query, top_k=self._client.top_k) if query else []
        gated = [h for h in hits if h.confidence >= self._client.threshold]

        augmented = messages
        if gated:
            injection = _format_memories(gated)
            augmented = _augment(messages, injection)

        response = self._upstream.chat.completions.create(model=model, messages=augmented, **kwargs)

        assistant_text = ""
        try:
            assistant_text = response.choices[0].message.content or ""
        except Exception:
            assistant_text = ""
        if query:
            offloader.observe("user", str(query))
        if assistant_text:
            offloader.observe("assistant", assistant_text)
        if offloader.should_flush(_FLUSH_TURNS, _FLUSH_SECS):
            offloader.save()
        return response


class _Chat:
    def __init__(self, client: "SqacClient", upstream, rack, offloader):
        self.completions = _Completions(client, upstream, rack, offloader)


class SqacClient:
    """OpenAI-compatible client augmented with SQAC memory."""

    def __init__(self, upstream, memory_dir: Optional[str | Path] = None,
                 threshold: float = 0.60, top_k: int = 5):
        d = _memory_dir(memory_dir)
        d.mkdir(parents=True, exist_ok=True)
        self._upstream = upstream
        self.threshold = threshold
        self.top_k = top_k
        self.memory_dir = d
        self.rack = _make_rack(d)
        self.offloader = ContextOffloader(
            d / "session.sqac", window=8, semantic=True,
            dms=DMS(budget=_SESSION_BUDGET),
        )
        self.chat = _Chat(self, self._upstream, self.rack, self.offloader)

    def __getattr__(self, name: str) -> Any:
        # fallthrough to the wrapped client for anything we don't model
        return getattr(self._upstream, name)


def wrap(client, memory_dir: Optional[str | Path] = None,
         threshold: float = 0.60, top_k: int = 5) -> SqacClient:
    """Wrap an OpenAI-compatible client with SQAC memory augmentation."""
    return SqacClient(client, memory_dir=memory_dir, threshold=threshold, top_k=top_k)
