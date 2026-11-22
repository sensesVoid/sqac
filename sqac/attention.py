"""SQAC attention block: compact Current/Pending task state for injection.

v1.4.0. The block is re-injected into LLM context every N turns
(SQAC_ATTENTION_TURNS, default 5) and immediately after task changes.
Hard cap: <= 150 tokens so it stays cheap.
"""

from __future__ import annotations

from typing import Sequence

CURRENT_CAP = 5
PENDING_CAP = 10
MAX_TOKENS = 150


def count_tokens(text: str) -> int:
    """Cheap token proxy: whitespace-delimited words (approx 0.75 tokens/word)."""
    return len(text.split()) * 4 // 3 or len(text.split())


def render_block(turn: int,
                 current: Sequence[tuple[str, str]],
                 pending: Sequence[tuple[str, str]]) -> str:
    """Render the compact attention block.

    current/pending: sequences of (content, mem_id) tuples.
    """
    lines = [f"[SQAC attention | turn {turn}]"]

    def _sec(name: str, items: Sequence[tuple[str, str]], cap: int) -> None:
        if not items:
            return
        lines.append(f"{name}:")
        shown = list(items[:cap])
        for content, mem_id in shown:
            content = (content or "").strip().replace("\n", " ")
            lines.append(f"- {content} (mem_id {mem_id})")
        if len(items) > cap:
            lines.append(f"- +{len(items) - cap} more")

    _sec("Current", current, CURRENT_CAP)
    _sec("Pending", pending, PENDING_CAP)
    return "\n".join(lines).rstrip() + "\n"
