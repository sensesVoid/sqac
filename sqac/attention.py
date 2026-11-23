"""SQAC attention block: compact Current/Pending task state for injection.

v1.4.0. The block is re-injected into LLM context every N turns
(SQAC_ATTENTION_TURNS, default 5) and immediately after task changes.
Hard cap: <= 150 tokens so it stays cheap.
"""

from __future__ import annotations

from typing import Sequence

CURRENT_CAP = 5
PENDING_CAP = 10
DEFAULT_PENDING = 5  # fallback pending turns (recent window)
MAX_TOKENS = 150
MAX_LINE_CHARS = 100  # per-item char budget (token budget enforcement)


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
            if len(content) > MAX_LINE_CHARS:
                content = content[: MAX_LINE_CHARS - 1].rstrip() + "…"
            lines.append(f"- {content} (mem_id {mem_id})")
        if len(items) > cap:
            lines.append(f"- +{len(items) - cap} more")

    _sec("Current", current, CURRENT_CAP)
    _sec("Pending", pending, PENDING_CAP)
    return "\n".join(lines).rstrip() + "\n"


def _goal_text(continuity, project: str) -> str:
    """The active goal from continuity (best-effort, empty when none)."""
    try:
        return continuity.record(project).get("goal") or ""
    except Exception:
        return ""


def render_block_with_fallback(turn: int,
                               current: Sequence[tuple[str, str]],
                               pending: Sequence[tuple[str, str]],
                               offloader=None,
                               continuity=None,
                               project: str | None = None) -> str:
    """Render the attention block, falling back to derived task state.

    When no explicit tasks exist (current and pending both empty):
      - Current = the active goal from continuity (v1.4.0; the MCP loader
        calls ContinuityStore.load so goals are visible in-process)
      - Pending = recent session exchanges (offloader.recent())
    """
    if current or pending:
        return render_block(turn, current, pending)
    cur: list[tuple[str, str]] = []
    if continuity is not None and project:
        goal = _goal_text(continuity, project)
        if goal:
            cur.append((goal[:200], ""))
    pnd: list[tuple[str, str]] = []
    if offloader is not None:
        try:
            for text in offloader.recent(DEFAULT_PENDING):
                pnd.append((text[:200], ""))
        except Exception:
            pass
    return render_block(turn, cur, pnd)
