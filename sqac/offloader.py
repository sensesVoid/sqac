#!/usr/bin/env python3
"""ContextOffloader — rolling-window session memory backed by a .sqac bucket.

The economics (measured in this repo): reasoning tokens are the most
expensive tokens an LLM spends, and a recall hit costs ~0.1ms encode plus
~100 tokens of injected text. When a conversation outgrows the window, the
choice is re-derivation (thousands of reasoning tokens) or recall (one tool
call). This module is the recall side.

Design rules — all measured via prototypes in this repo's history:

  1. Offload EXCHANGES, not turns. Verbatim turn indexing fails at
     back-references: "what was that bug?" matches the user's question turn
     and shadows the answer. Index (question -> answer) as one logical
     record; the user's phrasing is a perfect question-shaped trigger and
     it is free.

  2. Denyxis keys at write time. Temporal pointers ("that", "earlier",
     "you mentioned") appear in every back-reference and create ties.
     Keys carry entity anchors instead; deixis is RESOLVED at offload time
     (rewritten against recent context), never stored. Prototype result:
     verbatim turns 2/8 recall -> exchange indexing 6/8 -> denyxised keys 8/8.

  3. Fail safe. recall() returns "" when nothing clears min_confidence:
     the model is told "no memory of that" rather than being injected with
     a plausible-but-wrong exchange. The offloader must never confidently
     inject the wrong context.

  4. The bucket is ephemeral by contract. Entries are kind=turn in a
     separate session.sqac; durable knowledge graduates into team packs
     (kind=fact). Hot-swap separates them: drop the session bucket at
     session end, keep what graduated.

Usage:
    from sqac.offloader import ContextOffloader

    off = ContextOffloader("session.sqac", window=8)
    off.observe("user", "Our CI fails with a 504 on deploy")
    off.observe("assistant", "The 504 is the docker build timing out ...")
    ...
    text = off.recall("what was that 504 about?")   # -> text or ""
    off.save()                                       # survives restart

    # bulk: turn a transcript file into a bucket
    python -m sqac.offloader transcript.jsonl -o session.sqac
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .dms import DMS
from .store import KIND_TURN, SqacStore

# ── text utilities ──────────────────────────────────────────────────────────

_WS = re.compile(r"\s+")

# Function words excluded from anchor extraction (kept small on purpose).
_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "can", "may", "might", "to", "of", "in", "for", "on", "with",
    "at", "by", "from", "as", "into", "about", "and", "or", "but", "not",
    "if", "then", "else", "when", "what", "which", "who", "how", "why",
    "that", "this", "these", "those", "it", "its", "we", "our", "you",
    "your", "i", "me", "my", "there", "here", "ok", "okay", "please",
    "just", "also", "so", "up", "out", "again", "still", "lets", "let",
}

# Deixis / anaphora tokens: never allowed into KEYS. They are resolved
# against recent context at offload time (rule 2).
_DEIXIS = {
    "that", "this", "it", "its", "they", "them", "their", "those", "these",
    "earlier", "before", "previous", "previously", "above", "mentioned",
    "referenced", "same", "said", "aforementioned", "last", "prior",
}

# Salience markers: exchanges carrying these rank higher for forced flush.
_SALIENT = re.compile(
    r"\b(\d+(?:\.\d+)?\s*(?:ms|s|kb|mb|gb|%|percent)|error|fail(?:ed|ure|ing)?|"
    r"bug|crash|timeout|root\s*cause|decided|decision|agreed|deferred|todo|"
    r"blocked|deadline|never|always|must)\b",
    re.IGNORECASE,
)

_QUESTION = re.compile(r"\?")


def _content_words(text: str) -> list[str]:
    """Content tokens for ANCHOR extraction: stop-words and deixis excluded.

    Deixis must never enter anchors — anchors become keys, and keys carry
    entity anchors only (rule 2)."""
    words = [w.strip(".,;:!?()[]{}'\"").lower() for w in text.split()]
    seen: set[str] = set()
    out = []
    for w in words:
        if len(w) > 2 and w not in _STOP and w not in _DEIXIS and w not in seen:
            seen.add(w)
            out.append(w)
    return out


def _denyxis(text: str, substitutes: list[str]) -> str:
    """Strip deixis tokens; replace multi-word deixis phrases with anchors.

    Returns the cleaned text plus extracted anchors appended when the
    remainder is too thin to be a usable key.
    """
    words = [w.strip(".,;:!?()[]{}'\"").lower() for w in text.split()]
    kept = [w for w in words if w not in _DEIXIS]
    cleaned = _WS.sub(" ", " ".join(kept)).strip()
    if len([w for w in kept if w not in _STOP]) >= 2:
        return cleaned
    # too thin: append entity anchors from recent context
    extra = " ".join(substitutes[:3])
    return f"{cleaned} {extra}".strip()


# ── data model ──────────────────────────────────────────────────────────────

@dataclass
class Turn:
    role: str  # "user" | "assistant" | "system" | "tool"
    text: str


@dataclass
class Exchange:
    """One (question -> answer) logical record."""

    xid: int
    turns: list[Turn]
    turn_range: tuple[int, int]  # first..last turn index in the session
    salience: float = 0.0
    keys: list[str] = field(default_factory=list)      # filled by distiller
    content: str = ""                                   # filled by distiller


# A distiller maps an Exchange to (keys, content). The heuristic one ships
# here; an LLM-backed one is a drop-in (see README for the prompt contract).
Distiller = Callable[[Exchange], tuple[list[str], str]]


def heuristic_distill(ex: Exchange) -> tuple[list[str], str]:
    """No-LLM distillation: question-shape + entity anchors, zero deps.

    keys: (1) the user's phrasing, deixis-stripped — lands exact at conf 1.0
    when the user re-asks verbatim; (2) a mixed anchor phrase (question +
    answer content words) — covers short/thin questions; (3) the answer's
    leading content chunk — covers paraphrased back-references that share
    vocabulary with the ANSWER, not the question (measured: without key 3,
    "cause of that 504 timeout" missed an exchange whose question said
    "failing with 504" but whose answer said "timing out").

    content: "[turns a-b] Q: ... A: ..." — question text is retrieval-
    relevant vocabulary AND gives the model the resolution context.

    Measured on the 8-back-reference prototype conversation: 8/8 top-1.
    """
    user_turns = [t for t in ex.turns if t.role == "user"]
    asst_turns = [t for t in ex.turns if t.role != "user"]
    q = " ".join(t.text for t in user_turns).strip() or "context"
    a = " ".join(t.text for t in asst_turns).strip()

    qw = _content_words(q)
    aw = _content_words(a)
    anchors = qw[:4] + aw[:10]
    seen: set[str] = set()
    anchors = [w for w in anchors if not (w in seen or seen.add(w))][:12]

    keys = [_denyxis(q, anchors), " ".join(anchors[:8])]
    if len(aw) >= 3:
        keys.append(" ".join(aw[:6]))

    # dedupe, preserve order, drop empties
    keys = [k for k in keys if k.strip() and not (k in seen or seen.add(k))]

    content = f"[turns {ex.turn_range[0]}-{ex.turn_range[1]}] Q: {_denyxis(q, anchors)} A: {a}"
    return keys, content


# ── the offloader ───────────────────────────────────────────────────────────

class ContextOffloader:
    """Rolling-window conversation memory in a .sqac session bucket.

    observe() feeds turns; completed exchanges beyond `window` are distilled
    into kind=turn entries and evicted from the buffer. recall() answers
    back-references with plain text + confidence, or "" on no confident hit.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        window: int = 8,
        min_confidence: float = 0.60,
        distiller: Optional[Distiller] = None,
        semantic: bool = True,
        dms: Optional[DMS] = None,
    ):
        self.path = Path(path) if path else None
        self.window = max(2, window)
        self.min_confidence = min_confidence
        self.distiller = distiller or heuristic_distill
        self.semantic = semantic
        self.dms = dms  # optional Dynamic Memory Sparsification policy
        self._turn_count = 0
        self._xid = 0
        self._buffer: list[Turn] = []       # live turns inside the window
        self._evicted: list[int] = []       # xids flushed to the store
        if self.path and self.path.exists():
            self._store = SqacStore.load(self.path, fuzzy_threshold=min_confidence)
            # resume the exchange counter from the stored meta, NOT the entry
            # count — each exchange writes multiple keys (one entry per key)
            self._xid = max(
                (e.get("meta", {}).get("exchange", 0) for e in self._store._entries),
                default=0,
            )
            self._turn_count = self._restore_turn_count()
        else:
            self._store = SqacStore(semantic=semantic, fuzzy_threshold=min_confidence)

    def _restore_turn_count(self) -> int:
        """Restore the exact turn counter written by save() (a stored
        `turns=` header token). Buckets saved before the counter was kept
        fall back to the 2-per-exchange approximation."""
        desc = getattr(self._store, "_description", "") or ""
        m = re.search(r"\bturns=(\d+)\b", desc)
        if m:
            return int(m.group(1))
        return 2 * self._xid

    # ── write path ──────────────────────────────────────────────────────

    def observe(self, role: str, text: str) -> Optional[int]:
        """Feed one turn. Returns an evicted exchange id when the window
        overflows and an exchange is offloaded, else None."""
        text = (text or "").strip()
        if not text:
            return None
        self._turn_count += 1
        self._buffer.append(Turn(role=role, text=text))
        if len(self._buffer) >= self.window:
            return self._flush_one()
        return None

    def _flush_one(self) -> int:
        """Distill + store the oldest complete exchange in the buffer."""
        ex = self._pop_exchange()
        self._write_exchange(ex)
        return ex.xid

    def _pop_exchange(self) -> Exchange:
        """Pop one exchange: user turn(s) up to and including the next
        non-user reply. Falls back to a single turn when roles are uneven."""
        if not self._buffer:
            raise RuntimeError("buffer empty")
        taken: list[Turn] = []
        start = self._turn_count - len(self._buffer)
        # take leading user turns then the first non-user reply
        while self._buffer and self._buffer[0].role == "user":
            taken.append(self._buffer.pop(0))
        if self._buffer:
            taken.append(self._buffer.pop(0))  # the reply
        while not taken:
            taken.append(self._buffer.pop(0))
        self._xid += 1
        ex = Exchange(
            xid=self._xid,
            turns=taken,
            turn_range=(start, start + len(taken) - 1),
        )
        ex.salience = _salience(ex)
        return ex

    def _write_exchange(self, ex: Exchange) -> None:
        keys, content = self.distiller(ex)
        if not keys or not content.strip():
            return
        ex.keys, ex.content = keys, content
        meta = {
            "exchange": ex.xid,
            "turns": list(ex.turn_range),
            "salience": round(ex.salience, 3),
        }
        if self.dms is not None:
            self.dms.register(ex.xid, ex.salience, tier="bucket")
        for k in keys:
            k = k.strip()
            if not k:
                continue  # zero-anchor keys match everything at the floor
            self._store.add(
                content,
                key=k,
                meta=meta,
                source=f"turn:{ex.xid}",
                kind=KIND_TURN,
            )
        self._evicted.append(ex.xid)

    def offload(self) -> int:
        """Force-flush the whole buffer (end of session, or budget pressure).
        Returns the number of exchanges offloaded."""
        n = 0
        while self._buffer:
            self._flush_one()
            n += 1
        return n

    def sparsify(self, top_k: int | None = None) -> int:
        """Demote low-utility bucket entries to the durable fact tier.

        Runs the DMS eviction list against the stored turn entries. Each
        demoted exchange is rewritten as a kind=fact entry (exempt from the
        hot basket) and its DMS record is marked durable. Returns the number
        of exchanges demoted. No-op when no DMS is configured.
        """
        if self.dms is None:
            return 0
        xids = self.dms.evict_list(top_k=top_k)
        if not xids:
            return 0
        # re-key the surviving entries by exchange id for lookup
        by_exchange: dict[int, int] = {}
        for idx, e in enumerate(self._store._entries):
            if e.get("deleted"):
                continue
            by_exchange[e["meta"].get("exchange", idx)] = idx
        demoted = 0
        for xid in xids:
            idx = by_exchange.get(xid)
            if idx is None:
                continue
            e = self._store._entries[idx]
            # rewrite in place: turn -> durable fact (exempt from the basket)
            self._store.delete(idx)
            self._store.add(
                e["content"],
                key=e.get("key_norm") or e["content"],
                meta={**e.get("meta", {}), "tier": "durable"},
                source=e.get("source", "user"),
                kind="fact",
            )
            self.dms.promote(xid)  # mark the record durable, not evictable
            demoted += 1
        self._store.compact()
        return demoted

    # ── read path ───────────────────────────────────────────────────────

    def recall(self, query: str, top_k: int = 2) -> str:
        """Answer a back-reference with offloaded context, or "" (fail-safe).

        Returns a text block ready for prompt injection:

            [0.78] [turns 1-2] Q: ci 504 deploy A: The 504 is ...

        Empty string means "nothing confident" — the caller should let the
        model say so rather than guess.
        """
        hits = self._store.search(query, top_k=top_k * 3, kind="turn")
        lines = []
        seen: set[Any] = set()
        for h in hits:
            if h.confidence < self.min_confidence:
                continue
            xid = h.meta.get("exchange")
            if xid in seen:  # one exchange has many keys: inject it once
                continue
            seen.add(xid)
            if self.dms is not None:
                self.dms.touch(xid)
            lines.append(f"[{h.confidence:.2f}] {h.content}")
            if len(lines) >= top_k:
                break
        return "\n".join(lines)

    def recall_detailed(self, query: str, top_k: int = 2) -> list[dict[str, Any]]:
        """Structured recall for tool/JSON surfaces (MCP, function calling).

        Exchanges are deduplicated (many keys -> one logical record) and
        below-threshold hits are dropped (fail-safe).
        """
        out: list[dict[str, Any]] = []
        seen: set[Any] = set()
        for h in self._store.search(query, top_k=top_k * 3, kind="turn"):
            if h.confidence < self.min_confidence:
                continue
            xid = h.meta.get("exchange")
            if xid in seen:
                continue
            seen.add(xid)
            if self.dms is not None:
                self.dms.touch(xid)
            out.append(
                {
                    "content": h.content,
                    "confidence": round(h.confidence, 4),
                    "exchange": h.meta.get("exchange"),
                    "turns": h.meta.get("turns"),
                    "salience": h.meta.get("salience"),
                    "mode": h.mode,
                }
            )
            if len(out) >= top_k:
                break
        return out

    # ── persistence ─────────────────────────────────────────────────────

    def save(self, path: str | Path | None = None) -> Path:
        """Persist the bucket. In-memory buffer is flushed first: nothing
        stays volatile across save(). An explicit path is adopted as the
        configured path, so later save() calls reuse it."""
        self.offload()
        p = Path(path) if path else self.path
        if p is None:
            raise ValueError("no path given and none configured")
        if self.path is None:
            self.path = p
        self._store.save(
            p,
            name="session-bucket",
            description=(
                f"{self._store.stats()['entries']} turn entries, "
                f"{self._xid} exchanges, xid={self._xid} "
                f"turns={self._turn_count} window={self.window}"
            ),
        )
        return p

    def stats(self) -> dict[str, Any]:
        s = self._store.stats()
        s.update(
            {
                "exchanges_offloaded": self._xid,
                "buffer_turns": len(self._buffer),
                "window": self.window,
                "min_confidence": self.min_confidence,
            }
        )
        return s

    # ── bulk ingest ─────────────────────────────────────────────────────

    @classmethod
    def from_transcript(
        cls,
        transcript: str | Path,
        path: str | Path,
        window: int = 8,
        min_confidence: float = 0.60,
        distiller: Optional[Distiller] = None,
        semantic: bool = True,
    ) -> "ContextOffloader":
        """Build a session bucket from a JSONL transcript.

        Accepted per-line fields: role/content (also role/text/message).
        Lines missing either field are skipped. The whole transcript is
        replayed through observe(), then force-flushed and saved — so the
        resulting bucket indexes every exchange, not just the tail.
        """
        off = cls(
            path=None,  # build fresh, save once at the end
            window=window,
            min_confidence=min_confidence,
            distiller=distiller,
            semantic=semantic,
        )
        for ln in Path(transcript).read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
            except json.JSONDecodeError:
                continue
            role = str(obj.get("role", "user"))
            text = obj.get("content") or obj.get("text") or obj.get("message") or ""
            off.observe(role, str(text))
        off.path = Path(path)
        off.save(path)
        return off


def _salience(ex: Exchange) -> float:
    """Cheap importance estimate: salient markers + question count + length."""
    text = " ".join(t.text for t in ex.turns)
    score = 0.3 * len(_SALIENT.findall(text))
    score += 0.2 * len(_QUESTION.findall(text))
    score += min(0.4, 0.01 * len(text.split()))
    return round(min(score, 1.0), 3)


def main(argv: list[str] | None = None) -> int:
    """CLI: build/recall session buckets.

        python -m sqac.offloader transcript.jsonl -o session.sqac
        python -m sqac.offloader --recall "that 504 bug" --db session.sqac
        python -m sqac.offloader --stats --db session.sqac
    """
    import argparse

    ap = argparse.ArgumentParser(prog="sqac.offloader", description="session context buckets")
    ap.add_argument("transcript", nargs="?", help="JSONL transcript (role/content per line)")
    ap.add_argument("-o", "--out", default="session.sqac")
    ap.add_argument("--db", default="session.sqac", help="bucket for --recall/--stats")
    ap.add_argument("--recall", default=None, metavar="QUERY")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--min-conf", type=float, default=0.60)
    args = ap.parse_args(argv)

    if args.recall is not None:
        off = ContextOffloader(args.db, min_confidence=args.min_conf)
        text = off.recall(args.recall, top_k=args.k)
        if not text:
            print("no confident memory of that")
            return 1
        print(text)
        return 0

    if args.stats:
        off = ContextOffloader(args.db)
        for k, v in off.stats().items():
            print(f"{k}: {v}")
        return 0

    if args.transcript:
        if not Path(args.transcript).exists():
            print(f"transcript not found: {args.transcript}", file=sys.stderr)
            return 1
        off = ContextOffloader.from_transcript(
            args.transcript, args.out, window=args.window, min_confidence=args.min_conf
        )
        s = off.stats()
        print(f"offloaded {s['exchanges_offloaded']} exchanges -> {args.out}")
        print(f"bucket: {s['entries']} entries, kind breakdown: {s['kinds']}")
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
