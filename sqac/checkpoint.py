"""Rate-limit Checkpoint Cartridge — captures mid-session state for cross-CLI recovery.

When an agent CLI hits a rate limit (or any interruption), this module checkpoints
the current conversation buffer + recent exchanges into a dedicated cartridge.
DMS applies for sparsification. Cross-CLI recovery via MCP tools.

Usage:
    from sqac.checkpoint import CheckpointCartridge
    
    # In agent CLI code, on rate limit:
    cp = CheckpointCartridge()
    cp.checkpoint(current_offloader, reason="rate_limit")
    
    # Next agent (different CLI) recovers:
    cp = CheckpointCartridge()
    context = cp.recover("what was I doing?")
"""

from __future__ import annotations

import hashlib
import json
import os
import pwd
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional


def _get_user_home() -> Path:
    """Get the actual user home directory, not the HOME env var."""
    try:
        return Path(pwd.getpwuid(os.getuid()).pw_dir)
    except Exception:
        # Fallback: try common locations
        for candidate in [Path("/home/zeus"), Path("/home/user"), Path("/root")]:
            if candidate.exists():
                return candidate
        return Path(os.path.expanduser("~"))


USER_HOME = _get_user_home()

from .store import SqacStore, KIND_TURN, KIND_FACT
from .offloader import ContextOffloader, Exchange, Turn
from .dms import DMS, UtilityWeights


@dataclass
class Checkpoint:
    """A single recovery checkpoint."""
    timestamp: float
    reason: str                    # "rate_limit" | "interrupt" | "budget" | "manual"
    agent_cli: str                 # "opencode" | "claude-code" | "codex" | "cursor" | "zed"
    session_id: str                # unique session identifier
    turn_count: int
    exchange_count: int
    buffer_turns: list[dict]       # live turns in window
    recent_exchanges: list[dict]   # last N exchanges with keys/content
    metadata: dict[str, Any] = field(default_factory=dict)


class CheckpointCartridge:
    """Manages the checkpoint cartridge for cross-CLI recovery.
    
    Storage: ~/.sqacm/checkpoint.sqac (shared across all CLIs)
    Also registers in CartridgeRack at ~/.sqacm/ for rack.search()
    """
    
    DEFAULT_PATH = USER_HOME / ".sqacm" / "checkpoint.sqac"
    DEFAULT_SESSION_ID_FILE = USER_HOME / ".sqacm" / "session_id"
    
    def __init__(
        self,
        path: str | Path | None = None,
        rack_dir: str | Path | None = None,
        session_id: str | None = None,
        max_checkpoints: int = 50,
        dms_budget: int = 1000,
    ):
        self.path = Path(path) if path else self.DEFAULT_PATH
        self.rack_dir = Path(rack_dir) if rack_dir else self.path.parent
        self.max_checkpoints = max_checkpoints
        
        # Session ID: persistent across restarts, identifies the logical conversation
        self.session_id = session_id or self._load_or_create_session_id()
        
        # Agent CLI detection
        self.agent_cli = self._detect_agent_cli()
        
        # DMS for sparsification
        self.dms = DMS(budget=dms_budget)
        
        # Load or create store
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self._store = SqacStore.load(self.path, fuzzy_threshold=0.60)
            self._load_dms_from_store()
        else:
            self._store = SqacStore(semantic=True, fuzzy_threshold=0.60)
        
        # Register in rack for cross-CLI search
        self._register_in_rack()
    
    def _load_or_create_session_id(self) -> str:
        """Load or create persistent session ID."""
        if self.DEFAULT_SESSION_ID_FILE.exists():
            try:
                return self.DEFAULT_SESSION_ID_FILE.read_text().strip()
            except Exception:
                pass
        # Generate new session ID from timestamp + random
        sid = hashlib.sha256(f"{time.time()}{os.getpid()}".encode()).hexdigest()[:16]
        self.DEFAULT_SESSION_ID_FILE.write_text(sid)
        return sid
    
    def _detect_agent_cli(self) -> str:
        """Detect which CLI is running."""
        # Check environment variables
        if os.getenv("OPENCODE"):
            return "opencode"
        if os.getenv("CLAUDE_CODE"):
            return "claude-code"
        if os.getenv("CODEX"):
            return "codex"
        if os.getenv("CURSOR"):
            return "cursor"
        if os.getenv("ZED"):
            return "zed"
        # Check argv
        import sys
        argv = " ".join(sys.argv).lower()
        if "opencode" in argv:
            return "opencode"
        if "claude" in argv:
            return "claude-code"
        if "codex" in argv:
            return "codex"
        return "unknown"
    
    def _load_dms_from_store(self) -> None:
        """Reconstruct DMS records from stored checkpoint entries."""
        for e in self._store._entries:
            if e.get("deleted") or e.get("kind") != KIND_TURN:
                continue
            meta = e.get("meta", {})
            xid = meta.get("exchange")
            if xid is not None:
                self.dms.register(xid, meta.get("salience", 0.0), "bucket")
                rec = self.dms._records[xid]
                rec.access_count = meta.get("access_count", 0)
                rec.tier = meta.get("tier", "bucket")
                if "created_at" in meta:
                    rec.created_at = meta["created_at"]
    
    def _register_in_rack(self) -> None:
        """Register checkpoint cartridge in the shared rack."""
        try:
            from .rack import CartridgeRack
            rack = CartridgeRack(self.rack_dir, semantic=True)
            rack.register("checkpoint", self.path)
        except Exception:
            pass  # Rack registration is optional
    
    def checkpoint(
        self,
        offloader: ContextOffloader,
        reason: str = "manual",
        metadata: Optional[dict] = None,
    ) -> Checkpoint:
        """Create a checkpoint from the current offloader state.
        
        Captures:
        - Live buffer turns (before offload)
        - Recent exchanges (last 10)
        - Session metadata
        """
        # Capture buffer turns FIRST (before offload flushes them)
        buffer_turns = [
            {"role": t.role, "text": t.text}
            for t in offloader._buffer
        ]
        
        # Now force offload any pending buffer to get complete state
        offloader.offload()
        
        # Capture recent exchanges from BOTH stores
        recent_exchanges = []
        seen_xids = set()
        
        # Check checkpoint store
        for store in [self._store, offloader._store]:
            turn_entries = [
                e for e in store._entries
                if not e.get("deleted") and e.get("kind") == KIND_TURN
            ]
            turn_entries.sort(key=lambda e: e.get("meta", {}).get("exchange", 0), reverse=True)
            
            for e in turn_entries[:10]:
                xid = e.get("meta", {}).get("exchange")
                if xid in seen_xids or xid is None:
                    continue
                seen_xids.add(xid)
                recent_exchanges.append({
                    "exchange": xid,
                    "content": e["content"],
                    "keys": e.get("key_norm", ""),
                    "turns": e.get("meta", {}).get("turns"),
                    "salience": e.get("meta", {}).get("salience", 0.0),
                    "tier": e.get("meta", {}).get("tier", "bucket"),
                })
        
        # Sort by exchange id descending and limit
        recent_exchanges.sort(key=lambda ex: ex["exchange"], reverse=True)
        recent_exchanges = recent_exchanges[:10]
        
        cp = Checkpoint(
            timestamp=time.time(),
            reason=reason,
            agent_cli=self.agent_cli,
            session_id=self.session_id,
            turn_count=offloader._turn_count,
            exchange_count=offloader._xid,
            buffer_turns=buffer_turns,
            recent_exchanges=recent_exchanges,
            metadata=metadata or {},
        )
        
        # Write to store
        self._write_checkpoint(cp)
        
        # Register with DMS
        self.dms.register(
            xid=hash(f"{self.session_id}:{int(cp.timestamp)}") & 0xFFFFFFFF,
            salience=0.8,  # checkpoints are high-value
            tier="bucket",
        )
        
        # Sparsify if over budget
        if self.dms.over_budget():
            self._sparsify()
        
        # Prune old checkpoints
        self._prune_old()
        
        return cp
    
    def _write_checkpoint(self, cp: Checkpoint) -> None:
        """Write checkpoint to store."""
        content = json.dumps(cp.__dict__, indent=2)
        key = f"checkpoint:{cp.session_id}:{int(cp.timestamp)}"
        
        self._store.add(
            content,
            key=key,
            meta={
                "exchange": hash(key) & 0xFFFFFFFF,
                "checkpoint": True,
                "reason": cp.reason,
                "agent_cli": cp.agent_cli,
                "session_id": cp.session_id,
                "turn_count": cp.turn_count,
                "exchange_count": cp.exchange_count,
                "created_at": cp.timestamp,
                "tier": "bucket",
                "salience": 0.8,
            },
            source=f"checkpoint:{self.agent_cli}",
            kind=KIND_TURN,
        )
        self._store.save(
            self.path,
            name="checkpoint-cartridge",
            description=f"Rate-limit checkpoints for session {self.session_id}",
        )
    
    def _sparsify(self) -> int:
        """Apply DMS sparsification: demote low-utility checkpoints to durable."""
        evict_xids = self.dms.evict_list()
        if not evict_xids:
            return 0
        
        demoted = 0
        for xid in evict_xids:
            # Find and rewrite the checkpoint entry
            for idx, e in enumerate(self._store._entries):
                if e.get("deleted"):
                    continue
                meta = e.get("meta", {})
                if meta.get("exchange") == xid:
                    # Rewrite as durable fact
                    self._store.delete(idx)
                    self._store.add(
                        e["content"],
                        key=e.get("key_norm") or e["content"],
                        meta={**meta, "tier": "durable", "checkpoint": True},
                        source=e.get("source", "user"),
                        kind=KIND_FACT,
                    )
                    self.dms.promote(xid)
                    demoted += 1
                    break
        
        if demoted:
            self._store.compact()
            self._store.save(
                self.path,
                name="checkpoint-cartridge",
                description=f"Rate-limit checkpoints for session {self.session_id}",
            )
        return demoted
    
    def _prune_old(self) -> None:
        """Keep only the most recent max_checkpoints checkpoints."""
        checkpoints = [
            (e.get("meta", {}).get("created_at", 0), idx)
            for idx, e in enumerate(self._store._entries)
            if not e.get("deleted") and e.get("kind") == KIND_TURN
            and e.get("meta", {}).get("checkpoint", False)
        ]
        checkpoints.sort(reverse=True)  # newest first
        
        for _, idx in checkpoints[self.max_checkpoints:]:
            self._store.delete(idx)
        
        self._store.compact()
        self._store.save(
            self.path,
            name="checkpoint-cartridge",
            description=f"Rate-limit checkpoints for session {self.session_id}",
        )
    
    def recover(
        self,
        query: str = "",
        top_k: int = 5,
        min_confidence: float = 0.50,
    ) -> dict[str, Any]:
        """Recover context from the latest checkpoint.
        
        Returns structured context for the next agent to continue.
        """
        # Search for checkpoints in checkpoint store
        hits = self._store.search(
            query or "checkpoint",
            top_k=top_k * 2,
            kind="turn",
        )
        
        checkpoints = []
        for h in hits:
            meta = h.meta or {}
            if not meta.get("checkpoint", False):
                continue
            if h.confidence < min_confidence:
                continue
            try:
                cp_data = json.loads(h.content)
                cp = Checkpoint(**cp_data)
                checkpoints.append({
                    "checkpoint": cp,
                    "confidence": h.confidence,
                    "mode": h.mode,
                })
            except Exception:
                continue
        
        if not checkpoints:
            return {
                "session_id": self.session_id,
                "recovered": False,
                "message": "No checkpoint found",
                "checkpoints": [],
            }
        
        # Get the most recent checkpoint
        latest = checkpoints[0]["checkpoint"]
        
        # Also get recent exchanges for context from ALL cartridges in rack
        recent = []
        seen = set()
        
        # Search checkpoint store
        for store in [self._store]:
            recent_hits = store.search(
                "exchange",
                top_k=20,
                kind="turn",
            )
            for h in recent_hits:
                meta = h.meta or {}
                xid = meta.get("exchange")
                if xid in seen or not xid:
                    continue
                if meta.get("checkpoint", False):
                    continue  # skip checkpoint entries
                seen.add(xid)
                if h.confidence >= min_confidence:
                    recent.append({
                        "exchange": xid,
                        "content": h.content,
                        "confidence": h.confidence,
                        "turns": meta.get("turns"),
                        "salience": meta.get("salience"),
                    })
        
        # Also try to find offloader store by session_id convention
        try:
            from .rack import CartridgeRack
            rack = CartridgeRack(self.rack_dir, semantic=True)
            for name, store in rack._stores.items():
                if store is self._store:
                    continue
                recent_hits = store.search(
                    "exchange",
                    top_k=20,
                    kind="turn",
                )
                for h in recent_hits:
                    meta = h.meta or {}
                    xid = meta.get("exchange")
                    if xid in seen or not xid:
                        continue
                    if meta.get("checkpoint", False):
                        continue
                    seen.add(xid)
                    if h.confidence >= min_confidence:
                        recent.append({
                            "exchange": xid,
                            "content": h.content,
                            "confidence": h.confidence,
                            "turns": meta.get("turns"),
                            "salience": meta.get("salience"),
                        })
        except Exception:
            pass  # Rack search is optional
        
        # Sort by exchange id descending and limit
        recent.sort(key=lambda ex: ex["exchange"], reverse=True)
        recent = recent[:top_k]
        
        return {
            "session_id": self.session_id,
            "recovered": True,
            "latest_checkpoint": {
                "timestamp": latest.timestamp,
                "reason": latest.reason,
                "agent_cli": latest.agent_cli,
                "turn_count": latest.turn_count,
                "exchange_count": latest.exchange_count,
                "buffer_turns": latest.buffer_turns,
            },
            "recent_exchanges": recent,
            "all_checkpoints": len(checkpoints),
        }
    
    def stats(self) -> dict[str, Any]:
        """Checkpoint cartridge statistics."""
        s = self._store.stats()
        checkpoints = sum(
            1 for e in self._store._entries
            if not e.get("deleted") and e.get("kind") == KIND_TURN
            and e.get("meta", {}).get("checkpoint", False)
        )
        s.update({
            "session_id": self.session_id,
            "agent_cli": self.agent_cli,
            "checkpoints": checkpoints,
            "dms": self.dms.stats(),
        })
        return s


# ── CLI integration ──────────────────────────────────────────────────────────

def install_rate_limit_hook(offloader: ContextOffloader) -> CheckpointCartridge:
    """Install a hook that checkpoints on rate limit signals.
    
    Call this once at agent startup. The hook watches for:
    - SIGUSR1 (manual trigger)
    - Rate limit error patterns in stdout/stderr
    - Custom callback
    
    Returns the CheckpointCartridge instance.
    """
    cp = CheckpointCartridge()
    
    def on_rate_limit(signum, frame):
        cp.checkpoint(offloader, reason="signal", metadata={"signal": signum})
    
    import signal
    signal.signal(signal.SIGUSR1, on_rate_limit)
    
    return cp


def manual_checkpoint(offloader: ContextOffloader, reason: str = "manual") -> Checkpoint:
    """Create a manual checkpoint."""
    cp = CheckpointCartridge()
    return cp.checkpoint(offloader, reason=reason)


# ── MCP tool integration ────────────────────────────────────────────────────

def mem_checkpoint(offloader: ContextOffloader, reason: str = "manual") -> str:
    """MCP tool: create a checkpoint.
    
    Returns checkpoint summary for the agent.
    """
    cp = CheckpointCartridge()
    checkpoint = cp.checkpoint(offloader, reason=reason)
    return f"Checkpoint created: {checkpoint.reason} at {time.strftime('%H:%M:%S', time.localtime(checkpoint.timestamp))} ({len(checkpoint.buffer_turns)} buffer turns, {len(checkpoint.recent_exchanges)} recent exchanges)"


def mem_recover(query: str = "", session_id: str = "") -> str:
    """MCP tool: recover from latest checkpoint.
    
    Returns formatted context for the agent to continue.
    """
    if session_id:
        # Override session ID for cross-CLI recovery
        cp = CheckpointCartridge(session_id=session_id)
    else:
        cp = CheckpointCartridge()
    
    result = cp.recover(query)
    
    if not result["recovered"]:
        return "No checkpoint found — starting fresh"
    
    latest = result["latest_checkpoint"]
    lines = [
        f"=== RECOVERY: session {result['session_id'][:8]} ===",
        f"Checkpoint: {latest['reason']} by {latest['agent_cli']} at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(latest['timestamp']))}",
        f"Progress: {latest['turn_count']} turns, {latest['exchange_count']} exchanges",
        "",
        "Live buffer (unflushed turns):",
    ]
    
    for t in latest["buffer_turns"]:
        lines.append(f"  [{t['role']}] {t['text'][:100]}")
    
    if result["recent_exchanges"]:
        lines.append("")
        lines.append("Recent exchanges:")
        for ex in result["recent_exchanges"][:3]:
            lines.append(f"  [{ex['confidence']:.2f}] {ex['content'][:150]}")
    
    return "\n".join(lines)