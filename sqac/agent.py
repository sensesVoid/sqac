"""Agent cartridge: persona + working state with typed sections."""

from __future__ import annotations

import secrets
import time
from pathlib import Path
from typing import Optional

from .format import (
    CartridgeHeader,
    WRITE_VERSION,
    read_cartridge,
    write_cartridge,
)
from .store import SqacStore, Hit

# ── constants ──────────────────────────────────────────────────────────────

AGENT_DIR = Path.home() / ".sqacm" / "agents"
ACTIVE_AGENT_PATH = Path.home() / ".sqacm" / "agent_active"
SINGLETON_SECTIONS = {"identity", "instruction", "preference"}
LIST_SECTIONS = {"goal", "decision", "note"}
ALL_SECTIONS = SINGLETON_SECTIONS | LIST_SECTIONS
DEFAULT_BUDGET = 4000


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _gen_key(section: str) -> str:
    return f"{section}-{secrets.token_hex(4)}"


# ── AgentCartridge ─────────────────────────────────────────────────────────

class AgentCartridge:
    """Persona + working state cartridge."""

    def __init__(self, path, name: str, role: str = "", semantic: bool = False):
        self.path = Path(path)
        self.name = name
        self.role = role
        self.semantic = semantic
        self._store: Optional[SqacStore] = None

    @classmethod
    def create(cls, path, name: str, role: str = "", semantic: bool = False) -> "AgentCartridge":
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        agent = cls(path, name, role, semantic)
        # Create empty cartridge with agent header
        hdr = CartridgeHeader(
            version=WRITE_VERSION,
            dims=1024,
            extra={
                "cartridge_type": "agent",
                "agent": {
                    "schema": 1,
                    "name": name,
                    "role": role,
                    "created": _now(),
                    "updated": _now(),
                },
            },
        )
        write_cartridge(str(path), hdr, {}, [])
        agent._store = SqacStore(semantic=semantic)
        agent._store._description = f"agent:{name}"
        # Save with agent header by re-writing
        agent._save()
        return agent

    @classmethod
    def load(cls, path, semantic: bool = False) -> "AgentCartridge":
        path = Path(path)
        cart = read_cartridge(str(path))
        h = cart.header
        agent_info = h.extra.get("agent", {})
        name = agent_info.get("name", path.stem)
        role = agent_info.get("role", "")
        agent = cls(path, name, role, semantic)
        agent._store = SqacStore.load(str(path))
        agent._store._description = f"agent:{name}"
        return agent

    @property
    def store(self) -> SqacStore:
        if self._store is None:
            self._store = SqacStore.load(str(self.path))
        return self._store

    def _save(self) -> None:
        self.store.save(str(self.path), name=self.name, description=f"agent:{self.name}")
        # Re-write header with agent extra (save may overwrite header)
        cart = read_cartridge(str(self.path))
        cart.header.extra["cartridge_type"] = "agent"
        cart.header.extra["agent"] = {
            "schema": 1,
            "name": self.name,
            "role": self.role,
            "created": cart.header.extra.get("agent", {}).get("created", _now()),
            "updated": _now(),
        }
        write_cartridge(str(self.path), cart.header, {}, cart.entries)

    def set(self, section: str, content: str, key: Optional[str] = None, priority: int = 1, tags: Optional[list] = None) -> int:
        """Set an entry in a section. Returns entry index."""
        if section not in ALL_SECTIONS:
            raise ValueError(f"unknown section: {section}")
        if tags is None:
            tags = []

        if key is not None:
            # Upsert by key
            for i, e in enumerate(self.store._entries):
                if e.get("meta", {}).get("key") == key and e.get("meta", {}).get("section") == section:
                    self.store._entries[i]["content"] = content
                    self.store._entries[i]["meta"]["priority"] = priority
                    self.store._entries[i]["meta"]["tags"] = tags
                    self.store._entries[i]["meta"]["updated"] = _now()
                    self._save()
                    return i
            # Not found, add new
            key = key

        if key is None and section in SINGLETON_SECTIONS:
            # Replace single existing entry for this section
            for i, e in enumerate(self.store._entries):
                if e.get("meta", {}).get("section") == section:
                    self.store._entries[i]["content"] = content
                    self.store._entries[i]["meta"]["priority"] = priority
                    self.store._entries[i]["meta"]["tags"] = tags
                    self.store._entries[i]["meta"]["updated"] = _now()
                    self._save()
                    return i
            # No existing, generate key
            key = f"{section}-0"

        if key is None and section in LIST_SECTIONS:
            key = _gen_key(section)

        if key is None:
            key = f"{section}-0"

        idx = self.store.add(
            content,
            key=key,
            kind="fact",
            source="agent",
            meta={"section": section, "priority": priority, "status": "open", "tags": tags, "key": key, "updated": _now()},
        )
        self._save()
        return idx

    def note(self, content: str, tags: Optional[list] = None) -> int:
        return self.set("note", content, tags=tags)

    def complete(self, key_or_query: str) -> bool:
        """Mark a goal/decision as done. Returns True if resolved uniquely."""
        # Try exact key match first
        for i, e in enumerate(self.store._entries):
            if e.get("meta", {}).get("key") == key_or_query:
                self.store._entries[i]["meta"]["status"] = "done"
                self._save()
                return True
        # Try fuzzy search
        hits = self.store.search(key_or_query, top_k=5)
        # Filter to open goals/decisions
        candidates = [h for h in hits if h.meta.get("status") == "open" and h.meta.get("section") in ("goal", "decision")]
        if len(candidates) == 1:
            # Find the entry index
            for i, e in enumerate(self.store._entries):
                if e.get("content") == candidates[0].content:
                    self.store._entries[i]["meta"]["status"] = "done"
                    self._save()
                    return True
        return False

    def search(self, query: str, top_k: int = 3) -> list[Hit]:
        return self.store.search(query, top_k=top_k)

    def bootstrap(self, max_chars: int = DEFAULT_BUDGET) -> str:
        """Build a bootstrap packet for agent context."""
        parts = []
        used = 0

        # 1. Identity + instructions + preferences (all)
        for section in ["identity", "instruction", "preference"]:
            entries = [e for e in self.store._entries if e.get("meta", {}).get("section") == section]
            for e in entries:
                text = f"[{section}] {e['content']}"
                if used + len(text) + 2 > max_chars:
                    break
                parts.append(text)
                used += len(text) + 2

        # 2. Open goals by priority then recency
        goals = [e for e in self.store._entries if e.get("meta", {}).get("section") == "goal" and e.get("meta", {}).get("status") == "open"]
        goals.sort(key=lambda e: (-e.get("meta", {}).get("priority", 1), e.get("meta", {}).get("updated", "")), reverse=True)
        # Actually sort by priority desc, then recency desc
        goals.sort(key=lambda e: e.get("meta", {}).get("updated", ""), reverse=True)
        goals.sort(key=lambda e: e.get("meta", {}).get("priority", 1), reverse=True)
        for e in goals:
            text = f"[goal] {e['content']}"
            if used + len(text) + 2 > max_chars:
                break
            parts.append(text)
            used += len(text) + 2

        # 3. Last 5 decisions
        decisions = [e for e in self.store._entries if e.get("meta", {}).get("section") == "decision"]
        decisions.sort(key=lambda e: e.get("meta", {}).get("updated", ""), reverse=True)
        for e in decisions[:5]:
            text = f"[decision] {e['content']}"
            if used + len(text) + 2 > max_chars:
                break
            parts.append(text)
            used += len(text) + 2

        # 4. Notes only if budget remains
        notes = [e for e in self.store._entries if e.get("meta", {}).get("section") == "note"]
        for e in notes:
            text = f"[note] {e['content']}"
            if used + len(text) + 2 > max_chars:
                break
            parts.append(text)
            used += len(text) + 2

        header = f"## Agent: {self.name}"
        if self.role:
            header += f" ({self.role})"
        return header + "\n" + "\n".join(parts)

    def stats(self) -> dict:
        sections = {}
        for e in self.store._entries:
            sec = e.get("meta", {}).get("section", "unknown")
            sections[sec] = sections.get(sec, 0) + 1
        return {
            "name": self.name,
            "role": self.role,
            "entries": len(self.store),
            "sections": sections,
        }

    def save(self) -> None:
        self._save()


# ── module-level helpers ───────────────────────────────────────────────────

def _agent_path(name: str) -> Path:
    return AGENT_DIR / f"{name}.sqac"


def list_agents() -> list[str]:
    if not AGENT_DIR.exists():
        return []
    return sorted(p.stem for p in AGENT_DIR.glob("*.sqac"))


def get_active_agent() -> Optional[str]:
    if ACTIVE_AGENT_PATH.exists():
        return ACTIVE_AGENT_PATH.read_text(encoding="utf-8").strip()
    return None


def set_active_agent(name: str) -> None:
    ACTIVE_AGENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE_AGENT_PATH.write_text(name, encoding="utf-8")


def load_agent(name: str, semantic: bool = False) -> AgentCartridge:
    path = _agent_path(name)
    if not path.exists():
        raise FileNotFoundError(f"agent not found: {name}")
    return AgentCartridge.load(path, semantic=semantic)


def create_agent(name: str, role: str = "", semantic: bool = False) -> AgentCartridge:
    path = _agent_path(name)
    return AgentCartridge.create(path, name, role, semantic)
