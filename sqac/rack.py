"""CartridgeRack — manage many .sqac cartridges as one hot-swappable set.

One cartridge is a memory file; a user is not limited to one. Life gets
messy the moment you hold several (team facts, personal notes, a per-project
skill pack, a session bucket). CartridgeRack is the thin organizer:

  - register/load/unload named cartridges backed by a directory
  - cross-cartridge search: one query, hits merged and ranked with the
    cartridge they came from tagged on each result
  - route writes by knowledge kind to the right cartridge (the user selects
    a file once and forgets it)
  - graduation target: a single cartridge where the session->fact promotion
    pass lands

The session->durable boundary is the graduation pass. A context bucket is
ephemeral (kind=turn); facts worth keeping graduate into a team cartridge
(kind=fact). See offloader.py design rule 4 and README.

Routing map (kind name -> cartridge name):
    rack = CartridgeRack(dir, routes={"fact": "team", "skill": "skills"})
The rack keeps cartridges for any registered names; write_routed() picks the
target from the route table, falling back to a default cartridge when one is
set and the kind is unmapped. A routed or default cartridge that is not yet
mounted is created on demand (when a path is resolvable), so the example
above just works. Cartridges are loaded automatically from `directory` on
construction (auto_load=True, defaults to a named cartridge set; the
reserved session bucket is never auto-mounted), so "reopen, still knows"
holds for racks as it does for the offloader.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .store import KIND_NAMES, KIND_TURN, Hit, SqacStore, resolve_kind


class RackError(ValueError):
    """Raised for invalid cartridge-name / routing operations."""


# A grader decides, per exchange, whether that exchange is worth keeping.
# grad(graded_exchange) -> bool. Returning True = promote to durable fact.
Grader = Callable[["GradedExchange"], bool]


@dataclass
class GradedExchange:
    """One session exchange under graduation review.

    Carries everything the pass needs to judge and promote an exchange:
    the reconstructed text (content), its measured salience, its turn span,
    and the metadata the source bucket stored. A custom grader (LLM-backed,
    rule-based) inspects this object and returns True to promote.
    """

    xid: int
    content: str
    salience: float
    turns: tuple[int, int]
    meta: dict[str, Any] = field(default_factory=dict)
    promoted: bool = False


def _gather_exchanges(store: SqacStore) -> list[GradedExchange]:
    """Reconstruct the logical exchanges stored in a session bucket.

    The offloader writes one store entry PER KEY for each exchange, all
    sharing meta["exchange"] and meta["salience"]. This dedupes by exchange
    id, keeps the highest salience, and pairs it with the best content.
    """
    by_xid: dict[int, GradedExchange] = {}
    for e in store._entries:
        if e.get("deleted"):
            continue
        kind = e.get("kind", 0)
        if kind != KIND_TURN:
            continue
        meta = e.get("meta", {})
        xid = meta.get("exchange")
        if xid is None:
            continue
        g = by_xid.get(xid)
        content = e.get("content", "")
        salience = float(meta.get("salience", 0.0) or 0.0)
        tr = meta.get("turns")
        turns = (int(tr[0]), int(tr[1])) if isinstance(tr, (list, tuple)) and len(tr) >= 2 else (0, 0)
        if g is None:
            by_xid[xid] = GradedExchange(
                xid=int(xid), content=content, salience=salience,
                turns=turns, meta=dict(meta),
            )
        else:
            # keep the richest content + highest observed salience
            if len(content) > len(g.content):
                g.content = content
            if salience > g.salience:
                g.salience = salience
    # stable order by exchange id
    return [by_xid[x] for x in sorted(by_xid)]


def default_grader(threshold: float = 0.40) -> Grader:
    """Salience-threshold grader. Returns a grader closure.

    salience (see offloader._salience) rewards salient markers (errors,
    decisions, numbers), question shape, and length. Chitchat scores ~0.1;
    a rooted decision or a bug diagnosis usually exceeds 0.4.
    """

    def _grade(g: GradedExchange) -> bool:
        return g.salience >= threshold

    return _grade


def graduation_pass(
    source: SqacStore,
    target: SqacStore,
    threshold: float = 0.40,
    grader: Optional[Grader] = None,
    source_name: str = "session",
    key_prefix: str = "fact:",
) -> dict[str, Any]:
    """Promote worthy kind=turn exchanges from a session bucket into a
    durable kind=fact cartridge. Returns a report dict.

    Decision (default): promote when salience >= `threshold`. Pass `grader`
    to override — a callable(GradedExchange)->bool that may be LLM-backed.

    The promoted fact is written once per exchange (key = the exchange's
    anchors with a `key_prefix`, kind=KIND_FACT). meta keeps the source
    exchange id so the pass is rerunnable without duplicating: exchanges
    already present in the target under meta["graduated_from"] are skipped.
    """
    grader = grader or default_grader(threshold)
    existing = {
        e.get("meta", {}).get("graduated_from")
        for e in target._entries
        if isinstance(e.get("meta"), dict) and e.get("meta", {}).get("graduated_from") is not None
    }

    reviewed = 0
    promoted = 0
    skipped = 0
    for g in _gather_exchanges(source):
        reviewed += 1
        if g.xid in existing:
            skipped += 1
            continue
        if not g.content.strip():
            skipped += 1
            continue
        if grader(g):
            # content already reads "[turns a-b] Q: ... A: ..."; strip the
            # leading turn range label for a durable, model-facing fact.
            body = g.content
            if body.startswith("[turns"):
                idx = body.find("] ")
                if idx != -1:
                    body = body[idx + 2 :].strip()
            key = _fact_key(body, g, key_prefix)
            target.add(
                body,
                key=key,
                meta={
                    **g.meta,
                    "graduated_from": g.xid,
                    "source": source_name,
                },
                source=f"graduate:{g.xid}",
                kind="fact",
            )
            g.promoted = True
            promoted += 1
        else:
            skipped += 1

    return {
        "reviewed": reviewed,
        "promoted": promoted,
        "skipped": skipped,
        "threshold": threshold,
        "grader": getattr(grader, "__name__", type(grader).__name__),
    }


def _fact_key(body: str, g: GradedExchange, prefix: str) -> str:
    """Build a stable-ish key for the promoted fact: first-words + xid."""
    words = [w for w in body.split() if w.lower() not in _STOP][:5]
    slug = " ".join(words).lower() if words else f"exchange-{g.xid}"
    return f"{prefix}{slug} {g.xid}"


_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as", "into",
    "about", "that", "this", "and", "or", "but", "we", "our", "you", "your",
    "i", "me", "it", "its", "what", "how", "do", "does", "can", "could",
}


class CartridgeRack:
    """A named set of SqacStore cartridges plus kind-based routing.

    Cartridges register by NAME; a `directory` makes save/load by name
    default to <dir>/<name>.sqac. `routes` maps a knowledge kind name
    ("fact"/"skill"/...) to a cartridge name for write_routed(). `default`
    is the fallback cartridge when a kind has no route.

    With `directory` set, existing cartridges are auto-mounted on
    construction (auto_load=True). The reserved `session` cartridge — the
    ContextOffloader's ephemeral bucket — is never auto-mounted so durable
    search stays clean and one file isn't double-managed. Individual files
    that fail to load are skipped and reported in `self._load_errors`.
    """

    # Name of the ephemeral session bucket (ContextOffloader). Never
    # auto-mounted, so durable rack search doesn't include turn memory and
    # the offloader keeps single-owner of the file.
    RESERVED_SESSION = "session"

    def __init__(
        self,
        directory: str | Path | None = None,
        routes: Optional[dict[str, str]] = None,
        default: Optional[str] = None,
        semantic: bool = True,
        min_confidence: float = 0.60,
        auto_load: bool = True,
    ):
        self.directory = Path(directory) if directory else None
        self.routes: dict[str, str] = dict(routes or {})
        self.default = default
        self.semantic = semantic
        self.min_confidence = min_confidence
        self._stores: dict[str, SqacStore] = {}
        self._paths: dict[str, Path] = {}
        self._load_errors: list[tuple[str, str]] = []
        if self.directory is not None and auto_load:
            self._auto_load_directory()

    # ── mount/unmount ───────────────────────────────────────────────────

    def _resolve_path(self, name: str, path: str | Path | None) -> Path:
        if path is not None:
            return Path(path)
        if self.directory is not None:
            return self.directory / f"{name}.sqac"
        raise RackError(
            f"no path for cartridge {name!r}: pass a path or set a rack directory"
        )

    def register(self, name: str, path: str | Path | None = None) -> "CartridgeRack":
        """Attach an existing cartridge file to the rack by name."""
        p = self._resolve_path(name, path)
        if not p.exists():
            raise RackError(f"cartridge file not found: {p}")
        self._stores[name] = SqacStore.load(p, fuzzy_threshold=self.min_confidence)
        self._paths[name] = p
        return self

    def _auto_load_directory(self) -> None:
        """Mount every *.sqac in the rack directory (except the reserved
        session bucket). Files that fail to load are skipped; the failures
        are recorded in self._load_errors."""
        for p in sorted(self.directory.glob("*.sqac")):
            if p.stem == self.RESERVED_SESSION or p.stem in self._stores:
                continue
            try:
                self.register(p.stem, p)
            except Exception as exc:  # FormatError / OSError / fingerprint
                self._load_errors.append((p.stem, str(exc)))

    def create(
        self,
        name: str,
        path: str | Path | None = None,
        description: str = "",
        overwrite: bool = False,
    ) -> "CartridgeRack":
        """Mount a named cartridge, creating it when it doesn't exist.

        Safe by default: if the cartridge file already exists on disk it is
        LOADED, never wiped. Pass overwrite=True to reset deliberately (the
        on-disk file is replaced with an empty cartridge)."""
        p = self._resolve_path(name, path)
        if p.exists() and not overwrite:
            store = SqacStore.load(p, fuzzy_threshold=self.min_confidence)
            self._stores[name] = store
            self._paths[name] = p
            return self
        store = SqacStore(semantic=self.semantic, fuzzy_threshold=self.min_confidence)
        store.format_version = 3
        store.save(p, name=name, description=description)
        self._stores[name] = store
        self._paths[name] = p
        return self

    def unload(self, name: str) -> bool:
        return self._stores.pop(name, None) is not None

    def drop(self, name: str, recursive_path: bool = False) -> bool:
        """Drop a cartridge from the rack; optionally delete the file."""
        removed = self.unload(name)
        p = self._paths.pop(name, None)
        if recursive_path and p is not None and p.exists():
            p.unlink()
        return removed

    def __contains__(self, name: str) -> bool:
        return name in self._stores

    def __getitem__(self, name: str) -> SqacStore:
        return self._require(name)

    def names(self) -> list[str]:
        return list(self._stores)

    def _require(self, name: str) -> SqacStore:
        try:
            return self._stores[name]
        except KeyError:
            raise RackError(f"cartridge not mounted: {name!r}")

    # ── write path ──────────────────────────────────────────────────────

    def write(
        self,
        name: str,
        content: str,
        key: Optional[str] = None,
        meta: Optional[dict[str, Any]] = None,
        source: str = "rack",
        kind: int | str | None = None,
    ) -> "CartridgeRack":
        """Write a memory into a named cartridge."""
        self._require(name).add(content, key=key, meta=meta, source=source, kind=kind)
        return self

    def write_routed(
        self,
        content: str,
        key: Optional[str] = None,
        meta: Optional[dict[str, Any]] = None,
        source: str = "rack",
        kind: int | str | None = None,
    ) -> "CartridgeRack":
        """Write, routing to the cartridge for `kind` (or the default).

        The user says "store this fact" — the rack decides which file. The
        routed (or default) cartridge is created on demand when it is not
        yet mounted and a path is resolvable; with no route, no default,
        and no mounted fallback this raises RackError.
        """
        target = self._target_for(kind)
        if target not in self._stores:
            self.create(target)
        self._require(target).add(
            content, key=key, meta=meta, source=source, kind=kind
        )
        return self

    def _target_for(self, kind: int | str | None) -> str:
        kname = KIND_NAMES.get(resolve_kind(kind), "generic")
        name = self.routes.get(kname) or self.routes.get(str(kind))
        if name is not None:
            return name
        if self.default is not None:
            return self.default
        raise RackError(
            f"no routing for kind {kname!r} and no default cartridge set"
        )

    # ── read path ───────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 3,
        kind: int | str | None = None,
        cartridges: Optional[list[str]] = None,
    ) -> list[Hit]:
        """Search across the rack and merge, ranked by confidence.

        Each Hit is tagged with meta["cartridge"] so the caller (or the
        model) knows which memory file supplied the answer.
        """
        pool: list[tuple[float, Hit]] = []
        targets = [c for c in (cartridges or self.names()) if c in self._stores]
        for name in targets:
            for h in self._require(name).search(query, top_k=top_k, kind=kind):
                h.meta = dict(h.meta)
                h.meta["cartridge"] = name
                pool.append((h.confidence, h))
        pool.sort(key=lambda x: x[0], reverse=True)
        seen: set[tuple[str, str]] = set()
        out: list[Hit] = []
        for conf, h in pool:
            dedupe = (h.meta.get("cartridge"), h.content)
            if dedupe in seen:
                continue
            seen.add(dedupe)
            out.append(h)
            if len(out) >= top_k:
                break
        return out

    def search_grouped(
        self,
        query: str,
        top_k: int = 3,
        group_key: str = "skill",
        recall: int = 40,
        kind: int | str | None = None,
        cartridges: Optional[list[str]] = None,
    ) -> list[Hit]:
        """Group-aware search across the rack (see store.search_grouped)."""
        pool: list[tuple[float, Hit]] = []
        targets = [c for c in (cartridges or self.names()) if c in self._stores]
        for name in targets:
            for h in self._require(name).search_grouped(
                query, top_k=top_k, group_key=group_key, recall=recall, kind=kind
            ):
                h.meta = dict(h.meta)
                h.meta["cartridge"] = name
                pool.append((h.confidence, h))
        pool.sort(key=lambda x: x[0], reverse=True)
        seen: set[tuple[str, str]] = set()
        out: list[Hit] = []
        for conf, h in pool:
            dedupe = (h.meta.get("cartridge"), h.meta.get(group_key), h.content)
            if dedupe in seen:
                continue
            seen.add(dedupe)
            out.append(h)
            if len(out) >= top_k:
                break
        return out

    # ── graduation ──────────────────────────────────────────────────────

    def graduate(
        self,
        source,
        target_name: Optional[str] = None,
        threshold: float = 0.40,
        grader: Optional[Grader] = None,
        source_name: str = "session",
    ) -> dict[str, Any]:
        """Run the graduation pass into a cartridge in this rack.

        `source` may be a ContextOffloader, a SqacStore, another rack, or
        the NAME of a mounted cartridge in this rack. `target_name` selects
        the rack cartridge that receives the promoted facts (default: the
        rack's default cartridge).
        """
        target = self._require(target_name or self._require_default())
        source_store = self._as_store(source)
        return graduation_pass(
            source_store,
            target,
            threshold=threshold,
            grader=grader,
            source_name=source_name,
        )

    def _require_default(self) -> str:
        if self.default is not None and self.default in self._stores:
            return self.default
        raise RackError("no default cartridge mounted; pass target_name")

    def _as_store(self, src) -> SqacStore:
        # ContextOffloader: session bucket source (avoid a hard import cycle;
        # duck-typed on _store to keep the dependency direction one-way).
        if hasattr(src, "_store") and isinstance(getattr(src, "_store"), SqacStore):
            return src._store
        if isinstance(src, CartridgeRack):
            return src._require(src._require_default())
        if isinstance(src, str):
            return self._require(src)
        if isinstance(src, SqacStore):
            return src
        raise RackError(
            "graduation source must be a ContextOffloader, a SqacStore, a "
            "CartridgeRack, or a mounted cartridge name in this rack"
        )

    # ── persistence ─────────────────────────────────────────────────────

    def save(self, names: Optional[list[str]] = None) -> "CartridgeRack":
        """Persist mounted cartridges back to their registered paths."""
        for name in (names or self.names()):
            store = self._require(name)
            p = self._paths.get(name)
            if p is None:
                raise RackError(f"no path for cartridge {name!r}; pass one to save")
            kinds = store.stats()["kinds"]
            desc = f"{name} rack: {sum(kinds.values())} entries, " + ", ".join(
                f"{k}={v}" for k, v in sorted(kinds.items())
            )
            store.save(p, name=name, description=desc)
        return self

    def stats(self, name: Optional[str] = None) -> dict[str, Any]:
        if name is not None:
            s = self._require(name).stats()
            s["name"] = name
            return s
        return {n: self._require(n).stats() for n in self.names()}
