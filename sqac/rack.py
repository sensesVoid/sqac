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
  - auto-split: when a cartridge hits max_entries, a new shard is created
    so search speed stays constant
  - folder routing: kind-based directory organization (facts/, skills/, etc.)

The session->durable boundary is the graduation pass. A context bucket is
ephemeral (kind=turn); facts worth keeping graduate into a team cartridge
(kind=fact). See offloader.py design rule 4 and README.

Routing map (kind name -> cartridge name or folder):
    rack = CartridgeRack(dir, routes={"fact": "team", "skill": "skills"})
    rack = CartridgeRack(dir, routes={"fact": "facts", "skill": "skills"},
                         folder_routing=True)  # facts/, skills/ subdirs

Auto-split:
    rack = CartridgeRack(dir, max_entries=25000)
    # When facts.sqac hits 25K entries, next write creates facts__2.sqac
    # Search merges across all shards automatically.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .store import KIND_NAMES, KIND_TURN, Hit, SqacStore, resolve_kind

# Default auto-split threshold. Matches the capacity benchmark sweet spot:
# 25K entries ~ 27ms fuzzy scan, 132MB RSS, 12MB file.
DEFAULT_MAX_ENTRIES = 25_000

# Shard naming separator. facts.sqac -> facts__2.sqac, facts__3.sqac, etc.
_SHARD_SEP = "__"


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


def _is_shard_name(stem: str) -> bool:
    """Check if a filename stem looks like a shard (e.g. facts__2)."""
    return _SHARD_SEP in stem


def _base_name(stem: str) -> str:
    """Extract the base cartridge name from a shard stem.

    >>> _base_name("facts__2")
    'facts'
    >>> _base_name("facts")
    'facts'
    """
    parts = stem.split(_SHARD_SEP)
    return parts[0]


def _shard_index(stem: str) -> int:
    """Extract the shard index from a shard stem, or 1 for the base.

    >>> _shard_index("facts__2")
    2
    >>> _shard_index("facts")
    1
    """
    parts = stem.split(_SHARD_SEP)
    if len(parts) >= 2:
        try:
            return int(parts[-1])
        except ValueError:
            pass
    return 1


class CartridgeRack:
    """A named set of SqacStore cartridges with kind-based routing,
    auto-split, and optional folder organization.

    Cartridges register by NAME; a `directory` makes save/load by name
    default to <dir>/<name>.sqac (or <dir>/<folder>/<name>.sqac when
    folder_routing is enabled). `routes` maps a knowledge kind name
    ("fact"/"skill"/...) to a cartridge name or folder for write_routed().

    Auto-split: when a cartridge exceeds `max_entries`, the next write
    creates a new shard (<name>__2.sqac, <name>__3.sqac, ...) so search
    speed stays constant. Search merges across all shards automatically.

    Folder routing: with folder_routing=True, routes map kind to subdirectory
    names. "fact" → "facts" means writes go to <dir>/facts/<name>.sqac.
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
        max_entries: int = DEFAULT_MAX_ENTRIES,
        folder_routing: bool = False,
    ):
        self.directory = Path(directory) if directory else None
        self.routes: dict[str, str] = dict(routes or {})
        self.default = default
        self.semantic = semantic
        self.min_confidence = min_confidence
        self.max_entries = max_entries
        self.folder_routing = folder_routing
        self._stores: dict[str, SqacStore] = {}
        self._paths: dict[str, Path] = {}
        # Track all shard paths per base cartridge name for merge-search.
        # Key = base name (e.g. "facts"), value = list of shard paths
        # including the active store's path.
        self._shards: dict[str, list[Path]] = {}
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

    def _resolve_folder_path(self, kind: int | str | None, name: str) -> Path:
        """Resolve a path using folder routing.

        With folder_routing=True and routes={"fact": "facts"}, a write with
        kind="fact" and name="team-a" resolves to <dir>/facts/team-a.sqac.
        """
        if not self.folder_routing or self.directory is None:
            return self._resolve_path(name, None)
        kname = KIND_NAMES.get(resolve_kind(kind), "generic")
        folder = self.routes.get(kname)
        if folder:
            return self.directory / folder / f"{name}.sqac"
        return self._resolve_path(name, None)

    def register(self, name: str, path: str | Path | None = None) -> "CartridgeRack":
        """Attach an existing cartridge file to the rack by name."""
        p = self._resolve_path(name, path)
        if not p.exists():
            raise RackError(f"cartridge file not found: {p}")
        self._stores[name] = SqacStore.load(p, fuzzy_threshold=self.min_confidence)
        self._paths[name] = p
        self._track_shard(name, p)
        return self

    def _track_shard(self, base: str, path: Path) -> None:
        """Record a shard path for merge-search."""
        self._shards.setdefault(base, [])
        if path not in self._shards[base]:
            self._shards[base].append(path)

    def _auto_load_directory(self) -> None:
        """Mount every *.sqac in the rack directory (recursively, except
        the reserved session bucket). Files that fail to load are skipped;
        the failures are recorded in self._load_errors and surfaced as
        warnings so a silently-dropped memory file is visible."""
        if self.directory is None:
            return
        for p in sorted(self.directory.rglob("*.sqac")):
            stem = p.stem
            if stem == self.RESERVED_SESSION:
                continue
            # Derive the rack name from the relative path
            rel = p.relative_to(self.directory)
            rack_name = rel.with_suffix("").as_posix().replace("/", "__")
            if rack_name in self._stores:
                continue
            try:
                self._stores[rack_name] = SqacStore.load(
                    p, fuzzy_threshold=self.min_confidence
                )
                self._paths[rack_name] = p
                # Track shard under the base name
                base = _base_name(stem) if _is_shard_name(stem) else rack_name
                self._track_shard(base, p)
            except Exception as exc:  # FormatError / OSError / fingerprint
                self._load_errors.append((rack_name, str(exc)))
                logging.warning(
                    "rack: not mounting %s (%s): %s", p.name, rack_name, exc
                )

    def create(
        self,
        name: str,
        path: str | Path | None = None,
        description: str = "",
        overwrite: bool = False,
        kind: int | str | None = None,
    ) -> "CartridgeRack":
        """Mount a named cartridge, creating it when it doesn't exist.

        Safe by default: if the cartridge file already exists on disk it is
        LOADED, never wiped. Pass overwrite=True to reset deliberately (the
        on-disk file is replaced with an empty cartridge).

        With folder_routing, path is resolved from kind if not given.
        """
        p = path if path is not None else (
            self._resolve_folder_path(kind, name) if self.folder_routing
            else self._resolve_path(name, None)
        )
        if p.exists() and not overwrite:
            store = SqacStore.load(p, fuzzy_threshold=self.min_confidence)
            self._stores[name] = store
            self._paths[name] = p
            base = _base_name(p.stem) if _is_shard_name(p.stem) else name
            self._track_shard(base, p)
            return self
        p.parent.mkdir(parents=True, exist_ok=True)
        store = SqacStore(semantic=self.semantic, fuzzy_threshold=self.min_confidence)
        store.format_version = 3
        store.save(p, name=name, description=description)
        self._stores[name] = store
        self._paths[name] = p
        base = _base_name(p.stem) if _is_shard_name(p.stem) else name
        self._track_shard(base, p)
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

    # ── auto-split ──────────────────────────────────────────────────────

    def _maybe_split(self, name: str, kind: int | str | None) -> str:
        """Check if `name` is at threshold; if so, archive the full store
        as a numbered shard file and reset the original to an empty store.
        Returns `name` (writes always go to the original name).
        """
        if name not in self._stores:
            return name
        store = self._stores[name]
        if len(store) < self.max_entries:
            return name
        # Find the next shard index
        existing = self._shards.get(name, [])
        max_idx = 1
        for p in existing:
            if _is_shard_name(p.stem):
                idx = _shard_index(p.stem)
                if idx > max_idx:
                    max_idx = idx
        next_idx = max_idx + 1
        shard_name = f"{name}{_SHARD_SEP}{next_idx}"
        # Resolve shard path: same directory as the original
        orig_path = self._paths.get(name)
        if orig_path is not None:
            shard_path = orig_path.parent / f"{shard_name}.sqac"
        elif self.folder_routing and self.directory is not None:
            kname = KIND_NAMES.get(resolve_kind(kind), "generic")
            folder = self.routes.get(kname)
            if folder:
                shard_path = self.directory / folder / f"{shard_name}.sqac"
            else:
                shard_path = self.directory / f"{shard_name}.sqac"
        else:
            shard_path = self._resolve_path(shard_name, None)
        # 1. Save the full store to the shard file on disk
        store.save(shard_path, name=shard_name)
        # 2. Track it in the shard index
        self._shards.setdefault(name, [])
        self._shards[name].append(shard_path)
        # 3. Create a fresh empty store for the original name
        new_store = SqacStore(
            semantic=self.semantic, fuzzy_threshold=self.min_confidence
        )
        new_store.format_version = 3
        self._stores[name] = new_store
        # paths[name] stays the same (original path)
        # 4. Register the shard store so search can find it
        self._stores[shard_name] = store
        self._paths[shard_name] = shard_path
        logging.info(
            "rack: auto-split %s → %s (threshold %d)", name, shard_name, self.max_entries
        )
        return name

    def _shard_stores(self, base: str) -> list[SqacStore]:
        """Return all stores (active + shards) for a base cartridge name."""
        stores = []
        for p in self._shards.get(base, []):
            # Find the store for this path
            for n, pp in self._paths.items():
                if pp == p and n in self._stores:
                    stores.append(self._stores[n])
                    break
        # Also include the active store if it's not already tracked
        if base in self._stores and self._stores[base] not in stores:
            stores.append(self._stores[base])
        return stores

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
        """Write a memory into a named cartridge.

        Auto-split: if the cartridge exceeds max_entries, a new shard is
        created automatically.
        """
        active = self._maybe_split(name, kind)
        if active != name:
            name = active
        self._require(name).add(content, key=key, meta=meta, source=source, kind=kind)
        return self

    def write_routed(
        self,
        content: str,
        key: Optional[str] = None,
        meta: Optional[dict[str, Any]] = None,
        source: str = "rack",
        kind: int | str | None = None,
        project: Optional[str] = None,
    ) -> "CartridgeRack":
        """Write, routing to the cartridge for `kind` (or the default).

        With folder_routing, writes go to <dir>/<kind_folder>/<name>.sqac.
        Auto-split: if the target cartridge exceeds max_entries, a new shard
        is created automatically.

        project: optional sub-name within the kind folder (e.g. "team-a"
        within facts/). When None, uses the route name as the file stem.
        """
        target = self._target_for(kind)
        if target not in self._stores:
            self.create(target, kind=kind)
        self.write(target, content, key=key, meta=meta, source=source, kind=kind)
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

    def _collect_search_targets(
        self, cartridges: Optional[list[str]] = None
    ) -> list[str]:
        """Collect all cartridge names to search, including shards."""
        if cartridges is not None:
            # Search specific cartridges + their shards
            targets = []
            for c in cartridges:
                if c in self._stores:
                    targets.append(c)
                # Add shards for this base name
                base = _base_name(c) if _is_shard_name(c) else c
                for p in self._shards.get(base, []):
                    for n, pp in self._paths.items():
                        if pp == p and n not in targets and n in self._stores:
                            targets.append(n)
            return targets
        # Search all cartridges
        all_names = self.names()
        # Dedupe: only include the active store for each base, plus all shards
        seen_bases: set[str] = set()
        targets = []
        for n in all_names:
            base = _base_name(n) if _is_shard_name(n) else n
            if base not in seen_bases:
                seen_bases.add(base)
            targets.append(n)
        return targets

    def search(
        self,
        query: str,
        top_k: int = 3,
        kind: int | str | None = None,
        cartridges: Optional[list[str]] = None,
    ) -> list[Hit]:
        """Search across the rack and merge, ranked by confidence.

        Searches across all shards for each cartridge. Each Hit is tagged
        with meta["cartridge"] so the caller knows which memory file
        supplied the answer.
        """
        pool: list[tuple[float, Hit]] = []
        targets = self._collect_search_targets(cartridges)
        for name in targets:
            if name not in self._stores:
                continue
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
        targets = self._collect_search_targets(cartridges)
        for name in targets:
            if name not in self._stores:
                continue
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
        """Persist mounted cartridges back to their registered paths.

        `names=None` persists every mounted cartridge; an EMPTY list
        deliberately persists nothing (allowing filtered saves)."""
        if names is None:
            names = self.names()
        for name in names:
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
