"""SqacStore — hybrid exact + fuzzy memory store.

Query flow (thesis Part IV, use cases 1+2):
  1. Exact lookup: normalized query -> O(1) dict hit -> confidence 1.0
  2. Fuzzy fallback: BSC key -> XOR+popcount scan over packed bytes
  3. Confidence-ranked results with source metadata

The store never shows a vector to the LLM: payloads are plaintext.
Deletes are tombstones (FLAG_DELETED), honoring the O(1) non-destructive
update property from the thesis.
"""

from __future__ import annotations

import collections
import functools
import hashlib
import heapq
import re
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .encoder import BSCEncoder, MiniLMSimHashEncoder
from .static_encoder import StaticSimHashEncoder
from .memid import new_mem_id
# Bundled Rust SIMD acceleration — 100-300x faster for fuzzy scan
from .simd import HAS_SIMD as _HAS_SIMD
from .simd import backend as _simd

from .format import (
    _HAS_LZ4,

    CartridgeHeader,
    Entry,
    FormatError,
    hamming,
    read_cartridge,
    write_cartridge,
)

# ── input sanitization ───────────────────────────────────────────────────────
# Controls what gets stored. Prevents prompt injection via stored content
# and keeps the cartridge size bounded.
_MAX_CONTENT_LENGTH = 50_000  # 50KB per entry — generous but bounded
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize_content(text: str) -> str:
    """Strip control characters and enforce length limits.

    Control characters can break JSON serialization and confuse LLM tokenizers.
    Length limits prevent a single malicious entry from bloating the cartridge.
    """
    text = _CONTROL_CHARS.sub("", text)
    text = text.strip()
    if len(text) > _MAX_CONTENT_LENGTH:
        text = text[:_MAX_CONTENT_LENGTH] + "... [truncated]"
    return text


def sanitize_key(text: str) -> str:
    """Normalize and sanitize a lookup key."""
    text = _CONTROL_CHARS.sub("", text)
    text = text.strip()
    if len(text) > 1024:
        text = text[:1024]
    return text


# ── knowledge kinds (v3 flag bits, format.py) ─────────────────────────
# Closed enum: 0 = generic/unknown (absent kind = scan always, back-compat).
# kind answers "what sort of knowledge is this" — independent of the
# indexing tiers (exact/lexical/semantic), which answer "how is it indexed".
KIND_GENERIC = 0
KIND_FACT = 1
KIND_SKILL = 2
KIND_DOC = 3
KIND_TURN = 4  # conversation-turn offload (context bucket)
KIND_TASK = 5  # v1.4.0 task-state cartridge entries (working memory)
KIND_NAMES = {
    KIND_GENERIC: "generic",
    KIND_FACT: "fact",
    KIND_SKILL: "skill",
    KIND_DOC: "doc",
    KIND_TURN: "turn",
    KIND_TASK: "task",
}
_NAME_TO_KIND = {v: k for k, v in KIND_NAMES.items()}


def resolve_kind(kind) -> int:
    """Accept int or name ("fact"/"skill"/"doc"/"turn"); 0 for unknown names."""
    if kind is None:
        return KIND_GENERIC
    if isinstance(kind, int):
        return kind
    return _NAME_TO_KIND.get(str(kind).lower(), KIND_GENERIC)


def _make_sem_encoder(model_name: str, dims: int):
    """Build the semantic encoder named in a cartridge header.

    static-potion-*  -> pure-numpy int8 static tier (no torch needed)
    anything else    -> MiniLM SimHash (torch+transformers)
    """
    if model_name.startswith("static-"):
        return StaticSimHashEncoder(dims=dims)
    return MiniLMSimHashEncoder(dims=dims, model=model_name)

_WS = re.compile(r"\s+")
_NONALNUM = re.compile(r"[^\w\s]")


@functools.lru_cache(maxsize=2048)
def normalize(text: str) -> str:
    """Canonical form for exact lookup."""
    t = _NONALNUM.sub(" ", text.lower())
    return _WS.sub(" ", t).strip()


@dataclass
class Hit:
    """One retrieval result. This is what the LLM layer receives."""

    content: str
    confidence: float
    source: str
    mode: str  # "exact" | "fuzzy"
    meta: dict[str, Any] = field(default_factory=dict)
    trust: dict[str, Any] = field(default_factory=dict)  # injection analysis
    mem_id: str = ""

    def as_dict(self) -> dict:
        d = {
            "content": self.content,
            "confidence": round(self.confidence, 4),
            "source": self.source,
            "mode": self.mode,
            "meta": dict(self.meta),
            "trust": dict(self.trust),
        }
        if self.mem_id:
            d["mem_id"] = self.mem_id
        return d


class SqacStore:
    """Hybrid memory: exact dict + fuzzy VSA scan, text payloads out.

    Three retrieval tiers, tried in order:
      1. exact     — normalized-key dict, O(1), confidence 1.0
      2. lexical   — BSC trigram bundling (typos, word overlap)
      3. semantic  — MiniLM SimHash (synonyms, paraphrase; optional,
                     requires torch+transformers)

    Production features:
      - Input sanitization: control chars stripped, content length bounded
      - Query cache: LRU cache on encoded query vectors (invalidated on write)
      - File locking: atomic writes via fcntl + thread lock
      - Compaction: remove tombstoned entries to reclaim space
    """

    # Both VSA tiers share the same noise floor (~0.5) and threshold
    # regime: lexical true matches >= 0.6, semantic paraphrases ~0.7+.
    DEFAULT_FUZZY_THRESHOLD = 0.60
    DEFAULT_SEMANTIC_THRESHOLD = 0.60  # measured: x86→ARM64 = 0.65, floor = 0.49
    DEFAULT_CACHE_SIZE = 512  # max cached query encodings

    def __init__(
        self,
        encoder: Optional[BSCEncoder | MiniLMSimHashEncoder] = None,
        dims: int = 1024,
        fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
        semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
        semantic: bool = False,
        semantic_model: Optional[str] = None,
        cache_size: int = DEFAULT_CACHE_SIZE,
        normalizer=None,
    ):
        self._normalizer = normalizer
        self.encoder = encoder or BSCEncoder(dims=dims, normalizer=normalizer)
        self._norm = normalizer if normalizer is not None else normalize
        self.dims = self.encoder.dims
        self.fuzzy_threshold = fuzzy_threshold
        self.semantic_threshold = semantic_threshold
        self.semantic = False
        self._sem_encoder = None
        self.cartridge_type = "memory"
        # Query cache: maps (query_hash, tier) -> (packed_bits, write_version)
        self._cache_size = cache_size
        self._query_cache: collections.OrderedDict[tuple[str, str], tuple[bytes, int]] = (
            collections.OrderedDict()
        )
        self._write_version = 0  # increments on every write; used for granular cache invalidation
        if semantic:
            if semantic_model and semantic_model.startswith("static"):
                self._sem_encoder = StaticSimHashEncoder(dims=self.dims)
            elif semantic_model:
                self._sem_encoder = MiniLMSimHashEncoder(dims=self.dims, model=semantic_model)
            else:
                # auto: prefer the light static tier (pure numpy, 8MB),
                # fall back to the MiniLM tier (torch) when its files are absent
                if StaticSimHashEncoder.available():
                    self._sem_encoder = StaticSimHashEncoder(dims=self.dims)
                elif MiniLMSimHashEncoder.available():
                    self._sem_encoder = MiniLMSimHashEncoder(dims=self.dims)
                else:
                    if semantic_model is None:
                        warnings.warn(
                            "semantic=True requested but no semantic encoder is "
                            "available; falling back to lexical-only",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        self._sem_encoder = None
                    else:
                        raise RuntimeError(
                            "semantic=True needs the static model files (see "
                            "sqac/static_encoder.py) or torch+transformers for MiniLM"
                        )
            self.semantic = self._sem_encoder is not None

        self._entries: list[dict[str, Any]] = []  # live entries
        self._exact: dict[str, int] = {}  # normalized key -> entry idx
        self._keys: list[bytes] = []  # packed lexical key vectors
        self._ckeys: list[bytes] = []  # packed lexical content vectors
        self._sem_keys: list[bytes] = []  # packed semantic key vectors
        self._sem_ckeys: list[bytes] = []  # packed semantic content vectors
        self._matrices: dict[str, tuple] = {}  # lazily unpacked (n, D) bit matrices
        self._stats_cache: tuple[int, dict] | None = None
        self._alive = 0
        self._kind_counts: dict[str, int] = {}
        self._id_index: dict[str, int] = {}
        self._rebuild_id_index()
    def _invalidate_cache(self) -> None:
        """Increment write version to invalidate stale cache entries.
        
        This is more granular than clearing the entire cache - entries created
        before this write will be considered stale on next lookup, but valid
        entries from before the write are retained.
        """
        self._write_version += 1
        self._stats_cache = None
        self._matrices.clear()  # bit matrices are stale on any write

    # ── write path ───────────────────────────────────────────────────────

    def _cached_encode(self, text: str, tier: str) -> bytes:
        """Encode text with LRU cache. Uses write version for granular invalidation."""
        cache_key = (hashlib.sha256(text.encode()).hexdigest()[:16], tier)
        if cache_key in self._query_cache:
            cached_bits, cached_version = self._query_cache[cache_key]
            if cached_version == self._write_version:
                self._query_cache.move_to_end(cache_key)
                return cached_bits
            # stale version - fall through to recompute
        if tier == "semantic" and self._sem_encoder:
            bits = self._sem_encoder.encode_bits(text)
        else:
            bits = self.encoder.encode_bits(text)
        self._query_cache[cache_key] = (bits, self._write_version)
        if len(self._query_cache) > self._cache_size:
            self._query_cache.popitem(last=False)
        return bits

    def add_with_mem_id(
        self,
        content: str,
        mem_id: str = "",
        **kwargs: Any,
    ) -> int:
        """add() with an externally-preserved mem_id (task graduation).

        The mem_id is honored as-is (never minted); the id index points at
        the new entry. meta: pass via kwargs (or it is {} for us to fill).
        """
        m = dict(kwargs.pop("meta", None) or {})
        if mem_id:
            m["mem_id"] = mem_id
        kwargs.pop("mem_id", None)
        idx = self.add(content, meta=m, **kwargs)
        if mem_id:
            self._id_index[mem_id] = idx
        return idx

    def add(
        self,
        content: str,
        key: Optional[str] = None,
        meta: Optional[dict[str, Any]] = None,
        source: str = "user",
        kind: int | str | None = KIND_GENERIC,
        validate: bool = True,
    ) -> int:
        """Teach the store one fact. O(1) amortized, non-destructive.

        kind: knowledge kind (KIND_* int or name: "fact"/"skill"/"doc"/
        "turn"). Kinds enable filtered retrieval and kind-aware ranking;
        they do not affect which index tiers an entry participates in.
        
        validate: if True (default), run credential detection and prompt
        injection analysis. Set to False for trusted bulk loads to skip
        these checks for performance.
        """
        content = sanitize_content(content)
        if not content:
            raise ValueError("content must be non-empty")

        if validate:
            # Credential detection — reject secrets before storage
            from .credentials import validate_content
            validate_content(content, strict=True)

            # Prompt injection analysis
            from .injection import analyze_content
            inj = analyze_content(content)
        else:
            from .injection import InjectionResult, RiskLevel
            inj = InjectionResult(risk=RiskLevel.SAFE, score=0.0, patterns=[], recommendation="skipped")

        key_text = sanitize_key(key) if key is not None else content
        norm = self._norm(key_text)
        self._invalidate_cache()

        kind_resolved = resolve_kind(kind)
        meta = dict(meta or {})
        if not meta.get("mem_id"):
            meta["mem_id"] = new_mem_id()
        idx = len(self._entries)
        self._entries.append(
            {
                "content": content,
                "key_norm": norm,
                "meta": meta,
                "source": source,
                "kind": kind_resolved,
                "trust": inj.to_dict(),
            }
        )
        kname = KIND_NAMES.get(kind_resolved, "generic")
        self._kind_counts[kname] = self._kind_counts.get(kname, 0) + 1
        self._alive += 1
        self._keys.append(self.encoder.encode_bits(key_text))
        self._ckeys.append(self.encoder.encode_bits(content))
        if self.semantic:
            self._sem_keys.append(self._sem_encoder.encode_bits(key_text))
            self._sem_ckeys.append(self._sem_encoder.encode_bits(content))
        self._exact[norm] = idx
        self._id_index[meta["mem_id"]] = idx
        return idx

    def delete(self, idx: int) -> bool:
        """Tombstone an entry (O(1))."""
        if not (0 <= idx < len(self._entries)):
            return False
        norm = self._entries[idx]["key_norm"]
        already = bool(self._entries[idx].get("deleted"))
        self._entries[idx]["deleted"] = True
        if not already:
            kname = KIND_NAMES.get(self._entries[idx].get("kind", KIND_GENERIC), "generic")
            if self._kind_counts.get(kname):
                self._kind_counts[kname] -= 1
            self._alive -= 1
            if self._alive < 0:
                self._alive = 0
        # Only drop the exact mapping if it still points at this entry: a
        # later add with the same key may have claimed it, and that sibling
        # must keep its O(1) exact tier.
        if self._exact.get(norm) == idx:
            self._exact.pop(norm, None)
        self._invalidate_cache()
        return True

    def compact(self) -> dict:
        """Remove tombstoned entries and rebuild the index in-place.

        This reclaims space from deleted entries. The store is rebuilt
        from the surviving entries, so all indices are fresh.
        Returns a report dict.
        """
        before = len(self._entries)
        alive = [e for e in self._entries if not e.get("deleted")]
        removed = before - len(alive)
        if removed == 0:
            return {"before": before, "after": before, "removed": 0}
        # Rebuild from scratch
        self._entries = []
        self._exact = {}
        self._keys = []
        self._ckeys = []
        self._sem_keys = []
        self._sem_ckeys = []
        self._matrices.clear()
        self._query_cache.clear()
        for e in alive:
            # ensure meta is a dict and backfill mem_id if missing
            meta = e.get("meta")
            if not isinstance(meta, dict):
                meta = {}
                e["meta"] = meta
            if not meta.get("mem_id"):
                meta["mem_id"] = new_mem_id()
            self._entries.append(e)
            norm = e["key_norm"]
            idx = len(self._entries) - 1
            self._keys.append(self.encoder.encode_bits(norm))
            self._ckeys.append(self.encoder.encode_bits(e["content"]))
            if self.semantic and self._sem_encoder:
                self._sem_keys.append(self._sem_encoder.encode_bits(norm))
                self._sem_ckeys.append(self._sem_encoder.encode_bits(e["content"]))
            if norm:
                self._exact[norm] = idx
        self._alive = len(self._entries)
        counts: dict[str, int] = {}
        for e in self._entries:
            kname = KIND_NAMES.get(e.get("kind", KIND_GENERIC), "generic")
            counts[kname] = counts.get(kname, 0) + 1
        self._kind_counts = counts
        self._rebuild_id_index()
        return {"before": before, "after": len(alive), "removed": removed}

    def __len__(self) -> int:
        return self._alive

    def get(self, mem_id: Any) -> Optional[dict[str, Any]]:
        """O(1) lookup of an entry by its canonical mem_id.

        Returns the entry dict when found and not tombstoned, else None.
        The index is not cleared on delete, so tombstoned entries are
        filtered here.
        """
        if mem_id is None:
            return None
        idx = self._id_index.get(mem_id)
        if idx is None:
            # Fallback: linear scan in case the index is stale/missing,
            # then repair the index.
            for i, e in enumerate(self._entries):
                meta = e.get("meta") if isinstance(e.get("meta"), dict) else {}
                if meta.get("mem_id") == mem_id:
                    idx = i
                    self._id_index[mem_id] = i
                    break
            if idx is None:
                return None
        if not (0 <= idx < len(self._entries)):
            return None
        e = self._entries[idx]
        if e.get("deleted"):
            return None
        return e

    def _rebuild_id_index(self) -> None:
        self._id_index = {}
        for i, e in enumerate(self._entries):
            meta = e.get("meta") if isinstance(e.get("meta"), dict) else {}
            mid = meta.get("mem_id") if isinstance(meta, dict) else None
            if mid is not None:
                self._id_index[mid] = i

    def delete_entry_by_mem_id(self, mem_id: str) -> Optional[int]:
        """Tombstone the entry with the given mem_id. Returns idx or None."""
        if not mem_id:
            return None
        idx = self._id_index.get(mem_id)
        if idx is None:
            for i, e in enumerate(self._entries):
                m = e.get("meta")
                if isinstance(m, dict) and m.get("mem_id") == mem_id:
                    idx = i
                    break
        if idx is None or idx >= len(self._entries):
            return None
        ok = self.delete(idx)
        return idx if ok else None

    def _backfill_mem_ids(self) -> None:
        for e in self._entries:
            meta = e.get("meta")
            if not isinstance(meta, dict):
                meta = {}
                e["meta"] = meta
            if not meta.get("mem_id"):
                meta["mem_id"] = new_mem_id()

    # ── read path ────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 3, kind: int | str | None = None, use_cache: bool = True) -> list[Hit]:
        """Exact-first, fuzzy-fallback retrieval with confidence.

        kind: optional filter (KIND_* int or name). None = all kinds.
        Kinds filter which entries are eligible; they never change which
        tiers run — the exact/lexical/semantic merge is unchanged.
        """
        t0 = time.perf_counter()
        norm = self._norm(query)
        entries = self._entries
        want_kind: Optional[int] = None
        if kind is not None:
            want_kind = kind if isinstance(kind, int) else resolve_kind(kind)

        def eligible(i: int) -> bool:
            if entries[i].get("deleted"):
                return False
            k = entries[i].get("kind", KIND_GENERIC)
            if want_kind is not None:
                if k != want_kind:
                    return False
            elif k == KIND_TASK:
                # v1.4.0: tasks are working memory; excluded from ordinary
                # search (task_list and the attention block are their surface)
                return False
            return True

        def _hit(idx: int, mode: str, confidence: float) -> Hit:
            e = self._entries[idx]
            meta = dict(e.get("meta") or {})
            return Hit(
                content=e["content"],
                confidence=confidence,
                source=e["source"],
                mode=mode,
                mem_id=str(meta.get("mem_id", "")),
                meta={
                    **meta,
                    "kind": KIND_NAMES.get(e.get("kind", KIND_GENERIC), "generic"),
                    "key_norm": e.get("key_norm"),
                    "latency_ms": _ms(time.perf_counter() - t0),
                },
                trust=dict(e.get("trust") or {}),
            )

        if top_k <= 0:
            return []

        # 1) Exact hit: O(1), confidence 1.0. Callers building a ranked window
        #    (search_grouped, rack merge) rely on up to top_k results even for
        #    exact keys, so the exact hit seeds the results and the fuzzy
        #    tiers fill the rest rather than short-circuiting entirely.
        out: list[Hit] = []
        seen_idx: set[int] = set()
        idx = self._exact.get(norm)
        if idx is not None and eligible(idx):
            out.append(_hit(idx, "exact", 1.0))
            seen_idx.add(idx)

        # 2-3) Fuzzy + Semantic tiers
        if want_kind is None:
            live = [i for i, e in enumerate(entries) if not e.get("deleted")]
        else:
            live = [i for i, e in enumerate(entries)
                    if not e.get("deleted") and e.get("kind", KIND_GENERIC) == want_kind]
        if len(out) < top_k:
            tier_results = self._search_tiers(query, live, use_cache, eligible, seen_idx, top_k - len(out))
            for sim, i, mode in tier_results:
                if eligible(i):
                    out.append(_hit(i, mode, sim))
        return out

    def _search_tiers(
        self,
        query: str,
        live: list[int],
        use_cache: bool,
        eligible: callable,
        seen_idx: set[int],
        top_k: int,
    ) -> list[tuple[float, int, str]]:
        """Run lexical and semantic tiers, return sorted (sim, idx, mode) tuples.
        
        This is shared between search() and search_grouped() to avoid duplication.
        """
        qbits = self._cached_encode(query, "lexical") if use_cache else self.encoder.encode_bits(query)
        results: list[tuple[float, int, str]] = []
        if not live:
            return results
        
        sims = self._bulk_similarity(qbits, live, tier="lexical")
        for i, sim in zip(live, sims):
            if sim >= self.fuzzy_threshold:
                results.append((sim, i, "fuzzy"))
        
        if self.semantic:
            qsem = self._cached_encode(query, "semantic") if use_cache else self._sem_encoder.encode_bits(query)
            sem_sims = self._bulk_similarity(qsem, live, tier="semantic")
            for i, sim in zip(live, sem_sims):
                if sim >= self.semantic_threshold:
                    results.append((sim, i, "semantic"))
        
        if seen_idx:
            return heapq.nlargest(top_k, ((sim, i, mode) for sim, i, mode in results if i not in seen_idx))
        return heapq.nlargest(top_k, results)

    def search_or_none(self, query: str, min_confidence: float = 0.75) -> Optional[Hit]:
        """Convenience: best hit above threshold, else None."""
        hits = self.search(query, top_k=1)
        if hits and hits[0].confidence >= min_confidence:
            return hits[0]
        return None

    def search_grouped(
        self,
        query: str,
        top_k: int = 3,
        group_key: str = "skill",
        recall: int = 40,
        kind: int | str | None = None,
    ) -> list[Hit]:
        """Group-aware ranking for multi-key packs (skills, docs-by-entity).

        Multi-key packs store many entries pointing at one logical item
        (meta[group_key]). Sibling triggers of a *wrong* item can crowd the
        flat top-k, so ranking must happen per group: fetch a generous
        recall window, keep each group's best hit, return the top_k groups
        ranked by their best member. Measured on the 50-problem skill bench:
        potion flat 45/50 -> grouped 49/50.

        Groups are (kind, meta[group_key]) pairs so same-named items of
        different kinds don't merge. kind= optionally restricts the whole
        search to one knowledge kind.

        Qualifying hits WITHOUT the group meta key are not dropped: each
        one is kept as its own singleton bucket, so grouped recall is a
        superset of what flat recall surfaces (never silently hides a hit).
        """
        hits = self.search(query, top_k=recall, kind=kind)
        best: dict[tuple[int, str], Hit] = {}
        orphan = 0
        for h in hits:
            g = h.meta.get(group_key, "")
            if not g:
                # meta-less hit: unique bucket per entry so it survives grouping
                orphan += 1
                gk = ("__ungrouped__", orphan)
            else:
                gk = (resolve_kind(h.meta.get("kind")), str(g))
            if gk not in best or h.confidence > best[gk].confidence:
                best[gk] = h
        return sorted(best.values(), key=lambda h: -h.confidence)[:top_k]

    # ── persistence ──────────────────────────────────────────────────────

    def save(self, path: str | Path, name: str = "", description: str = "") -> None:
        """Persist to a .sqac file.

        Writes the current v2 binary layout: auxiliary packed vectors
        (content, semantic key/content) are stored in a per-entry raw
        binary block.
        """
        entries = []
        for idx, (e, kbits, ckbits) in enumerate(zip(self._entries, self._keys, self._ckeys)):
            payload = {
                "content": e["content"],
                "key_norm": e["key_norm"],
                "source": e["source"],
                "kind": KIND_NAMES.get(e.get("kind", KIND_GENERIC), "generic"),
                **({"meta": e["meta"]} if e["meta"] else {}),
            }
            # v2+: always write raw binary vectors (no hex fallback)
            binvec = bytes(ckbits)
            if self.semantic:
                binvec += bytes(self._sem_keys[idx]) + bytes(self._sem_ckeys[idx])
            entries.append(
                Entry(
                    key_bits=bytearray(kbits),
                    payload=payload,
                    deleted=bool(e.get("deleted")),
                    binvec=binvec,
                    kind=e.get("kind", KIND_GENERIC),
                )
            )
        enc_name = f"bsc-ngram-v1:{self.encoder.seed}:{self.encoder.ngram}"
        if self.semantic:
            enc_name += f"|sem-v1:{self._sem_encoder.model_name}"
        header = CartridgeHeader(
            dims=self.dims,
            encoder_name=enc_name,
            name=name,
            description=description,
        )
        write_cartridge(path, header, {}, entries)

    @classmethod
    def load(cls, path: str | Path, fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
             normalizer=None) -> "SqacStore":
        """Load a cartridge. Refuses incompatible encoder/fingerprint.

        fuzzy_threshold defaults to the same value as the constructor
        (DEFAULT_FUZZY_THRESHOLD) so a cartridge behaves identically
        before and after a save/load roundtrip.
        """
        cart = read_cartridge(path)
        h = cart.header
        seed, ngram = _parse_encoder_name(h.encoder_name)
        enc = BSCEncoder(dims=h.dims, seed=seed, ngram=ngram, normalizer=normalizer)
        expected = h.compute_fingerprint({})  # seeded vocab is empty by design
        if h.vocab_fingerprint and h.vocab_fingerprint != expected:
            raise FormatError("vocab fingerprint mismatch: cartridge incompatible")
        store = cls(
            encoder=enc,
            dims=h.dims,
            fuzzy_threshold=fuzzy_threshold,
            semantic="|sem-v1:" in h.encoder_name,
            normalizer=normalizer,
        )
        store._description = h.description
        store.cartridge_type = h.cartridge_type
        if store.semantic:
            store._sem_encoder = _make_sem_encoder(
                h.encoder_name.split("|sem-v1:", 1)[1].split(":", 1)[0], h.dims
            )
        for e in cart.entries:
            norm = e.payload.get("key_norm", "")
            kind = e.kind if e.kind else resolve_kind(e.payload.get("kind"))
            idx = len(store._entries)
            meta = e.payload.get("meta", {})
            if not isinstance(meta, dict):
                meta = {}
            store._entries.append(
                {
                    "content": e.payload["content"],
                    "key_norm": norm,
                    "meta": meta,
                    "source": e.payload.get("source", "user"),
                    "kind": kind,
                    **({"deleted": True} if e.deleted else {}),
                }
            )
            store._keys.append(bytes(e.key_bits))
            # v2: auxiliary vectors in the raw binary block; v1: hex in payload.
            # Lexical content vector falls back to the key vector when absent.
            if e.binvec:
                kl = h.dims // 8
                # layout: [ckey][sem_key][sem_ckeys] — key_bits travels separately
                store._ckeys.append(e.binvec[0:kl] or bytes(e.key_bits))
                # sem lists stay EMPTY when semantic is off — add() only
                # appends to them on the semantic path; keep that invariant
                if store.semantic:
                    store._sem_keys.append(
                        e.binvec[kl : 2 * kl] if len(e.binvec) >= 2 * kl else None
                    )
                    store._sem_ckeys.append(
                        e.binvec[2 * kl : 3 * kl] if len(e.binvec) >= 3 * kl else None
                    )
            else:
                store._ckeys.append(
                    bytes.fromhex(e.payload["content_key"])
                    if e.payload.get("content_key")
                    else bytes(e.key_bits)
                )
                if store.semantic:
                    sk = e.payload.get("sem_key", "")
                    sck = e.payload.get("sem_content_key", "")
                    store._sem_keys.append(bytes.fromhex(sk) if sk else None)
                    store._sem_ckeys.append(bytes.fromhex(sck) if sck else None)
            if norm and not e.deleted:
                store._exact[norm] = idx
        # Backfill mem_id for entries that predate it (old cartridges).
        store._backfill_mem_ids()
        for e in store._entries:
            if e.get("deleted"):
                continue
            kname = KIND_NAMES.get(e.get("kind", KIND_GENERIC), "generic")
            store._kind_counts[kname] = store._kind_counts.get(kname, 0) + 1
            store._alive += 1
        store._rebuild_id_index()
        if store.semantic:
            store._backfill_semantic()
        return store

    # ── stats ────────────────────────────────────────────────────────────

    def _backfill_semantic(self) -> None:
        """Batch-encode semantic vectors for entries missing them
        (cartridges saved before the semantic tier existed)."""
        missing = [i for i, v in enumerate(self._sem_keys) if v is None]
        if not missing:
            return
        contents = [self._entries[i]["content"] for i in missing]
        packed = self._sem_encoder.encode_bits_batch(contents)
        for i, p in zip(missing, packed):
            self._sem_keys[i] = p
        # content vectors too
        missing_c = [i for i, v in enumerate(self._sem_ckeys) if v is None]
        if missing_c:
            packed_c = self._sem_encoder.encode_bits_batch(
                [self._entries[i]["content"] for i in missing_c]
            )
            for i, p in zip(missing_c, packed_c):
                self._sem_ckeys[i] = p

    def _bitmatrix(self, which: str):
        """Unpack a packed vector list into an (n, D) uint8 matrix, cached.

        `which` names one of the four packed lists; the cache is per-list
        because they share length and would otherwise collide.
        """
        import numpy as np

        lists = {
            "keys": self._keys,
            "ckeys": self._ckeys,
            "sem_keys": self._sem_keys,
            "sem_ckeys": self._sem_ckeys,
        }
        keys = lists[which]
        cache = self._matrices.setdefault(which, (None, -1))
        if cache[0] is None or cache[1] != len(keys):
            n = len(keys)
            mat = np.frombuffer(b"".join(keys), dtype=np.uint8).reshape(n, self.dims // 8)
            bits = np.unpackbits(mat, axis=1)  # (n, D), MSB-first matches pack order
            cache = (bits, n)
            self._matrices[which] = cache
        return cache[0]

    def _bulk_similarity(self, qbits: bytes, live: list[int], tier: str = "lexical") -> list[float]:
        """Vectorized 1 - hamming/D against key and content matrices.

        Fast path: Rust SIMD (sqac_simd) — 100-300x faster than Python.
        Fallback: NumPy bit-matrix, then pure Python.
        """
        # Fast path: Rust SIMD on the packed byte vectors directly
        if _HAS_SIMD and live:
            return self._bulk_similarity_simd(qbits, live, tier)

        try:
            import numpy as np
        except ImportError:
            pairs = (self._keys, self._ckeys) if tier == "lexical" else (self._sem_keys, self._sem_ckeys)
            out = []
            for i in live:
                best = 0.0
                for keys in pairs:
                    best = max(best, 1.0 - hamming(qbits, keys[i]) / self.dims)
                out.append(best)
            return out

        q = np.unpackbits(np.frombuffer(qbits, dtype=np.uint8))
        kname, cname = ("keys", "ckeys") if tier == "lexical" else ("sem_keys", "sem_ckeys")
        sims_key = 1.0 - (self._bitmatrix(kname) != q).sum(axis=1) / self.dims
        sims_cont = 1.0 - (self._bitmatrix(cname) != q).sum(axis=1) / self.dims
        stacked = np.maximum(sims_key, sims_cont)
        return [float(stacked[i]) for i in live]

    def _bulk_similarity_simd(self, qbits: bytes, live: list[int], tier: str) -> list[float]:
        """Rust SIMD fast path: pack live vectors and compute similarity in one call."""
        key_len = self.dims // 8
        kname, cname = ("keys", "ckeys") if tier == "lexical" else ("sem_keys", "sem_ckeys")
        kvecs = getattr(self, f"_{kname}")
        cvecs = getattr(self, f"_{cname}")

        # Pack and compute similarity in one Rust call (avoids Python round-trip)
        sims_k, sims_c = _simd.pack_and_similarity(qbits, kvecs, cvecs, live, key_len, self.dims)
        return [max(sk, sc) for sk, sc in zip(sims_k, sims_c)]

    def stats(self) -> dict:
        wv = self._write_version
        if self._stats_cache is not None and self._stats_cache[0] == wv:
            cached = self._stats_cache[1]
            return {**cached, "kinds": dict(cached["kinds"])}
        out = {
            "entries": self._alive,
            "dims": self.dims,
            "encoder": f"{self.encoder.seed}:ngram={self.encoder.ngram}",
            "fuzzy_threshold": self.fuzzy_threshold,
            "semantic": self.semantic,
            "semantic_threshold": getattr(self, "semantic_threshold", None),
            "simd": _HAS_SIMD,
            "lz4": _HAS_LZ4,
            "kinds": dict(self._kind_counts),
        }
        self._stats_cache = (wv, out)
        return {**out, "kinds": dict(out["kinds"])}


def _ms(seconds: float) -> float:
    return round(seconds * 1000.0, 3)


def _parse_encoder_name(name: str) -> tuple[str, int]:
    # "bsc-ngram-v1:<seed>:<ngram>"
    parts = name.split(":")
    if len(parts) >= 3:
        try:
            return parts[1], int(parts[2])
        except ValueError:
            pass
    return "sqac-v1", 3
