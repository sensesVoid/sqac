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
import hashlib
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .encoder import BSCEncoder, MiniLMSimHashEncoder
from .static_encoder import DEFAULT_STATIC_MODEL, StaticSimHashEncoder

# Optional Rust SIMD acceleration — 100-300x faster for fuzzy scan
try:
    import sqac_simd as _simd
    _HAS_SIMD = True
except ImportError:
    _simd = None
    _HAS_SIMD = False

from .format import (
    _HAS_LZ4,

    VERSION,
    Cartridge,
    CartridgeHeader,
    Entry,
    FormatError,
    compact_cartridge,
    hamming,
    locked_cartridge,
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
KIND_NAMES = {
    KIND_GENERIC: "generic",
    KIND_FACT: "fact",
    KIND_SKILL: "skill",
    KIND_DOC: "doc",
    KIND_TURN: "turn",
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

    def as_dict(self) -> dict:
        return {
            "content": self.content,
            "confidence": round(self.confidence, 4),
            "source": self.source,
            "mode": self.mode,
            "meta": self.meta,
        }


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
        semantic: bool = False,
        semantic_model: Optional[str] = None,
        cache_size: int = DEFAULT_CACHE_SIZE,
    ):
        self.encoder = encoder or BSCEncoder(dims=dims)
        self.dims = self.encoder.dims
        self.fuzzy_threshold = fuzzy_threshold
        self.semantic_threshold = self.DEFAULT_SEMANTIC_THRESHOLD
        self.semantic = False
        self._sem_encoder = None
        self.format_version = VERSION  # v2 writes raw binary vectors; 1 = legacy hex
        # Query cache: maps (query_hash, tier) -> packed bits
        self._cache_size = cache_size
        self._query_cache: collections.OrderedDict[tuple[str, str], bytes] = (
            collections.OrderedDict()
        )
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
                    raise RuntimeError(
                        "semantic=True needs the static model files (see "
                        "sqac/static_encoder.py) or torch+transformers for MiniLM"
                    )
            self.semantic = True

        self._entries: list[dict[str, Any]] = []  # live entries
        self._exact: dict[str, int] = {}  # normalized key -> entry idx
        self._keys: list[bytes] = []  # packed lexical key vectors
        self._ckeys: list[bytes] = []  # packed lexical content vectors
        self._sem_keys: list[bytes] = []  # packed semantic key vectors
        self._sem_ckeys: list[bytes] = []  # packed semantic content vectors
        self._matrices: dict[str, tuple] = {}  # lazily unpacked (n, D) bit matrices

    def _invalidate_cache(self) -> None:
        """Drop all cached query encodings (call after any write)."""
        self._query_cache.clear()
        self._matrices.clear()  # bit matrices are stale too

    # ── write path ───────────────────────────────────────────────────────

    def _cached_encode(self, text: str, tier: str) -> bytes:
        """Encode text with LRU cache. Invalidated on any write."""
        cache_key = (hashlib.sha256(text.encode()).hexdigest()[:16], tier)
        if cache_key in self._query_cache:
            self._query_cache.move_to_end(cache_key)
            return self._query_cache[cache_key]
        if tier == "semantic" and self._sem_encoder:
            bits = self._sem_encoder.encode_bits(text)
        else:
            bits = self.encoder.encode_bits(text)
        self._query_cache[cache_key] = bits
        if len(self._query_cache) > self._cache_size:
            self._query_cache.popitem(last=False)
        return bits

    def add(
        self,
        content: str,
        key: Optional[str] = None,
        meta: Optional[dict[str, Any]] = None,
        source: str = "user",
        kind: int | str | None = KIND_GENERIC,
    ) -> int:
        """Teach the store one fact. O(1) amortized, non-destructive.

        kind: knowledge kind (KIND_* int or name: "fact"/"skill"/"doc"/
        "turn"). Kinds enable filtered retrieval and kind-aware ranking;
        they do not affect which index tiers an entry participates in.
        """
        content = sanitize_content(content)
        if not content:
            raise ValueError("content must be non-empty")
        key_text = sanitize_key(key) if key is not None else content
        norm = normalize(key_text)
        self._invalidate_cache()
        idx = len(self._entries)
        self._entries.append(
            {
                "content": content,
                "key_norm": norm,
                "meta": meta or {},
                "source": source,
                "kind": resolve_kind(kind),
            }
        )
        self._keys.append(self.encoder.encode_bits(key_text))
        self._ckeys.append(self.encoder.encode_bits(content))
        if self.semantic:
            self._sem_keys.append(self._sem_encoder.encode_bits(key_text))
            self._sem_ckeys.append(self._sem_encoder.encode_bits(content))
        self._exact[norm] = idx
        return idx

    def delete(self, idx: int) -> bool:
        """Tombstone an entry (O(1))."""
        if not (0 <= idx < len(self._entries)):
            return False
        self._entries[idx]["deleted"] = True
        self._exact.pop(self._entries[idx]["key_norm"], None)
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
            self._entries.append(e)
            norm = e["key_norm"]
            idx = len(self._entries) - 1
            key_text = norm  # reconstruct from normalized key
            self._keys.append(self.encoder.encode_bits(norm))
            self._ckeys.append(self.encoder.encode_bits(e["content"]))
            if self.semantic and self._sem_encoder:
                self._sem_keys.append(self._sem_encoder.encode_bits(norm))
                self._sem_ckeys.append(self._sem_encoder.encode_bits(e["content"]))
            if norm:
                self._exact[norm] = idx
        return {"before": before, "after": len(alive), "removed": removed}

    def __len__(self) -> int:
        return sum(1 for e in self._entries if not e.get("deleted"))

    # ── read path ────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 3, kind: int | str | None = None, use_cache: bool = True) -> list[Hit]:
        """Exact-first, fuzzy-fallback retrieval with confidence.

        kind: optional filter (KIND_* int or name). None = all kinds.
        Kinds filter which entries are eligible; they never change which
        tiers run — the exact/lexical/semantic merge is unchanged.
        """
        t0 = time.perf_counter()
        norm = normalize(query)
        want_kind: Optional[int] = None
        if kind is not None:
            want_kind = kind if isinstance(kind, int) else resolve_kind(kind)

        def eligible(i: int) -> bool:
            if self._entries[i].get("deleted"):
                return False
            if want_kind is not None and self._entries[i].get("kind", KIND_GENERIC) != want_kind:
                return False
            return True

        # 1) Exact hit: O(1), confidence 1.0
        idx = self._exact.get(norm)
        if idx is not None and eligible(idx):
            e = self._entries[idx]
            return [
                Hit(
                    content=e["content"],
                    confidence=1.0,
                    source=e["source"],
                    mode="exact",
                    meta={
                        **e["meta"],
                        "kind": KIND_NAMES.get(e.get("kind", KIND_GENERIC), "generic"),
                        "latency_ms": _ms(time.perf_counter() - t0),
                    },
                )
            ]

        # 2) Lexical tier: BSC trigram bundling — typos, word overlap.
        #    XOR+popcount scan against BOTH key and content vectors, best
        #    similarity wins. Numpy bit-matrix fast path (Python stand-in
        #    for the archived Rust engine; the O(n) wall is fundamental
        #    at this tier — thesis Part VI).
        qbits = self._cached_encode(query, "lexical") if use_cache else self.encoder.encode_bits(query)
        results: list[tuple[float, int, str]] = []
        live = [i for i in range(len(self._entries)) if eligible(i)]
        if live:
            sims = self._bulk_similarity(qbits, live, tier="lexical")
            for i, sim in zip(live, sims):
                if sim >= self.fuzzy_threshold:
                    results.append((sim, i, "fuzzy"))
            # 3) Semantic tier: MiniLM SimHash — synonyms/paraphrase with
            #    no lexical overlap ("x86" -> AMD64 rule, "db" -> database).
            #    Same noise floor (~0.5) and confidence scale as lexical.
            if self.semantic:
                qsem = self._cached_encode(query, "semantic") if use_cache else self._sem_encoder.encode_bits(query)
                sem_sims = self._bulk_similarity(qsem, live, tier="semantic")
                for i, sim in zip(live, sem_sims):
                    if sim >= self.semantic_threshold:
                        results.append((sim, i, "semantic"))
        results.sort(reverse=True)
        out = []
        seen_idx: set[int] = set()
        for sim, i, mode in results:
            if i in seen_idx:  # same entry hit by both tiers: keep best
                continue
            seen_idx.add(i)
            e = self._entries[i]
            out.append(
                Hit(
                    content=e["content"],
                    confidence=sim,
                    source=e["source"],
                    mode=mode,
                    meta={
                        **e["meta"],
                        "kind": KIND_NAMES.get(e.get("kind", KIND_GENERIC), "generic"),
                        "latency_ms": _ms(time.perf_counter() - t0),
                    },
                )
            )
            if len(out) >= top_k:
                break
        return out

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
        """Write the store as a .sqac cartridge (atomic replace).

        v2 layout: auxiliary packed vectors (content, semantic key/content)
        go into a per-entry raw binary block — ~2x smaller than the v1
        hex-in-JSON encoding. Use format_version=1 to emit the legacy
        layout (readable by pre-v2 runtimes).
        """
        key_len = self.dims // 8
        entries = []
        for idx, (e, kbits, ckbits) in enumerate(zip(self._entries, self._keys, self._ckeys)):
            payload = {
                "content": e["content"],
                "key_norm": e["key_norm"],
                "source": e["source"],
                "kind": KIND_NAMES.get(e.get("kind", KIND_GENERIC), "generic"),
                **({"meta": e["meta"]} if e["meta"] else {}),
            }
            if self.format_version >= 2:
                binvec = bytes(ckbits)
                if self.semantic:
                    binvec += bytes(self._sem_keys[idx]) + bytes(self._sem_ckeys[idx])
            else:
                binvec = b""
                payload["content_key"] = ckbits.hex()
                if self.semantic:
                    payload["sem_key"] = self._sem_keys[idx].hex()
                    payload["sem_content_key"] = self._sem_ckeys[idx].hex()
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
    def load(cls, path: str | Path, fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD) -> "SqacStore":
        """Load a cartridge. Refuses incompatible encoder/fingerprint.

        fuzzy_threshold defaults to the same value as the constructor
        (DEFAULT_FUZZY_THRESHOLD) so a cartridge behaves identically
        before and after a save/load roundtrip.
        """
        cart = read_cartridge(path)
        h = cart.header
        seed, ngram = _parse_encoder_name(h.encoder_name)
        enc = BSCEncoder(dims=h.dims, seed=seed, ngram=ngram)
        expected = h.compute_fingerprint({})  # seeded vocab is empty by design
        if h.vocab_fingerprint and h.vocab_fingerprint != expected:
            raise FormatError("vocab fingerprint mismatch: cartridge incompatible")
        store = cls(
            encoder=enc,
            dims=h.dims,
            fuzzy_threshold=fuzzy_threshold,
            semantic="|sem-v1:" in h.encoder_name,
        )
        store._description = h.description
        if store.semantic:
            store._sem_encoder = _make_sem_encoder(
                h.encoder_name.split("|sem-v1:", 1)[1].split(":", 1)[0], h.dims
            )
        for e in cart.entries:
            norm = e.payload.get("key_norm", "")
            kind = e.kind if e.kind else resolve_kind(e.payload.get("kind"))
            idx = len(store._entries)
            store._entries.append(
                {
                    "content": e.payload["content"],
                    "key_norm": norm,
                    "meta": e.payload.get("meta", {}),
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
        """Rust SIMD fast path: pack live vectors into a contiguous matrix
        and call sqac_simd.bulk_similarity."""
        key_len = self.dims // 8
        kname, cname = ("keys", "ckeys") if tier == "lexical" else ("sem_keys", "sem_ckeys")
        kvecs = getattr(self, f"_{kname}")
        cvecs = getattr(self, f"_{cname}")

        # Pack live vectors into contiguous buffers
        n = len(live)
        kbuf = bytearray(n * key_len)
        cbuf = bytearray(n * key_len)
        for j, i in enumerate(live):
            offset = j * key_len
            kbuf[offset:offset + key_len] = kvecs[i]
            cbuf[offset:offset + key_len] = cvecs[i]

        sims_k = _simd.bulk_similarity(qbits, bytes(kbuf), n, self.dims)
        sims_c = _simd.bulk_similarity(qbits, bytes(cbuf), n, self.dims)
        return [max(sk, sc) for sk, sc in zip(sims_k, sims_c)]

    def stats(self) -> dict:
        kinds: dict[str, int] = {}
        for e in self._entries:
            if e.get("deleted"):
                continue
            k = KIND_NAMES.get(e.get("kind", KIND_GENERIC), "generic")
            kinds[k] = kinds.get(k, 0) + 1
        return {
            "entries": len(self),
            "dims": self.dims,
            "encoder": f"{self.encoder.seed}:ngram={self.encoder.ngram}",
            "fuzzy_threshold": self.fuzzy_threshold,
            "semantic": self.semantic,
            "semantic_threshold": getattr(self, "semantic_threshold", None),
            "simd": _HAS_SIMD,
            "lz4": _HAS_LZ4,
            "kinds": kinds,
        }


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
