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

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .encoder import BSCEncoder, MiniLMSimHashEncoder
from .static_encoder import DEFAULT_STATIC_MODEL, StaticSimHashEncoder
from .format import (
    VERSION,
    Cartridge,
    CartridgeHeader,
    Entry,
    FormatError,
    hamming,
    read_cartridge,
    write_cartridge,
)


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
    """

    # Both VSA tiers share the same noise floor (~0.5) and threshold
    # regime: lexical true matches >= 0.6, semantic paraphrases ~0.7+.
    DEFAULT_FUZZY_THRESHOLD = 0.60
    DEFAULT_SEMANTIC_THRESHOLD = 0.60  # measured: x86→ARM64 = 0.65, floor = 0.49

    def __init__(
        self,
        encoder: Optional[BSCEncoder | MiniLMSimHashEncoder] = None,
        dims: int = 1024,
        fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
        semantic: bool = False,
        semantic_model: Optional[str] = None,
    ):
        self.encoder = encoder or BSCEncoder(dims=dims)
        self.dims = self.encoder.dims
        self.fuzzy_threshold = fuzzy_threshold
        self.semantic_threshold = self.DEFAULT_SEMANTIC_THRESHOLD
        self.semantic = False
        self._sem_encoder = None
        self.format_version = VERSION  # v2 writes raw binary vectors; 1 = legacy hex
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

    # ── write path ───────────────────────────────────────────────────────

    def add(
        self,
        content: str,
        key: Optional[str] = None,
        meta: Optional[dict[str, Any]] = None,
        source: str = "user",
    ) -> int:
        """Teach the store one fact. O(1) amortized, non-destructive."""
        content = content.strip()
        if not content:
            raise ValueError("content must be non-empty")
        key_text = key if key is not None else content
        norm = normalize(key_text)
        idx = len(self._entries)
        self._entries.append(
            {"content": content, "key_norm": norm, "meta": meta or {}, "source": source}
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
        return True

    def __len__(self) -> int:
        return sum(1 for e in self._entries if not e.get("deleted"))

    # ── read path ────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 3) -> list[Hit]:
        """Exact-first, fuzzy-fallback retrieval with confidence."""
        t0 = time.perf_counter()
        norm = normalize(query)

        # 1) Exact hit: O(1), confidence 1.0
        idx = self._exact.get(norm)
        if idx is not None and not self._entries[idx].get("deleted"):
            e = self._entries[idx]
            return [
                Hit(
                    content=e["content"],
                    confidence=1.0,
                    source=e["source"],
                    mode="exact",
                    meta={**e["meta"], "latency_ms": _ms(time.perf_counter() - t0)},
                )
            ]

        # 2) Lexical tier: BSC trigram bundling — typos, word overlap.
        #    XOR+popcount scan against BOTH key and content vectors, best
        #    similarity wins. Numpy bit-matrix fast path (Python stand-in
        #    for the archived Rust engine; the O(n) wall is fundamental
        #    at this tier — thesis Part VI).
        qbits = self.encoder.encode_bits(query)
        results: list[tuple[float, int, str]] = []
        live = [i for i in range(len(self._entries)) if not self._entries[i].get("deleted")]
        if live:
            sims = self._bulk_similarity(qbits, live, tier="lexical")
            for i, sim in zip(live, sims):
                if sim >= self.fuzzy_threshold:
                    results.append((sim, i, "fuzzy"))
            # 3) Semantic tier: MiniLM SimHash — synonyms/paraphrase with
            #    no lexical overlap ("x86" -> AMD64 rule, "db" -> database).
            #    Same noise floor (~0.5) and confidence scale as lexical.
            if self.semantic:
                qsem = self._sem_encoder.encode_bits(query)
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
                    meta={**e["meta"], "latency_ms": _ms(time.perf_counter() - t0)},
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
    def load(cls, path: str | Path, fuzzy_threshold: float = 0.70) -> "SqacStore":
        """Load a cartridge. Refuses incompatible encoder/fingerprint."""
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
        if store.semantic:
            store._sem_encoder = _make_sem_encoder(
                h.encoder_name.split("|sem-v1:", 1)[1].split(":", 1)[0], h.dims
            )
        for e in cart.entries:
            norm = e.payload.get("key_norm", "")
            idx = len(store._entries)
            store._entries.append(
                {
                    "content": e.payload["content"],
                    "key_norm": norm,
                    "meta": e.payload.get("meta", {}),
                    "source": e.payload.get("source", "user"),
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
        """Vectorized 1 - hamming/D against key and content matrices."""
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

    def stats(self) -> dict:
        import os

        return {
            "entries": len(self),
            "dims": self.dims,
            "encoder": f"{self.encoder.seed}:ngram={self.encoder.ngram}",
            "fuzzy_threshold": self.fuzzy_threshold,
            "semantic": self.semantic,
            "semantic_threshold": getattr(self, "semantic_threshold", None),
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
