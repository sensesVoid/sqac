"""Zero-dependency BSC encoder for SQAC.

Encodes text into binary spatter-code hypervectors using seeded random
hypervectors per atom (n-gram), bound with position via cyclic permutation,
then bundled (majority vote) into a single D-bit key vector.

Why seeded randomness instead of a learned encoder for v1:
  - Deterministic across machines (vocabulary IS the ABI; seeded atoms make
    any two cartridges with the same fingerprint query-compatible without
    shipping the vocab itself).
  - Zero training. Zero dependencies. The thesis's O(1) non-destructive
    write path depends on this.
  - Optional MiniLM hook exists for the later upgrade where fuzzy recall
    needs true semantic similarity; the store accepts any encoder object
    with .encode(text) -> packed bytes.

VSA operations (Kanerva BSC):
  bind    = XOR            (order-significant, self-inverse)
  permute = cyclic shift   (position, without coordinate-bundling interference
                            -- see thesis Part II 2.5: permutation dominates)
  bundle  = majority vote  (superposition; capacity ~1.2-1.4 bits/dim)

Encoding scheme: character trigram bundling with permutation.
Character n-grams beat word tokens for v1 because:
  - No tokenizer dependency; any language, any code, any typo still matches.
  - Fuzzy robustness: a one-character change perturbs only nearby grams,
    leaving the bundle mostly intact (this IS the fuzzy tolerance).
"""

from __future__ import annotations

import hashlib
import struct
from typing import Optional

# Default dimension. Multiple of 8 for byte packing; 1024 chosen to match
# the FissFus Stage 2 experiments (~94% retrieval at 128 items/bucket).
DEFAULT_DIMS = 1024

NGRAM = 3  # character n-gram size


class BSCEncoder:
    """Deterministic text -> D-bit BSC hypervector encoder."""

    def __init__(self, dims: int = DEFAULT_DIMS, seed: str = "sqac-v1", ngram: int = NGRAM):
        if dims % 8 != 0:
            raise ValueError("dims must be a multiple of 8")
        if dims < 64:
            raise ValueError("dims must be >= 64")
        self.dims = dims
        self.seed = seed
        self.ngram = ngram
        self._atom_cache: dict[str, bytes] = {}

    # ── atom hypervectors ────────────────────────────────────────────────

    def atom(self, token: str) -> bytes:
        """Deterministic pseudo-random D-bit hypervector for an atom.

        SHA-256 based: the same token always yields the same bits on any
        machine, so the vocab dict need not ship with the cartridge.
        """
        cached = self._atom_cache.get(token)
        if cached is not None:
            return cached
        need = self.dims // 8
        out = bytearray()
        counter = 0
        while len(out) < need:
            h = hashlib.sha256(f"{self.seed}:{token}:{counter}".encode()).digest()
            out.extend(h)
            counter += 1
        hv = bytes(out[:need])
        self._atom_cache[token] = hv
        return hv

    # ── VSA primitives on packed bytes ───────────────────────────────────

    def _permute(self, hv: bytes, shift: int) -> bytes:
        """Cyclic byte-level rotation by `shift` bits (bit-exact)."""
        if shift == 0:
            return hv
        nbytes = self.dims // 8
        total = nbytes * 8
        s = shift % total
        byte_shift = s // 8
        bit_shift = s % 8
        if bit_shift == 0:
            return hv[-byte_shift:] + hv[:-byte_shift] if byte_shift else hv
        n = int.from_bytes(hv, "big")
        n = ((n << s) | (n >> (total - s))) & ((1 << total) - 1)
        return n.to_bytes(nbytes, "big")

    def _bundle(self, vecs: list[bytes]) -> bytes:
        """Majority vote via per-byte bit accumulation (8x fewer ops than bit-loop)."""
        nbytes = self.dims // 8
        # counts[i][b] = how many vectors have bit b set in byte i
        masks = (0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01)
        counts = [[0] * 8 for _ in range(nbytes)]
        for v in vecs:
            for i in range(nbytes):
                b = v[i]
                ci = counts[i]
                if b & 0x80:
                    ci[0] += 1
                if b & 0x40:
                    ci[1] += 1
                if b & 0x20:
                    ci[2] += 1
                if b & 0x10:
                    ci[3] += 1
                if b & 0x08:
                    ci[4] += 1
                if b & 0x04:
                    ci[5] += 1
                if b & 0x02:
                    ci[6] += 1
                if b & 0x01:
                    ci[7] += 1
        half = len(vecs) / 2
        out = bytearray(nbytes)
        for i in range(nbytes):
            ci = counts[i]
            byte = 0
            for b in range(8):
                if ci[b] > half:
                    byte |= masks[b]
            out[i] = byte
        return bytes(out)


    # ── public API ───────────────────────────────────────────────────────

    def encode_bits(self, text: str) -> bytes:
        """Encode text into a packed D-bit BSC key vector.

        Bundles the trigram multiset WITHOUT positional permutation:
        position-shifted trigrams only match at identical offsets, which
        crushes recall for paraphrases (measured: aligned-prefix queries
        scored 0.76 while equal-overlap misaligned queries scored ~0.50).
        Order-free bundling is the FissFus Exp C "fusion only" variant and
        was the right operating point for retrieval at this scale.
        """
        text = f"  {text.strip().lower()}  "  # pad so word edges form grams
        if not text.strip():
            return bytes(self.dims // 8)
        grams = [
            text[i : i + self.ngram]
            for i in range(0, max(len(text) - self.ngram + 1, 1))
        ]
        return self._bundle([self.atom(g) for g in grams])

    def encode_bits_batch(self, texts: list[str]) -> list[bytes]:
        return [self.encode_bits(t) for t in texts]

    def similarity(self, a: bytes, b: bytes) -> float:
        """Normalized similarity = 1 - normalized Hamming distance."""
        if hasattr(int, "bit_count"):
            n = 0
            for x, y in zip(a, b):
                n += (x ^ y).bit_count()
        else:  # Python < 3.10 fallback
            n = sum(bin(x ^ y).count("1") for x, y in zip(a, b))
        return 1.0 - n / self.dims
class MiniLMSimHashEncoder:
    """Semantic tier: frozen MiniLM -> seeded projection -> sign bits.

    SimHash property (Goemans-Williamson): P[bit agrees] = 1 - theta/pi,
    where theta is the angle between L2-normalized embeddings. Therefore
    BSC similarity is an affine function of embedding cosine:
      unrelated (cos~0)   -> ~0.50  (same noise floor as the lexical tier)
      paraphrase (cos~0.7) -> ~0.75
      identical            -> 1.00
    Same threshold regime as BSCEncoder, so the store can merge both
    tiers under one confidence scale.

    Projection is deterministic from the seed (vocab-as-ABI property kept:
    cartridges stay portable without shipping embeddings or matrices).
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    _model_cache: dict[str, tuple] = {}  # class-level: (tokenizer, model, torch)

    def __init__(self, dims: int = DEFAULT_DIMS, seed: str = "sqac-sem-v1", model: str = DEFAULT_MODEL):
        if dims % 8 != 0:
            raise ValueError("dims must be a multiple of 8")
        self.dims = dims
        self.seed = seed
        self.model_name = model
        self._proj = self._build_projection()

    def _build_projection(self):
        """Seeded D x 384 rademacher matrix; sha-derived seed = process-stable."""
        import hashlib as _hl
        import numpy as np

        seed_int = int(_hl.sha256(self.seed.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed_int)
        return rng.choice([-1.0, 1.0], size=(384, self.dims)).astype(np.float32)

    def _embed(self, texts: list[str]):
        """L2-normalized mean-pooled MiniLM embeddings (n, 384)."""
        import numpy as np
        import torch
        from transformers import AutoModel, AutoTokenizer

        cls = MiniLMSimHashEncoder
        if self.model_name not in cls._model_cache:
            tok = AutoTokenizer.from_pretrained(self.model_name)
            mdl = AutoModel.from_pretrained(self.model_name)
            mdl.eval()
            cls._model_cache[self.model_name] = (tok, mdl, torch)
        tok, mdl, torch = cls._model_cache[self.model_name]

        enc = tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            out = mdl(**enc).last_hidden_state  # (n, seq, 384)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            emb = torch.nn.functional.normalize(emb, dim=1)
        return emb.numpy().astype(np.float32)

    def encode_bits(self, text: str) -> bytes:
        import numpy as np

        if not text or not text.strip():
            return bytes(self.dims // 8)
        emb = self._embed([text])
        proj = emb @ self._proj  # (1, D)
        bits = (proj > 0).astype(np.uint8)
        return np.packbits(bits, bitorder="big").tobytes()

    def encode_bits_batch(self, texts: list[str]) -> list[bytes]:
        import numpy as np

        texts = [t if t and t.strip() else " " for t in texts]
        emb = self._embed(texts)
        proj = emb @ self._proj
        bits = (proj > 0).astype(np.uint8)
        packed = np.packbits(bits, axis=1, bitorder="big")
        return [row.tobytes() for row in packed]

    def similarity(self, a: bytes, b: bytes) -> float:
        if hasattr(int, "bit_count"):
            n = 0
            for x, y in zip(a, b):
                n += (x ^ y).bit_count()
        else:
            n = sum(bin(x ^ y).count("1") for x, y in zip(a, b))
        return 1.0 - n / self.dims

    @classmethod
    def available(cls) -> bool:
        try:
            import torch, transformers  # noqa: F401
            import numpy  # noqa: F401
            return True
        except ImportError:
            return False

