"""Quantized static-embedding encoder — the light semantic tier.

Replaces the torch+MiniLM stack with a pure-numpy int8 static model
(potion-base-8M). Rationale (all measured in this repo):

  - Model: 7.6MB int8 (vs 90MB fp32 MiniLM) — a token embedding table +
    tokenizer, nothing else. No attention, no torch, no onnxruntime.
  - Latency: ~0.1ms/sentence CPU (vs 10-20ms MiniLM).
  - Quality on SQAC's calibration tasks (tests/poc_quant_encoder.py):
    x86→ARM64 0.64 (MiniLM 0.65), db→repository 0.68 (MiniLM 0.56 —
    potion CATCHES the case MiniLM misses), paraphrase 0.66, floor ~0.5.

SimHash property (Goemans-Williamson): P[bit agrees] = 1 - theta/pi for
L2-normalized embeddings under a random sign projection, so BSC similarity
is affine in embedding cosine — same noise floor (~0.5) and threshold
regime as the lexical tier, one confidence scale across the store.

The embedding matrix is quantized per-row asymmetrically to uint8 codes
(0..255; int8 storage would overflow) with fp32 scale/min. Measured
similarity drift vs fp32: <0.001 on every calibration pair.

Model files are NOT bundled with the package (size). Point
SQAC_STATIC_MODEL_DIR at a directory containing model.safetensors +
tokenizer.json (default: ./sqac/models/potion-8m, populate with
tests/poc_quant_encoder.py or hf_hub_download of minishlab/potion-base-8M).
"""

from __future__ import annotations

import json
import os
import struct

from .encoder import DEFAULT_DIMS

DEFAULT_STATIC_MODEL = "sqac/models/potion-8m"
STATIC_MODEL_NAME = "static-potion-8m-int8"


def _load_safetensors_matrix(path: str):
    """Parse a safetensors file and return its 2-D float matrix (no deps)."""
    import numpy as np

    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        buf = f.read()
    emb_key = None
    for k, v in header.items():
        if k == "__metadata__":
            continue
        if isinstance(v, dict) and v.get("dtype") in ("F32", "F16", "BF16") and len(v.get("shape", [])) == 2:
            emb_key = k
            break
    if emb_key is None:
        raise RuntimeError(f"no 2-D float matrix in {path}")
    info = header[emb_key]
    shape = info["shape"]
    off = info["data_offsets"][0]
    count = int(np.prod(shape))
    if info["dtype"] == "F32":
        return np.frombuffer(buf, np.float32, count, off).reshape(shape).copy()
    if info["dtype"] == "F16":
        return np.frombuffer(buf, np.float16, count, off).reshape(shape).copy()
    if info["dtype"] == "BF16":
        raw = np.frombuffer(buf, np.uint16, count, off).reshape(shape).astype(np.uint32) << 16
        return raw.view(np.float32).copy()
    raise RuntimeError(f"unsupported dtype {info['dtype']}")


def _quantize_uint8(W):
    """Per-row asymmetric quantization. Returns (codes u8, scale f32, min f32)."""
    import numpy as np

    wmin = W.min(axis=1, keepdims=True)
    wmax = W.max(axis=1, keepdims=True)
    scale = (wmax - wmin) / 255.0
    scale = np.where(scale == 0, 1e-8, scale)
    q = np.round((W - wmin) / scale).astype(np.uint8)
    return q, scale.squeeze(1).astype(np.float32), wmin.squeeze(1).astype(np.float32)


class StaticSimHashEncoder:
    """Semantic tier: int8 static embeddings -> seeded projection -> sign bits.

    Drop-in for MiniLMSimHashEncoder: same encode_bits / encode_bits_batch /
    similarity interface, same threshold regime. Pure numpy at inference.
    """

    DEFAULT_MODEL_DIR = DEFAULT_STATIC_MODEL
    _weight_cache: dict[str, tuple] = {}  # class-level: (codes, scale, wmin, vocab, specials)

    def __init__(self, dims: int = DEFAULT_DIMS, seed: str = "sqac-sem-static-v1",
                 model_dir: str = DEFAULT_STATIC_MODEL):
        if dims % 8 != 0:
            raise ValueError("dims must be a multiple of 8")
        self.dims = dims
        self.seed = seed
        self.model_name = STATIC_MODEL_NAME
        self.model_dir = model_dir
        self._weights = self._load_weights(model_dir)
        self._proj = self._build_projection()
        self._tok_cache: dict[str, list[int]] = {}

    # ── weights ──────────────────────────────────────────────────────────

    def _load_weights(self, model_dir: str):
        import numpy as np

        cache = StaticSimHashEncoder._weight_cache
        if model_dir in cache:
            return cache[model_dir]
        npz = os.path.join(model_dir, "potion-8m-int8.npz")
        st = os.path.join(model_dir, "model.safetensors")
        tok = os.path.join(model_dir, "tokenizer.json")
        if os.path.exists(npz):
            # preferred: pre-quantized artifact (shipped) — instant load
            z = np.load(npz)
            codes = z["codes"]
            scale = z["scale"]
            wmin = z["wmin"]
            vocab = json.loads(str(z["vocab"]))
        elif os.path.exists(st) and os.path.exists(tok):
            W = _load_safetensors_matrix(st)
            codes, scale, wmin = _quantize_uint8(W)
            del W
            with open(tok) as f:
                m = json.load(f)["model"]
            if m["type"] != "WordPiece":
                raise RuntimeError(f"only WordPiece tokenizers supported, got {m['type']}")
            vocab = m["vocab"]
        else:
            raise RuntimeError(
                f"static semantic model not found in {model_dir!r}. "
                "Expected potion-8m-int8.npz or model.safetensors+tokenizer.json "
                "(download minishlab/potion-base-8M) or set SQAC_STATIC_MODEL_DIR."
            )
        specials = {k: vocab[k] for k in ("[UNK]", "[CLS]", "[SEP]") if k in vocab}
        cache[model_dir] = (codes, scale, wmin, vocab, specials)
        return cache[model_dir]

    def _build_projection(self):
        """Seeded emb_dim x D Rademacher matrix; sha-derived seed = process-stable."""
        import hashlib as _hl
        import numpy as np

        _, _, _, _, _ = self._weights  # ensures emb dim known via load
        emb_dim = self._weights[0].shape[1]
        seed_int = int(_hl.sha256(self.seed.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed_int)
        return rng.choice([-1.0, 1.0], size=(emb_dim, self.dims)).astype(np.float32)

    # ── tokenization (bert-uncased-style WordPiece) ──────────────────────

    def _wordpiece(self, word: str, vocab, unk: int) -> list[int]:
        hit = self._tok_cache.get(word)
        if hit is not None:
            return hit
        ids: list[int] = []
        prefix = ""
        start = 0
        while start < len(word):
            end = len(word)
            piece = None
            while start < end:
                sub = prefix + word[start:end]
                if sub in vocab:
                    piece = sub
                    break
                end -= 1
            if piece is None:
                ids.append(unk)
                prefix = ""
                break
            ids.append(vocab[piece])
            prefix = "##"
            start = end
        self._tok_cache[word] = ids
        return ids

    def _tokenize(self, text: str, vocab, unk: int, cls_id: int, sep_id: int) -> list[int]:
        words = "".join(c.lower() if (c.isalnum() or c in "-_") else " " for c in text).split()
        ids = [cls_id] if cls_id >= 0 else []
        for w in words:
            ids.extend(self._wordpiece(w, vocab, unk))
        if sep_id >= 0:
            ids.append(sep_id)
        return ids[:256]

    # ── embedding ────────────────────────────────────────────────────────

    def _embed(self, texts: list[str]):
        import numpy as np

        codes, scale, wmin, vocab, specials = self._weights
        unk = specials.get("[UNK]", 0)
        cls_id = specials.get("[CLS]", -1)
        sep_id = specials.get("[SEP]", -1)
        all_ids = [self._tokenize(t, vocab, unk, cls_id, sep_id) or [unk] for t in texts]
        maxlen = max(len(s) for s in all_ids)
        ids = np.full((len(texts), maxlen), sep_id if sep_id >= 0 else 0, dtype=np.int64)
        mask = np.zeros((len(texts), maxlen), dtype=np.float32)
        for r, seq in enumerate(all_ids):
            ids[r, : len(seq)] = seq
            mask[r, : len(seq)] = 1.0
        rows = codes[ids].astype(np.float32)
        emb = rows * scale[ids, None] + wmin[ids, None]
        summed = (emb * mask[:, :, None]).sum(axis=1)
        mean = summed / np.maximum(mask.sum(axis=1, keepdims=True), 1.0)
        norm = np.linalg.norm(mean, axis=1, keepdims=True)
        return mean / np.maximum(norm, 1e-9)

    # ── public API (mirrors MiniLMSimHashEncoder) ────────────────────────

    def encode_bits(self, text: str) -> bytes:
        import numpy as np

        if not text or not text.strip():
            return bytes(self.dims // 8)
        emb = self._embed([text])
        proj = emb @ self._proj
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
    def available(cls, model_dir: str = DEFAULT_STATIC_MODEL) -> bool:
        try:
            import numpy  # noqa: F401
        except ImportError:
            return False
        d = os.environ.get("SQAC_STATIC_MODEL_DIR", model_dir)
        return (
            os.path.exists(os.path.join(d, "potion-8m-int8.npz"))
            or (
                os.path.exists(os.path.join(d, "model.safetensors"))
                and os.path.exists(os.path.join(d, "tokenizer.json"))
            )
        )


def build_npz(model_dir: str = DEFAULT_STATIC_MODEL, out_path: str | None = None) -> str:
    """One-time conversion: safetensors+tokenizer -> pre-quantized .npz artifact.

    The npz is the shippable form (int8 codes + fp32 scale/min + vocab, ~8MB);
    it loads instantly and skips the runtime quantize step.
    """
    import numpy as np

    st = os.path.join(model_dir, "model.safetensors")
    tok = os.path.join(model_dir, "tokenizer.json")
    out = out_path or os.path.join(model_dir, "potion-8m-int8.npz")
    W = _load_safetensors_matrix(st)
    codes, scale, wmin = _quantize_uint8(W)
    with open(tok) as f:
        vocab = json.load(f)["model"]["vocab"]
    np.savez(
        out,
        codes=codes,
        scale=scale,
        wmin=wmin,
        vocab=np.array(json.dumps(vocab)),
    )
    return out
