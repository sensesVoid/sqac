#!/usr/bin/env python3
"""PoC: can we replace the torch+MiniLM semantic tier with a pure-numpy
int8-quantized STATIC embedding model (no torch, no onnxruntime)?

Static embedding = token embedding lookup + mean pooling. That's a matrix
multiply with ones — numpy does it natively. We quantize the embedding
matrix to int8 ourselves (per-row asymmetric) and keep the SimHash
projection identical to the existing MiniLMSimHashEncoder.

Validation targets (from our earlier calibration):
  - random floor               ~ 0.49-0.51
  - paraphrase pair            ~ 0.75-0.80
  - x86 -> ARM64 rule          >= 0.60  (the semantic catch)
  - discriminative miss        < 0.60   (no leakage between unrelated keys)

Run: python tests/poc_quant_encoder.py
"""

from __future__ import annotations

import json
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_DIR = "sqac/models/static-retrieval"
ST_DIR = os.path.join(MODEL_DIR, "model.safetensors")
TOK_PATH = os.path.join(MODEL_DIR, "tokenizer.json")


# ── safetensors parsing (no deps) ────────────────────────────────────────
def load_safetensors_matrix(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        buf = f.read()
    # find the embedding matrix key
    emb_key = None
    for k, v in header.items():
        if k == "__metadata__":
            continue
        if isinstance(v, dict) and v.get("dtype") in ("F32", "F16", "BF16") and len(v.get("shape", [])) == 2:
            emb_key = k
            break
    if emb_key is None:
        raise RuntimeError(f"no 2-D float matrix in {path}; keys={list(header)}")
    info = header[emb_key]
    shape = info["shape"]
    off = info["data_offsets"][0]
    count = int(np.prod(shape))
    if info["dtype"] == "F32":
        return np.frombuffer(buf, np.float32, count, off).reshape(shape).copy()
    if info["dtype"] == "F16":
        return np.frombuffer(buf, np.float16, count, off).reshape(shape).copy()
    raise RuntimeError(f"unsupported dtype {info['dtype']}")


# ── uint8 quantization (per-row asymmetric) ──────────────────────────────
def quantize_int8(W: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (q_u8 [V,D], scale fp32 [V], min fp32 [V]) with W ≈ q*scale + min per row.

    Storage must be uint8: codes span 0..255 and int8 would overflow.
    """
    wmin = W.min(axis=1, keepdims=True)
    wmax = W.max(axis=1, keepdims=True)
    scale = (wmax - wmin) / 255.0
    scale = np.where(scale == 0, 1e-8, scale)
    q = np.round((W - wmin) / scale).astype(np.uint8)
    return q, scale.squeeze(1).astype(np.float32), wmin.squeeze(1).astype(np.float32)


class QuantizedStaticEmbedder:
    """Pure-numpy static embeddings from an int8-quantized embedding matrix."""

    def __init__(self, st_path: str = ST_DIR, tok_path: str = TOK_PATH, quantize: bool = True):
        W = load_safetensors_matrix(st_path)
        self.vocab_size, self.dim = W.shape
        if quantize:
            self.q, self.scale, self.wmin = quantize_int8(W)
            self.W = None  # free the fp32 copy
        else:
            self.W = W
        # tokenizer
        with open(tok_path) as f:
            tok = json.load(f)
        m = tok["model"]
        assert m["type"] == "WordPiece", f"only WordPiece supported, got {m['type']}"
        self.vocab: dict[str, int] = m["vocab"]
        self.unk = self.vocab["[UNK]"]
        self.cls = self.vocab["[CLS]"]
        self.sep = self.vocab["[SEP]"]
        self.max_len = 256
        # lowercase basic normalization (bert-base-uncased behavior)
        self._cache: dict[str, list[int]] = {}

    def _dequant_rows(self, ids: np.ndarray) -> np.ndarray:
        if self.W is not None:
            return self.W[ids]
        rows = self.q[ids].astype(np.float32)
        return rows * self.scale[ids, None] + self.wmin[ids, None]

    def _wordpiece(self, word: str) -> list[int]:
        if word in self._cache:
            return self._cache[word]
        ids: list[int] = []
        prefix = ""
        pieces: list[str] = []
        start = 0
        while start < len(word):
            end = len(word)
            piece = None
            while start < end:
                sub = prefix + word[start:end]
                if sub in self.vocab:
                    piece = sub
                    break
                end -= 1
            if piece is None:
                ids.append(self.unk)
                pieces = []
                break
            pieces.append(piece)
            ids.append(self.vocab[piece])
            prefix = "##"
            start = end
        self._cache[word] = ids
        return ids

    def _tokenize(self, text: str) -> list[int]:
        words = "".join(c.lower() if c.isalpha() or c == "-" else " " for c in text).split()
        ids = [self.cls]
        for w in words:
            ids.extend(self._wordpiece(w))
        ids.append(self.sep)
        return ids[: self.max_len]

    def encode(self, texts: list[str]) -> np.ndarray:
        all_ids = [self._tokenize(t) for t in texts]
        maxlen = max(len(i) for i in all_ids)
        ids = np.full((len(texts), maxlen), self.sep, dtype=np.int64)
        mask = np.zeros((len(texts), maxlen), dtype=np.float32)
        for r, seq in enumerate(all_ids):
            ids[r, : len(seq)] = seq
            mask[r, : len(seq)] = 1.0
        emb = self._dequant_rows(ids)  # (B, T, D)
        summed = (emb * mask[:, :, None]).sum(axis=1)
        counts = np.maximum(mask.sum(axis=1, keepdims=True), 1.0)
        mean = summed / counts
        norm = np.linalg.norm(mean, axis=1, keepdims=True)
        return mean / np.maximum(norm, 1e-9)


def simhash_bits(emb: np.ndarray, proj: np.ndarray) -> np.ndarray:
    bits = (emb @ proj > 0).astype(np.uint8)
    packed = np.zeros((emb.shape[0], bits.shape[1] // 8), dtype=np.uint8)
    for i in range(8):
        packed |= bits[:, i::8] << i
    return packed


def bits_to_np(packed: np.ndarray) -> bytes:
    return packed[0].tobytes()


def hamming_sim(a: bytes, b: bytes) -> float:
    if hasattr(int, "bit_count"):
        x = int.from_bytes(a, "little") ^ int.from_bytes(b, "little")
        return 1.0 - x.bit_count() / (len(a) * 8)
    return 1.0 - bin(int.from_bytes(a, "little") ^ int.from_bytes(b, "little")).count("1") / (len(a) * 8)


def build_projection(seed: str, emb_dim: int, out_bits: int = 1024) -> np.ndarray:
    """Seeded gaussian projection (emb_dim -> out_bits), PoC-internal."""
    import hashlib

    seed_int = int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed_int)
    return rng.standard_normal((emb_dim, out_bits), dtype=np.float32) / np.sqrt(emb_dim)


# ── evaluation: SQAC's own calibration tasks ─────────────────────────────
CALIB_PAIRS = [
    ("paraphrase", "If a system component fails, the backup takes over seamlessly",
     "When a part of the system goes down, the standby system continues operation without interruption"),
    ("x86-arm64", "all production deploys must target ARM64 architecture",
     "the build pipeline only supports x86 images"),
    ("db-repo", "never access the database directly from business logic",
     "the repository layer owns all database access"),
    ("middleware", "all API requests pass through authentication middleware",
     "the auth middleware validates tokens before routing"),
    ("floor-a", "quantum flux capacitor illuminates purple elephants",  # unrelated pair 1
     "our deployment policy requires two approvers"),
    ("floor-b", "the chef garnished the risotto with truffle oil",  # unrelated pair 2
     "kernel panics often indicate driver incompatibility"),
]

DISCRIM_PAIRS = [
    ("our deployment policy requires two approvers", "team atlas maintains the payments service"),
    ("the jug puzzle needs state space search", "forty two is the answer to everything"),
]


def main() -> None:
    print("loading model (this allocates ~125MB fp32, then frees)...")
    t0 = time.time()
    for quantize in (True, False):
        emb = QuantizedStaticEmbedder(quantize=quantize)
        proj = build_projection("sqac-sem-v1", 1024)
        label = "INT8 " if quantize else "FP32 "
        print(f"\n=== {label} (load {time.time()-t0:.1f}s) ===")

        # calibration similarities
        print("pair similarities (cosine of embeddings | hamming of BSC bits):")
        for name, a, b in CALIB_PAIRS:
            e = emb.encode([a, b])
            cos = float(e[0] @ e[1])
            bits = simhash_bits(e, proj)
            ham = hamming_sim(bits[0].tobytes(), bits[1].tobytes())
            print(f"  {name:12}  cos={cos:.3f}  bits={ham:.3f}")

        # discriminative check: query vs two keys, closer to its own
        print("discrimination (cos to own key vs unrelated key):")
        for q, own in [("deployment approval count for production releases", "our deployment policy requires two approvers"),
                       ("how to solve the water jug puzzle", "the jug puzzle needs state space search")]:
            e = emb.encode([q, own])
            print(f"  cos={float(e[0] @ e[1]):.3f}  q={q[:40]!r}")

        # latency
        texts = ["all production deploys must target ARM64 architecture"] * 32
        t0 = time.time()
        emb.encode(texts)
        dt = time.time() - t0
        print(f"latency: {dt/32*1000:.2f} ms/sentence ({32} in {dt:.3f}s)")

        # memory footprint
        if quantize:
            total = emb.q.nbytes + emb.scale.nbytes + emb.wmin.nbytes + os.path.getsize(TOK_PATH)
        else:
            total = emb.W.nbytes + os.path.getsize(TOK_PATH)
        print(f"runtime memory: {total/1e6:.1f} MB")
        t0 = time.time()


if __name__ == "__main__":
    main()
