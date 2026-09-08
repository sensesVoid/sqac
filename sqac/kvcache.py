"""KV-cache footprint estimator — sparse semantic recall vs context stuffing.

SQAC does not touch a model's KV *tensors* (that is the serving layer's job:
vLLM/SGLang + LMCache/Mooncake). What it *can* provably reduce is the number of
tokens that enter context and therefore the KV cache the model must materialize.
In the agentic regime, stuffing the full accumulated memory into the window is the
"5M-token" fantasy; sparse recall injects only the relevant top-k.

This module is the canonical, model-agnostic estimator. It boxes the two inputs a
user actually controls — how many tokens of memory exist, and how many the recall
injects — and derives KV bytes + savings from a per-token constant.

Formula (canonical, linear-in-tokens):

    KV_bytes/token (per sequence) = n_layers × n_kv_heads × 2 × head_dim × bytes/value
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional

# Bytes per value in supported precisions.
_BYTES_PER_VALUE = {"fp16": 2, "bf16": 2, "fp32": 4, "int8": 1, "fp8": 1}

# Known model KV geometry: n_layers, n_kv_heads, head_dim.
# GQA note: KV cache uses n_kv_heads (not n_heads).
_KV_GEOMETRY: Dict[str, tuple[int, int, int]] = {
    "llama-3.1-8b": (32, 8, 128),
    "llama-3.1-70b": (80, 8, 128),
    "llama-3.1-405b": (126, 8, 128),
    "llama-3-1b": (16, 8, 64),
    "qwen2.5-7b": (28, 4, 128),
    "qwen2.5-32b": (64, 8, 128),
    "mistral-7b": (32, 8, 128),
    "mixtral-8x7b": (32, 8, 128),
}


@dataclass(frozen=True)
class KVEstimate:
    """KV footprint of a context strategy for one model at one Bertie size."""

    model: str
    kv_bytes_per_token: int
    stuffing_tokens: int
    stuffing_kv_bytes: int
    sparse_tokens: int
    sparse_kv_bytes: int
    reduction_x: float
    savings_pct: float

    @property
    def reduction(self) -> str:
        return f"{self.reduction_x:.1f}× ({self.savings_pct:.1f}%)"


def kv_bytes_per_token(
    n_layers: int, n_kv_heads: int, head_dim: int, precision: str = "fp16"
) -> int:
    """KV cache bytes for one token of a sequence (GQA-aware)."""
    if precision not in _BYTES_PER_VALUE:
        raise ValueError(f"unknown precision {precision!r}; use {sorted(_BYTES_PER_VALUE)}")
    return n_layers * n_kv_heads * 2 * head_dim * _BYTES_PER_VALUE[precision]


def model_kv_bytes_per_token(model: str, precision: str = "fp16") -> int:
    """KV bytes/token for a known model id (raises KeyError for unknown ids)."""
    layers, kv_heads, head_dim = _KV_GEOMETRY[model]
    return kv_bytes_per_token(layers, kv_heads, head_dim, precision)


def known_models() -> tuple[str, ...]:
    return tuple(sorted(_KV_GEOMETRY))


def estimate_sparse_recall(
    bucket_tokens: int,
    recall_tokens: int,
    model: str = "llama-3.1-8b",
    precision: str = "fp16",
) -> KVEstimate:
    """KV footprint of stuffing the whole bucket vs injecting recall_tokens.

    bucket_tokens : total tokens in the offloaded memory (the "5M-token" bucket).
    recall_tokens : tokens actually injected into the prompt (top-k ~= k×45 avg).
    """
    per_token = model_kv_bytes_per_token(model, precision)
    stuffing = bucket_tokens * per_token
    sparse = recall_tokens * per_token
    reduction = stuffing / sparse if sparse else float("inf")
    savings = (1 - sparse / stuffing) * 100 if stuffing else 0.0
    return KVEstimate(
        model=model,
        kv_bytes_per_token=per_token,
        stuffing_tokens=bucket_tokens,
        stuffing_kv_bytes=stuffing,
        sparse_tokens=recall_tokens,
        sparse_kv_bytes=sparse,
        reduction_x=reduction,
        savings_pct=savings,
    )


def estimate_sweep(
    bucket_tokens: int,
    recall_budgets: tuple[int, ...] = (1, 3, 10),
    tokens_per_exchange: int = 45,
    model: str = "llama-3.1-8b",
    precision: str = "fp16",
) -> list[KVEstimate]:
    """KV-reduction curve across recall budgets (each = k × tokens/exchange)."""
    out = []
    for k in recall_budgets:
        out.append(
            estimate_sparse_recall(
                bucket_tokens,
                k * tokens_per_exchange,
                model=model,
                precision=precision,
            )
        )
    return out


def table(metric: Literal["mb", "gb"] = "mb") -> str:
    """Human-readable markdown table over known models at 1M token bucket / k=3."""

    def fmt(b: int) -> str:
        v = b / (1024**2) if metric == "mb" else b / (1024**3)
        unit = "MiB" if metric == "mb" else "GiB"
        return f"{v:,.1f} {unit}"

    lines = [
        "| model | KV/token | stuffing (1M tok) | sparse (k=3) | reduction |",
        "|---|---|---|---|---|",
    ]
    for model in known_models():
        est = estimate_sparse_recall(1_000_000, 3 * 45, model=model)
        lines.append(
            f"| {model} | {fmt(est.kv_bytes_per_token)}/tok | "
            f"{fmt(est.stuffing_kv_bytes)} | {fmt(est.sparse_kv_bytes)} | "
            f"{est.reduction} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="KV footprint estimator (canonical).")
    p.add_argument("--bucket", type=int, default=1_000_000, help="total memory tokens")
    p.add_argument("--k", type=int, nargs="+", default=[1, 3, 10], help="recall budgets (exchanges)")
    p.add_argument("--tpe", type=int, default=45, help="avg tokens per exchange")
    p.add_argument("--model", choices=list(_KV_GEOMETRY), default="llama-3.1-8b")
    p.add_argument("--precision", choices=list(_BYTES_PER_VALUE), default="fp16")
    a = p.parse_args()

    est = estimate_sweep(a.bucket, tuple(a.k), a.tpe, a.model, a.precision)
    print(f"model={a.model} ({a.precision})   KV/token={model_kv_bytes_per_token(a.model, a.precision):,} B")
    print(f"bucket={a.bucket:,} tokens  tpe={a.tpe}")
    print(f"{'k':>3} {'recall tok':>12} {'stuffing':>14} {'sparse':>14} {'reduction':>12}")
    for e in est:
        print(f"{e.sparse_tokens//a.tpe:>3} {e.sparse_tokens:>12,} "
              f"{e.stuffing_kv_bytes/2**20:>11,.1f}MiB {e.sparse_kv_bytes/2**20:>11,.1f}MiB "
              f"{e.reduction_x:>10.1f}x")
