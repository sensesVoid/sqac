"""Bundled Rust SIMD backend, with a legacy external-package fallback."""

from __future__ import annotations

try:
    from . import _simd as backend
except ImportError:  # pragma: no cover - depends on build platform
    try:
        import sqac_simd as backend  # legacy sqac[simd] installs
    except ImportError:
        backend = None

HAS_SIMD = backend is not None

__all__ = ["backend", "HAS_SIMD"]
