"""SQAC — hot-swappable VSA memory cartridges.

Public API:
    SqacStore          — hybrid exact + fuzzy memory store
    ContextOffloader   — ephemeral turn bucket (observe/save/recall)
    CartridgeRack      — named cartridges + kind-based routing
    BSCEncoder         — zero-dependency binary spatter-code encoder
    read/write         — .sqac binary format v1
    extract_project_units / build_store / track_once / track
                        — project auto-build and realtime tracking
"""

from .autobuild import (
    build_store,
    extract_project_units,
    track,
    track_once,
    _sync_rack,
)
from .continuity import ContinuityStore, bootstrap_packet, detect_host, server_instructions
from .dms import DMS, UtilityWeights
from .encoder import BSCEncoder
from .format import CartridgeHeader, read_cartridge, write_cartridge
from .offloader import ContextOffloader
from .rack import CartridgeRack, GradedExchange, RackError, graduation_pass
from .store import (
    KIND_DOC,
    KIND_FACT,
    KIND_GENERIC,
    KIND_NAMES,
    KIND_SKILL,
    KIND_TURN,
    Hit,
    SqacStore,
    resolve_kind,
)

__version__ = "0.1.5"

_KVCACHE_NAMES = frozenset(
    {
        "KVEstimate",
        "estimate_sparse_recall",
        "estimate_sweep",
        "known_models",
        "kv_bytes_per_token",
        "model_kv_bytes_per_token",
        "table",
    }
)


def __getattr__(name: str):
    """Lazily import sqac.kvcache names so `python -m sqac.kvcache` stays clean
    (eager import would pre-load the submodule and trigger a benign CPython
    'found in sys.modules' RuntimeWarning)."""
    if name in _KVCACHE_NAMES:
        from . import kvcache

        return getattr(kvcache, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "SqacStore",
    "ContextOffloader",
    "CartridgeRack",
    "BSCEncoder",
    "DMS",
    "UtilityWeights",
    "KVEstimate",
    "estimate_sparse_recall",
    "estimate_sweep",
    "known_models",
    "kv_bytes_per_token",
    "model_kv_bytes_per_token",
    "CartridgeHeader",
    "read_cartridge",
    "write_cartridge",
    "GradedExchange",
    "RackError",
    "graduation_pass",
    "KIND_DOC",
    "KIND_FACT",
    "KIND_SKILL",
    "KIND_TURN",
    "KIND_GENERIC",
    "KIND_NAMES",
    "Hit",
    "resolve_kind",
    "extract_project_units",
    "build_store",
    "track_once",
    "track",
    "_sync_rack",
    "ContinuityStore",
    "bootstrap_packet",
    "detect_host",
    "server_instructions",
]
