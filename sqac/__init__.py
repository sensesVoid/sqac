"""SQAC — hot-swappable VSA memory cartridges.

Public API:
    SqacStore          — hybrid exact + fuzzy memory store
    ContextOffloader   — ephemeral turn bucket (observe/save/recall)
    CartridgeRack      — named cartridges + kind-based routing
    BSCEncoder         — zero-dependency binary spatter-code encoder
    read/write         — .sqac binary format v1
"""

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

__version__ = "0.1.0"

__all__ = [
    "SqacStore",
    "ContextOffloader",
    "CartridgeRack",
    "BSCEncoder",
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
]
