"""SQAC — hot-swappable VSA memory cartridges.

Public API:
    SqacStore       — hybrid exact + fuzzy memory store
    BSCEncoder      — zero-dependency binary spatter-code encoder
    read/write      — .sqac binary format v1
"""

from .encoder import BSCEncoder
from .store import SqacStore
from .format import read_cartridge, write_cartridge, CartridgeHeader

__version__ = "0.1.0"

__all__ = [
    "SqacStore",
    "BSCEncoder",
    "CartridgeHeader",
    "read_cartridge",
    "write_cartridge",
]
