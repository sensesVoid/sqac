"""mem_id generation: 26-char Crockford base32 ULIDs (stdlib only)."""

import os
import time

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def new_mem_id() -> str:
    """Return a new time-sortable 26-char Crockford base32 ULID."""
    timestamp_ms = time.time_ns() // 1_000_000 & ((1 << 48) - 1)
    randomness = int.from_bytes(os.urandom(10), "big")
    value = (timestamp_ms << 80) | randomness
    chars = []
    for _ in range(26):
        chars.append(_CROCKFORD[value & 0x1F])
        value >>= 5
    chars.reverse()
    return "".join(chars)
