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


def _crockford_decode(s: str) -> int:
    """Decode a Crockford base32 string into an int (folds chars MSB-first)."""
    value = 0
    for ch in s:
        value = value * 32 + _CROCKFORD.index(ch)
    return value


def decode_mem_id(mem_id: str) -> tuple[int, int]:
    """Return ``(timestamp_ms, randomness)`` from a 26-char Crockford ULID."""
    value = _crockford_decode(mem_id)
    return value >> 80, value & ((1 << 80) - 1)


def decode_mem_id_ms(mem_id: str) -> int:
    """Return just the millisecond timestamp from a 26-char Crockford ULID."""
    return decode_mem_id(mem_id)[0]


class MonotonicMemId:
    """Monotonic ULID generator: strictly increasing per instance."""

    def __init__(self) -> None:
        self._last_ms = 0
        self._last_rand = 0

    def mint(self) -> str:
        """Return a 26-char ULID that is strictly greater than the previous one."""
        timestamp_ms = time.time_ns() // 1_000_000 & ((1 << 48) - 1)
        if timestamp_ms > self._last_ms:
            self._last_ms = timestamp_ms
            self._last_rand = int.from_bytes(os.urandom(10), "big")
        else:
            self._last_rand += 1
            if self._last_rand > (1 << 80) - 1:
                self._last_rand = 0
                self._last_ms += 1
        return self._encode(self._last_ms, self._last_rand)

    def seed(self, timestamp_ms: int, randomness: int) -> None:
        """Resume from a prior high-water mark."""
        self._last_ms = timestamp_ms
        self._last_rand = randomness

    def seed_from(self, mem_id: str) -> None:
        """Decode an existing id and seed from it."""
        timestamp_ms, randomness = decode_mem_id(mem_id)
        self.seed(timestamp_ms, randomness)

    @staticmethod
    def _encode(timestamp_ms: int, randomness: int) -> str:
        value = (timestamp_ms << 80) | randomness
        chars = []
        for _ in range(26):
            chars.append(_CROCKFORD[value & 0x1F])
            value >>= 5
        chars.reverse()
        return "".join(chars)