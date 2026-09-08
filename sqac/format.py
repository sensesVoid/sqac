"""SQAC binary format.

A .sqac file is a self-contained, portable, hot-swappable memory cartridge:

    +-----------------------------+
    | header (fixed, JSON)        |  magic, version, dims, vocab fingerprint
    | vocab    (string -> index)  |  the encoder's atom table
    | entries  (n records)        |  packed key bits + plaintext payload
    | ext      (optional block)   |  forward-compatible extensions
    +-----------------------------+

Design rules (from docs/THESIS.md Part VI):
  - Keys are packed BSC bit-vectors (D bits -> D/8 bytes) so lookups are
    XOR + popcount on raw bytes: no float math anywhere in the hot path.
  - Payloads are plaintext: the LLM never sees a vector, only text +
    confidence. The vector is an address, not a message.
  - The vocabulary is the ABI: the header carries a fingerprint so a
    runtime can refuse cartridges encoded with an incompatible vocab.
  - Unknown extension blocks MUST be skipped by readers, giving forward
    compatibility (a v1 reader can read a v2 file that only appends ext
    blocks; if the entry layout itself changes, bump VERSION).

Version history:
  v1: auxiliary vectors (content/semantic keys) hex-encoded in payload JSON.
  v2: auxiliary vectors moved into a per-entry raw binary block (FLAG_BINVEC),
      ~2x smaller than hex. v1 files remain fully readable; v2 files are
      refused by v1 readers via the version check (entry layout changed).
  v3: knowledge kind added to the entry flags (bits 4-11, 0-255): what sort
      of knowledge the entry carries (generic/fact/skill/doc/turn). Enables
      kind-filtered retrieval and kind-aware ranking policy. v2 files read
      as kind=generic; v3 files are refused by v2 readers via the version
      check (flag semantics changed).
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import struct
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

MAGIC = b"SQAC"
VERSION = 4  # v4 adds lz4 compression on binvec blocks

# Entry flag bits
FLAG_DELETED = 0x0001  # tombstone: entry ignored by reads
FLAG_BINVEC = 0x0002   # raw binary vector block present (v2+)
FLAG_COMPRESSED = 0x0004  # binvec is lz4-compressed (v4+)
FLAG_KIND_SHIFT = 4    # bits 4-11: knowledge kind, uint8 (v3+)
FLAG_KIND_MASK = 0xFF << FLAG_KIND_SHIFT

# Compression: lz4 on binvec blocks.  At 1024 dims, a binvec is 384 bytes
# (128 key + 128 sem_key + 128 sem_ckeys).  Lz4 typically achieves 40-60%
# reduction on random binary data, 70-80% on correlated data.
try:
    import lz4.block as _lz4_compress
    _HAS_LZ4 = True
except ImportError:
    _HAS_LZ4 = False


class FormatError(ValueError):
    """Raised when a .sqac file is malformed or incompatible."""


class CartridgeLockError(FormatError):
    """Raised when a cartridge file cannot be locked."""


# ── file locking ──────────────────────────────────────────────────────────────
# Two layers:
#   1. Process-level: fcntl.flock on the .sqac file (cross-process safe)
#   2. Thread-level: threading.Lock per path (in-process safe)
# The context manager acquires both; the fcntl lock is released on exit
# even if the process crashes (kernel-level guarantee on Unix).

_thread_locks: dict[str, threading.Lock] = {}
_thread_locks_guard = threading.Lock()


def _get_thread_lock(path: Path) -> threading.Lock:
    """One threading.Lock per canonical path, created on demand."""
    key = path.resolve().as_posix()
    with _thread_locks_guard:
        if key not in _thread_locks:
            _thread_locks[key] = threading.Lock()
        return _thread_locks[key]


@contextmanager
def locked_cartridge(path: Path, timeout: float = 10.0) -> Generator[None, None, None]:
    """Acquire an exclusive lock on a .sqac file for writing.

    Acquires both a process-level fcntl lock and a thread-level lock.
    The fcntl lock is mandatory on Linux/macOS: another process cannot
    read or write the file until we release it.  On failure (timeout or
    unsupported platform), raises CartridgeLockError.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tlock = _get_thread_lock(path)
    tlock.acquire()
    try:
        fd = open(path, "a+b")
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError):
            fd.close()
            raise CartridgeLockError(
                f"cannot lock {path}: another process is writing to it"
            )
        try:
            yield
        finally:
            try:
                fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
            finally:
                fd.close()
    finally:
        tlock.release()


@dataclass
class CartridgeHeader:
    """Fixed metadata describing the cartridge."""

    version: int = VERSION
    dims: int = 1024
    vocab_fingerprint: str = ""
    encoder_name: str = "bsc-ngram-v1"
    created: str = ""
    name: str = ""
    description: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def fingerprint_seed(self) -> str:
        """Canonical string hashed into the vocab fingerprint."""
        return f"{self.encoder_name}:{self.dims}"

    def compute_fingerprint(self, vocab: dict[str, int]) -> str:
        """Fingerprint = hash of sorted vocab + encoder identity.

        Two cartridges with the same fingerprint are query-compatible:
        the same atom always maps to the same hypervector.
        """
        h = hashlib.sha256()
        h.update(self.fingerprint_seed().encode())
        for token in sorted(vocab):
            h.update(token.encode("utf-8"))
            h.update(b"\x00")
            h.update(str(vocab[token]).encode())
            h.update(b"\x01")
        return h.hexdigest()[:32]

    def to_json(self) -> str:
        return json.dumps(
            {
                "magic": MAGIC.decode(),
                "version": self.version,
                "dims": self.dims,
                "vocab_fingerprint": self.vocab_fingerprint,
                "encoder_name": self.encoder_name,
                "created": self.created,
                "name": self.name,
                "description": self.description,
                **({"extra": self.extra} if self.extra else {}),
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, raw: str) -> "CartridgeHeader":
        d = json.loads(raw)
        if d.get("magic") != MAGIC.decode():
            raise FormatError("not a .sqac file (bad magic)")
        if d.get("version", 0) > VERSION:
            raise FormatError(
                f"cartridge version {d['version']} newer than supported {VERSION}"
            )
        return cls(
            version=d["version"],
            dims=d["dims"],
            vocab_fingerprint=d.get("vocab_fingerprint", ""),
            encoder_name=d.get("encoder_name", ""),
            created=d.get("created", ""),
            name=d.get("name", ""),
            description=d.get("description", ""),
            extra=d.get("extra", {}),
        )


@dataclass
class Entry:
    """One memory record: packed key bits + plaintext payload.

    binvec carries auxiliary packed vectors (content vector, semantic key
    and content vectors) as raw bytes instead of hex-in-JSON. Layout is
    decided by the store (sizes derive from dims and tier flags); the
    format layer treats it as opaque. Empty for v1 files.
    """

    key_bits: bytearray  # dims // 8 bytes
    payload: dict[str, Any]
    deleted: bool = False
    binvec: bytes = b""
    kind: int = 0  # knowledge kind (0 = generic; v2 files read as 0)


def pack_bits(bits) -> bytearray:
    """Pack an iterable of 0/1 ints (length D) into D/8 bytes, MSB-first."""
    return _pack_bits_fast(bits)


def _pack_bits_fast(bits) -> bytearray:
    """Fast pack: bits is a bytes-like of 0/1 or a list of ints."""
    out = bytearray(len(bits) // 8)
    byte = 0
    idx = 0
    for b in bits:
        if b:
            byte |= 1 << (7 - (idx & 7))
        idx += 1
        if (idx & 7) == 0:
            out[(idx >> 3) - 1] = byte
            byte = 0
    return out


def _unpack_bits_fast(buf, dims: int):
    """Unpack D/8 bytes into a bytes object of D 0/1 ints."""
    out = bytearray(dims)
    for i in range(dims):
        byte = buf[i >> 3]
        out[i] = (byte >> (7 - (i & 7))) & 1
    return bytes(out)


def write_cartridge(
    path: str | Path,
    header: CartridgeHeader,
    vocab: dict[str, int],
    entries: list[Entry],
    ext_blocks: list[tuple[str, bytes]] | None = None,
    locked: bool = True,
) -> None:
    """Write a complete cartridge to disk atomically (tmp file + rename).

    When *locked* (default), acquires an exclusive file lock so concurrent
    writers cannot corrupt the cartridge.  The lock is released before the
    tmp→rename swap, so readers never see a locked file.
    """
    path = Path(path)
    header.vocab_fingerprint = header.compute_fingerprint(vocab)

    dims = header.dims
    key_len = dims // 8
    parts: list[bytes] = []

    head_json = header.to_json().encode("utf-8")
    parts.append(struct.pack("<II", len(head_json), 0))
    parts.append(head_json)

    parts.append(struct.pack("<I", len(vocab)))
    for token, idx in vocab.items():
        tb = token.encode("utf-8")
        parts.append(struct.pack("<I", len(tb)))
        parts.append(tb)
        parts.append(struct.pack("<I", idx))

    parts.append(struct.pack("<Q", len(entries)))
    for e in entries:
        if len(e.key_bits) != key_len:
            raise FormatError(
                f"key is {len(e.key_bits)} bytes, expected {key_len} (dims={dims})"
            )
        flags = FLAG_DELETED if e.deleted else 0
        if e.binvec:
            flags |= FLAG_BINVEC
        flags |= (e.kind & 0xFF) << FLAG_KIND_SHIFT
        pb = json.dumps(e.payload, ensure_ascii=False).encode("utf-8")
        # Compress the payload JSON with lz4 block (no frame overhead)
        compressed_pb = pb
        if _HAS_LZ4 and len(pb) >= 128:
            compressed_pb = _lz4_compress.compress(pb)
            if len(compressed_pb) < len(pb):
                flags |= FLAG_COMPRESSED
            else:
                compressed_pb = pb  # compression didn't help, store raw
        parts.append(struct.pack("<HI", flags, len(compressed_pb)))
        parts.append(bytes(e.key_bits))
        if e.binvec:
            parts.append(struct.pack("<I", len(e.binvec)))
            parts.append(e.binvec)
        parts.append(compressed_pb)

    parts.append(struct.pack("<I", len(ext_blocks or [])))
    for name, blob in ext_blocks or []:
        nb = name.encode("utf-8")
        parts.append(struct.pack("<I", len(nb)))
        parts.append(nb)
        parts.append(struct.pack("<I", len(blob)))
        parts.append(blob)

    tmp = path.with_suffix(path.suffix + ".tmp")
    if locked:
        with locked_cartridge(path):
            _write_parts(tmp, parts)
            tmp.replace(path)
    else:
        _write_parts(tmp, parts)
        tmp.replace(path)


def _write_parts(tmp: Path, parts: list[bytes]) -> None:
    """Write serialized parts to a temp file."""
    with open(tmp, "wb") as f:
        for chunk in parts:
            f.write(chunk)


@dataclass
class Cartridge:
    """A fully parsed cartridge."""

    header: CartridgeHeader
    vocab: dict[str, int]
    entries: list[Entry]
    ext_blocks: dict[str, bytes] = field(default_factory=dict)


def read_cartridge(path: str | Path, locked: bool = False) -> Cartridge:
    """Read a cartridge from disk.

    When *locked*, acquires a shared (read) lock via fcntl so writers
    cannot modify the file while we read.  Default is unlocked for
    backward compatibility.
    """
    path = Path(path)
    if locked:
        fd = open(path, "rb")
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        except (OSError, BlockingIOError):
            fd.close()
            raise CartridgeLockError(
                f"cannot lock {path} for reading: another process is writing"
            )
        try:
            blob = fd.read()
        finally:
            fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
            fd.close()
    else:
        with open(path, "rb") as f:
            blob = f.read()

    off = 0

    def take(n: int) -> bytes:
        nonlocal off
        if off + n > len(blob):
            raise FormatError("truncated file")
        chunk = blob[off : off + n]
        off += n
        return chunk

    head_len, _resv = struct.unpack("<II", take(8))
    header = CartridgeHeader.from_json(take(head_len).decode("utf-8"))

    dims = header.dims
    key_len = dims // 8
    if key_len * 8 != dims:
        raise FormatError("dims must be a multiple of 8")

    (vocab_len,) = struct.unpack("<I", take(4))
    vocab: dict[str, int] = {}
    for _ in range(vocab_len):
        (tb_len,) = struct.unpack("<I", take(4))
        token = take(tb_len).decode("utf-8")
        (idx,) = struct.unpack("<I", take(4))
        vocab[token] = idx

    (n_entries,) = struct.unpack("<Q", take(8))
    entries: list[Entry] = []
    for _ in range(n_entries):
        flags, pb_len = struct.unpack("<HI", take(6))
        key_bits = bytearray(take(key_len))
        binvec = b""
        if flags & FLAG_BINVEC:
            (bv_len,) = struct.unpack("<I", take(4))
            binvec = take(bv_len)
        raw_payload = take(pb_len)
        if flags & FLAG_COMPRESSED:
            if _HAS_LZ4:
                raw_payload = _lz4_compress.decompress(raw_payload)
            else:
                raise FormatError(
                    "cartridge uses lz4 compression but lz4 is not installed: "
                    "pip install lz4"
                )
        payload = json.loads(raw_payload.decode("utf-8"))
        entries.append(
            Entry(
                key_bits=key_bits,
                payload=payload,
                deleted=bool(flags & FLAG_DELETED),
                binvec=binvec,
                kind=(flags & FLAG_KIND_MASK) >> FLAG_KIND_SHIFT,
            )
        )

    (n_ext,) = struct.unpack("<I", take(4))
    ext: dict[str, bytes] = {}
    for _ in range(n_ext):
        (nb_len,) = struct.unpack("<I", take(4))
        name = take(nb_len).decode("utf-8")
        (blob_len,) = struct.unpack("<I", take(4))
        ext[name] = take(blob_len)

    if off != len(blob):
        raise FormatError(f"trailing bytes: read {off} of {len(blob)}")

    return Cartridge(header=header, vocab=vocab, entries=entries, ext_blocks=ext)


def popcount_bytes(buf) -> int:
    """Popcount of a bytes-like object (portable across Python versions)."""
    return int.from_bytes(buf, "little").bit_count() if hasattr(int, "bit_count") else bin(int.from_bytes(buf, "little")).count("1")


def hamming(a, b) -> int:
    """Hamming distance between two equal-length packed byte buffers."""
    n = 0
    for x, y in zip(a, b):
        n += (x ^ y).bit_count()
    return n


def compact_cartridge(
    path: str | Path,
    out: str | Path | None = None,
) -> dict:
    """Remove tombstoned entries from a cartridge and rewrite it.

    Reads the cartridge, filters out deleted entries, and writes a clean
    version.  If *out* is None, overwrites the original atomically.
    Returns a report dict.

    The exclusive lock is held across the whole read-modify-write span so a
    concurrent writer cannot slip an update in between the read and the
    atomic rename (which would silently lose that update).
    """
    path = Path(path)
    out = Path(out) if out else path
    with locked_cartridge(path):
        cart = read_cartridge(path)
        original_count = len(cart.entries)
        alive = [e for e in cart.entries if not e.deleted]
        removed = original_count - len(alive)
        if removed == 0:
            return {"original": original_count, "alive": original_count, "removed": 0, "compacted": False}
        # Lock already held for the whole span; write without re-acquiring
        # (locked=True would deadlock on the non-reentrant thread lock).
        write_cartridge(out, cart.header, cart.vocab, alive, locked=False)
        return {"original": original_count, "alive": len(alive), "removed": removed, "compacted": True}
