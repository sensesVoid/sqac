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

import hashlib
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MAGIC = b"SQAC"
VERSION = 3

# Entry flag bits
FLAG_DELETED = 0x0001  # tombstone: entry ignored by reads
FLAG_BINVEC = 0x0002   # raw binary vector block present (v2+)
FLAG_KIND_SHIFT = 4    # bits 4-11: knowledge kind, uint8 (v3+)
FLAG_KIND_MASK = 0xFF << FLAG_KIND_SHIFT


class FormatError(ValueError):
    """Raised when a .sqac file is malformed or incompatible."""


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
) -> None:
    """Write a complete cartridge to disk atomically (tmp file + rename)."""
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
        parts.append(struct.pack("<HI", flags, len(pb)))
        parts.append(bytes(e.key_bits))
        if e.binvec:
            parts.append(struct.pack("<I", len(e.binvec)))
            parts.append(e.binvec)
        parts.append(pb)

    parts.append(struct.pack("<I", len(ext_blocks or [])))
    for name, blob in ext_blocks or []:
        nb = name.encode("utf-8")
        parts.append(struct.pack("<I", len(nb)))
        parts.append(nb)
        parts.append(struct.pack("<I", len(blob)))
        parts.append(blob)

    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        for chunk in parts:
            f.write(chunk)
    tmp.replace(path)


@dataclass
class Cartridge:
    """A fully parsed cartridge."""

    header: CartridgeHeader
    vocab: dict[str, int]
    entries: list[Entry]
    ext_blocks: dict[str, bytes] = field(default_factory=dict)


def read_cartridge(path: str | Path) -> Cartridge:
    path = Path(path)
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
        payload = json.loads(take(pb_len).decode("utf-8"))
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
