"""PII detection: regex patterns + Luhn validation."""

from __future__ import annotations

import re
from typing import Any


def _luhn_valid(num: str) -> bool:
    digits = [int(c) for c in num if c.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    total = 0
    parity = len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


_PATTERNS = [
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("phone", re.compile(r"\+?\d[\d\s\-]{8,}\d")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("credit_card", None),  # handled separately with Luhn
    ("ip", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("iban", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4,30}\b")),
]


def detect_pii(text: str) -> list[dict[str, Any]]:
    """Return list of {type, span, value} findings. Non-blocking."""
    findings = []
    for name, pat in _PATTERNS:
        if pat is None:
            continue
        for m in pat.finditer(text):
            findings.append({"type": name, "start": m.start(), "end": m.end(), "value": m.group()})

    # Credit cards: find digit sequences and validate with Luhn
    for m in re.finditer(r"\b(?:\d[ -]*?){13,19}\b", text):
        raw = m.group()
        digits = re.sub(r"\D", "", raw)
        if _luhn_valid(digits):
            findings.append({"type": "credit_card", "start": m.start(), "end": m.end(), "value": raw})

    return findings


def scan_cartridge(path: str) -> list[dict[str, Any]]:
    """Scan a plaintext cartridge for PII. Returns list of findings with cartridge path."""
    from .format import read_cartridge

    findings = []
    try:
        cart = read_cartridge(path)
    except Exception:
        return findings
    for entry in cart.entries:
        payload = entry.payload if isinstance(entry.payload, dict) else {}
        content = payload.get("content", "") or payload.get("text", "")
        if not isinstance(content, str):
            continue
        for f in detect_pii(content):
            f = dict(f)
            f["cartridge"] = path
            f["entry_key"] = payload.get("key", "")
            findings.append(f)
    return findings


def warn_if_pii(text: str, context: str = "") -> list[dict[str, Any]]:
    findings = detect_pii(text)
    if findings:
        import warnings
        warnings.warn(f"content in {context or 'memory'} looks like PII: {[f['type'] for f in findings]}", stacklevel=2)
    return findings
