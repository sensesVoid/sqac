"""Pluggable pre-encoding text normalization for SQAC hybrid retrieval.

Character-trigram BSC encoding is language-agnostic but naive about:
accented Latin, case, stopwords (noise grams dominate long sentences), and
morphological affixes (Austronesian/affix-heavy and agglutinative languages
inflate the gram space without adding signal).

This module provides a small, composable pipeline:

    SQAC_NORMALIZE="unicode_fold,stopwords:en,affix:en"

Supported steps (comma-separated, applied in the order given):

- ``unicode_fold`` — NFKD decomposition, strip combining marks
  (``naïve`` → ``naive``, ``café`` → ``cafe``). Case is left to the encoder.
- ``stopwords:<lang>`` — drop function words. Built-in lists: ``en``
  (minimal high-frequency set). For other languages pass
  ``stopwords:file=<path>`` — a UTF-8 word-per-line file.
- ``affix:<lang>`` — lightweight rule-based suffix stripping. ``en``:
  strips ``-ing``, ``-ed``, ``-es``, ``-s``, ``-ly`` when the stem stays
  ≥ 3 characters. Other languages: ``affix:file=<path>`` — one suffix per
  line, longest first (order preserved).

Normalization is **opt-in and additive**. Cartridges written before a
normalizer was enabled remain valid: vectors are computed in the same
space, and similarity simply degrades gracefully for cross-type pairs.
Default behavior (no env var) is byte-identical to before.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Callable, Iterable, Optional

# Minimal English high-frequency stoplist (function words only — keep it
# conservative; content words and prepositions carrying meaning stay).
_EN_STOPWORDS = frozenset(
    "a an the and or but if then else for in on at to of by with from as is are "
    "was were be been being it its this that these those i you he she we they "
    "me him her us them my your his its our their what which who whom when where "
    "how why can could should would may might must will shall not no do does did".split()
)

_EN_AFFIXES = ("ing", "ed", "es", "s", "ly")

_TOKEN = re.compile(r"\S+")


def unicode_fold(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def _token_filter_factory(words: frozenset[str]) -> Callable[[str], str]:
    def _filter(text: str) -> str:
        return " ".join(t for t in _TOKEN.findall(text) if t.lower() not in words)
    return _filter


def stopwords_en() -> Callable[[str], str]:
    return _token_filter_factory(_EN_STOPWORDS)


def _read_words(path: str) -> frozenset[str]:
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"SQAC_NORMALIZE: word list not found: {path}")
    return frozenset(
        line.strip().lower()
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    )


def _affix_factory(suffixes: Iterable[str], min_stem: int = 3) -> Callable[[str], str]:
    affixes = tuple(sorted({s.lower() for s in suffixes}, key=len, reverse=True))

    def _strip(text: str) -> str:
        out = []
        for token in _TOKEN.findall(text):
            low = token.lower()
            for affix in affixes:
                if low.endswith(affix) and len(low) - len(affix) >= min_stem:
                    low = low[: -len(affix)]
                    break
            out.append(low)
        return " ".join(out)

    return _strip


def stopwords(lang: str) -> Callable[[str], str]:
    if lang == "en":
        return stopwords_en()
    if lang.startswith("file="):
        return _token_filter_factory(_read_words(lang[5:]))
    raise ValueError(f"SQAC_NORMALIZE: unknown stopwords list: {lang!r}")


def affix(lang: str) -> Callable[[str], str]:
    if lang == "en":
        return _affix_factory(_EN_AFFIXES)
    if lang.startswith("file="):
        suffixes = _read_words(lang[5:])
        return _affix_factory(suffixes)
    raise ValueError(f"SQAC_NORMALIZE: unknown affix rules: {lang!r}")


def _resolve_step(spec: str) -> Callable[[str], str]:
    if spec == "unicode_fold":
        return unicode_fold
    if ":" not in spec:
        raise ValueError(f"SQAC_NORMALIZE: unknown step: {spec!r}")
    name, arg = spec.split(":", 1)
    if name == "stopwords":
        return stopwords(arg)
    if name == "affix":
        return affix(arg)
    raise ValueError(f"SQAC_NORMALIZE: unknown step: {spec!r}")


def resolve(spec: str | None) -> Optional[Callable[[str], str]]:
    """Parse a normalization spec into a callable, or None when empty.

    Example: ``unicode_fold,stopwords:en,affix:en``
    """
    if not spec:
        return None
    steps = [_resolve_step(s.strip()) for s in spec.split(",") if s.strip()]
    if not steps:
        return None

    def _apply(text: str) -> str:
        for step in steps:
            text = step(text)
        return text

    return _apply


def from_env(var: str = "SQAC_NORMALIZE") -> Optional[Callable[[str], str]]:
    import os

    return resolve(os.environ.get(var))
