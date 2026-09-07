#!/usr/bin/env python3
"""Dataset -> .sqac cartridge ingestion.

Turns datasets into queryable memory:

    # JSONL with question/answer (or auto-detected text fields)
    python -m sqac.ingest knowledge.jsonl -o kb.sqac --semantic

    # plain text, one fact per line
    python -m sqac.ingest notes.txt -o notes.sqac

    # dedup near-duplicates (greedy, VSA similarity)
    python -m sqac.ingest corpus.jsonl -o kb.sqac --dedup 0.92

Auto-detected content fields: content, text, answer, fact, value, body, summary
Auto-detected key fields:     key, question, query, prompt, title, name, id
All other JSON fields become searchable metadata (returned with hits).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .encoder import BSCEncoder, MiniLMSimHashEncoder
from .store import SqacStore

CONTENT_FIELDS = ["content", "text", "answer", "fact", "value", "body", "summary", "output"]
KEY_FIELDS = ["key", "question", "query", "prompt", "title", "name", "id", "input"]


def pick_field(obj: dict, candidates: list[str]) -> str | None:
    for f in candidates:
        if f in obj and isinstance(obj[f], str) and obj[f].strip():
            return f
    return None


def read_items(path: Path) -> list[dict]:
    """Read JSONL (dict per line) or plain text (one fact per line)."""
    items: list[dict] = []
    raw = path.read_text(encoding="utf-8")
    is_json = path.suffix in {".jsonl", ".json"} or _looks_like_jsonl(raw)
    if is_json:
        for ln in raw.splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            obj = json.loads(ln)
            if isinstance(obj, str):
                obj = {"content": obj}
            items.append(obj)
    else:
        for ln in raw.splitlines():
            ln = ln.strip()
            if ln and not ln.startswith("#"):
                items.append({"content": ln})
    return items


def _looks_like_jsonl(raw: str) -> bool:
    for ln in raw.splitlines()[:5]:
        ln = ln.strip()
        if ln and not ln.startswith("{"):
            return False
    return bool(raw.strip())


def dedup_store(store: SqacStore, items: list[dict], threshold: float, kind: int | str | None = None) -> tuple[int, int]:
    """Greedy near-dup removal using the store's best available encoder.

    Returns (added, skipped). Dedup compares candidate content against
    already-accepted content; the first representative of a cluster wins.
    """
    added, skipped = 0, 0
    accepted: list[bytes] = []
    enc = store._sem_encoder if store.semantic else store.encoder
    # batch-encode in chunks to keep memory sane
    CHUNK = 256
    for start in range(0, len(items), CHUNK):
        chunk = items[start : start + CHUNK]
        texts = [str(o.get("_content", "")) for o in chunk]
        vecs = (
            enc.encode_bits_batch(texts)
            if hasattr(enc, "encode_bits_batch")
            else [enc.encode_bits(t) for t in texts]
        )
        for o, v in zip(chunk, vecs):
            dup = False
            for av in accepted[-4096:]:  # cap scan for very large sets
                if enc.similarity(v, av) >= threshold:
                    dup = True
                    break
            if dup:
                skipped += 1
                continue
            accepted.append(v)
            store.add(o["_content"], key=o.get("_key"), meta=o["_meta"], source=o["_source"], kind=kind)
            added += 1
    return added, skipped


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sqac.ingest", description="dataset -> .sqac cartridge")
    ap.add_argument("input", help="JSONL or plain-text file")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--text-field", default=None, help="JSON field for fact content")
    ap.add_argument("--key-field", default=None, help="JSON field for lookup key")
    ap.add_argument("--source-field", default=None, help="JSON field to use as provenance")
    ap.add_argument("--semantic", action="store_true", help="enable MiniLM semantic tier")
    ap.add_argument("--dedup", type=float, default=0.0, help="near-dup similarity threshold (e.g. 0.92); 0 = off")
    ap.add_argument("--kind", default=None, help="knowledge kind stamp: fact/skill/doc/turn (default: generic)")
    ap.add_argument("--name", default="")
    args = ap.parse_args(argv)

    path = Path(args.input)
    items = read_items(path)
    if not items:
        print(f"no items found in {path}")
        return 1

    # resolve fields per item
    prepared: list[dict] = []
    field_stats = {"content": args.text_field, "key": args.key_field}
    for obj in items:
        if not isinstance(obj, dict):
            obj = {"content": str(obj)}
        tfield = args.text_field or pick_field(obj, CONTENT_FIELDS)
        if tfield is None:
            continue  # no content field; skip line
        content = obj[tfield].strip()
        if not content:
            continue
        kfield = args.key_field or pick_field(obj, KEY_FIELDS)
        key = obj[kfield].strip() if kfield and kfield != tfield else None
        sfield = args.source_field or ("source" if "source" in obj else None)
        source = str(obj.get(sfield, path.name)) if sfield else path.name
        meta = {k: v for k, v in obj.items() if k not in {tfield, kfield, sfield, "source"}}
        prepared.append({"_content": content, "_key": key, "_source": source, "_meta": meta})

    store = SqacStore(semantic=args.semantic)
    if args.dedup > 0:
        added, skipped = dedup_store(store, prepared, args.dedup, kind=args.kind)
    else:
        for o in prepared:
            store.add(o["_content"], key=o["_key"], meta=o["_meta"], source=o["_source"], kind=args.kind)
        added, skipped = len(prepared), 0

    store.save(args.out, name=args.name or path.stem, description=f"ingested from {path}")
    size = Path(args.out).stat().st_size
    print(f"ingested {added} facts ({skipped} near-duplicates skipped) -> {args.out}")
    print(f"cartridge: {size / 1024:.1f} KB, {len(store)} entries, semantic={'on' if args.semantic else 'off'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
