#!/usr/bin/env python3
"""SQAC CLI — teach, search, pack, stats.

    python -m sqac.cli teach "our deploys are ARM64 only" [--key deployment] [--db memory.sqac]
    python -m sqac.cli search "deployment target?" [--db memory.sqac] [--k 3]
    python -m sqac.cli pack rules.txt -o rules.sqac     # one fact per line
    python -m sqac.cli stats --db memory.sqac
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .store import SqacStore

DEFAULT_DB = "memory.sqac"


def _open(path: str) -> SqacStore:
    p = Path(path)
    return SqacStore.load(p) if p.exists() else SqacStore()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sqac", description="VSA memory cartridges")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_db(p):
        p.add_argument("--db", default=DEFAULT_DB, help="cartridge path (default: ./memory.sqac)")

    p_teach = sub.add_parser("teach", help="teach one fact")
    p_teach.add_argument("content")
    p_teach.add_argument("--key", default=None, help="lookup key (defaults to content)")
    p_teach.add_argument("--source", default="cli")
    add_db(p_teach)

    p_search = sub.add_parser("search", help="query the memory")
    p_search.add_argument("query")
    p_search.add_argument("--k", type=int, default=3)
    p_search.add_argument("--threshold", type=float, default=None)
    add_db(p_search)

    p_pack = sub.add_parser("pack", help="build a cartridge from a text file (one fact per line)")
    p_pack.add_argument("input")
    p_pack.add_argument("-o", "--out", required=True)
    p_pack.add_argument("--name", default="")
    p_pack.add_argument("--source", default="pack")

    p_stats = sub.add_parser("stats", help="show cartridge stats")
    add_db(p_stats)

    args = ap.parse_args(argv)

    if args.cmd == "teach":
        store = _open(args.db)
        store.add(args.content, key=args.key, source=args.source)
        store.save(args.db)
        print(f"taught: {args.content[:70]}{'…' if len(args.content) > 70 else ''}")
        print(f"cartridge: {args.db} ({len(store)} entries)")

    elif args.cmd == "search":
        store = _open(args.db)
        if args.threshold is not None:
            store.fuzzy_threshold = args.threshold
        hits = store.search(args.query, top_k=args.k)
        if not hits:
            print("no hit (confidence below threshold)")
            return 1
        for h in hits:
            tag = "EXACT" if h.mode == "exact" else f"fuzzy"
            print(f"[{tag} {h.confidence:.3f}] {h.content}")
            if h.source:
                print(f"          source: {h.source}")

    elif args.cmd == "pack":
        lines = [
            ln.strip()
            for ln in Path(args.input).read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        store = SqacStore()
        for ln in lines:
            store.add(ln, source=args.source)
        store.save(args.out, name=args.name, description=f"packed from {args.input}")
        print(f"packed {len(lines)} facts -> {args.out} ({len(store)} entries)")

    elif args.cmd == "stats":
        store = _open(args.db)
        for k, v in store.stats().items():
            print(f"{k}: {v}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
