#!/usr/bin/env python3
"""SQAC CLI — teach, search, pack, stats, init, track.

    python -m sqac.cli teach "our deploys are ARM64 only" [--key deployment] [--db memory.sqac]
    python -m sqac.cli search "deployment target?" [--db memory.sqac] [--k 3]
    python -m sqac.cli pack rules.txt -o rules.sqac     # one fact per line
    python -m sqac.cli stats --db memory.sqac
    python -m sqac.cli init .                           # auto-build cartridge from project
    python -m sqac.cli track . --interval 5             # realtime tracking (Ctrl-C to stop)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .store import SqacStore

DEFAULT_DB = "memory.sqac"
DEFAULT_SQAC_DIR = ".sqac"


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
    p_teach.add_argument("--kind", default=None, help="knowledge kind: fact/skill/doc/turn (default: generic)")
    p_teach.add_argument("--source", default="cli")
    add_db(p_teach)

    p_search = sub.add_parser("search", help="query the memory")
    p_search.add_argument("query")
    p_search.add_argument("--k", type=int, default=3)
    p_search.add_argument("--kind", default=None, help="filter by knowledge kind: fact/skill/doc/turn")
    p_search.add_argument("--threshold", type=float, default=None)
    add_db(p_search)

    p_pack = sub.add_parser("pack", help="build a cartridge from a text file (one fact per line)")
    p_pack.add_argument("input")
    p_pack.add_argument("-o", "--out", required=True)
    p_pack.add_argument("--kind", default=None, help="knowledge kind stamp: fact/skill/doc/turn")
    p_pack.add_argument("--name", default="")
    p_pack.add_argument("--source", default="pack")

    p_stats = sub.add_parser("stats", help="show cartridge stats")
    add_db(p_stats)

    p_init = sub.add_parser("init", help="auto-build a project cartridge from a directory")
    p_init.add_argument("root", nargs="?", default=".", help="project root (default: .)")
    p_init.add_argument("-o", "--out-dir", default=DEFAULT_SQAC_DIR,
                        help="output .sqac dir (default: .sqac)")
    p_init.add_argument("--semantic", action="store_true",
                        help="enable semantic tier in the built cartridge")

    p_track = sub.add_parser("track", help="realtime project tracking (Ctrl-C to stop)")
    p_track.add_argument("root", nargs="?", default=".", help="project root (default: .)")
    p_track.add_argument("-o", "--out-dir", default=DEFAULT_SQAC_DIR,
                         help="output .sqac dir (default: .sqac)")
    p_track.add_argument("--interval", type=float, default=5.0,
                         help="poll interval in seconds (default: 5)")
    p_track.add_argument("--log", default=None, metavar="FILE",
                         help="append JSONL audit log (one event per line)")
    p_track.add_argument("--rack", default=None, metavar="DIR",
                         help="auto-register cartridge into a CartridgeRack directory")
    p_track.add_argument("--rack-name", default=None,
                         help="name for the cartridge in the rack (default: project dir name)")

    args = ap.parse_args(argv)

    if args.cmd == "teach":
        store = _open(args.db)
        store.add(args.content, key=args.key, source=args.source, kind=args.kind)
        store.save(args.db)
        print(f"taught: {args.content[:70]}{'…' if len(args.content) > 70 else ''}")
        print(f"cartridge: {args.db} ({len(store)} entries)")

    elif args.cmd == "search":
        store = _open(args.db)
        if args.threshold is not None:
            store.fuzzy_threshold = args.threshold
        hits = store.search(args.query, top_k=args.k, kind=args.kind)
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
            store.add(ln, source=args.source, kind=args.kind)
        store.save(args.out, name=args.name, description=f"packed from {args.input}")
        print(f"packed {len(lines)} facts -> {args.out} ({len(store)} entries)")

    elif args.cmd == "stats":
        store = _open(args.db)
        for k, v in store.stats().items():
            if k == "kinds":
                if v:
                    print("kinds:")
                    for kk, vv in sorted(v.items()):
                        print(f"  {kk}: {vv}")
            else:
                print(f"{k}: {v}")

    elif args.cmd == "init":
        from .autobuild import extract_project_units, build_store
        root = Path(args.root).resolve()
        out_dir = Path(args.out_dir)
        print(f"scanning {root} ...")
        units = extract_project_units(root)
        if not units:
            print("no units extracted — empty or unsupported project")
            return 1
        store = build_store(units)
        out_dir.mkdir(parents=True, exist_ok=True)
        cart_path = out_dir / "project.sqac"
        store.save(cart_path, name=root.name,
                   description=f"autobuilt from {root}")
        stats = store.stats()
        size_kb = cart_path.stat().st_size / 1024
        print(f"built {cart_path} ({stats['entries']} entries, {size_kb:.1f} KB)")
        print(f"  kinds: {', '.join(f'{k}={v}' for k, v in sorted(stats['kinds'].items()))}")
        # also write state.json so track can diff against the initial build
        from .autobuild import save_state, unit_map
        save_state(out_dir, unit_map(units))
        print(f"  state: {out_dir / 'state.json'}")

    elif args.cmd == "track":
        from .autobuild import track
        root = Path(args.root).resolve()
        out_dir = Path(args.out_dir)
        log_path = Path(args.log) if args.log else None
        rack_dir = Path(args.rack) if args.rack else None
        rack_name = args.rack_name
        try:
            track(root, out_dir, args.interval, log_path=log_path,
                  rack_dir=rack_dir, rack_name=rack_name)
        except KeyboardInterrupt:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
