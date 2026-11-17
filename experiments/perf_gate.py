"""Performance gate — fail CI on >20% regression vs recorded baseline.

Usage:
    python experiments/perf_gate.py --baseline experiments/PERF_BASELINE.json \
        --current results.jsonl --threshold 0.20

Exit code 0 = pass, 1 = fail. Prints a report to stdout.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Optional


METRICS = ["build_s", "load_s", "exact_ms", "fuzzy_hit_ms", "fuzzy_miss_ms"]


def check(baseline: dict, current: dict, threshold: float = 0.20) -> tuple[bool, str]:
    """Compare current results against baseline. Returns (pass, report)."""
    regressions = []
    lines = ["Performance gate report:", f"  threshold: {threshold:.0%}"]
    for m in METRICS:
        if m not in baseline or m not in current:
            continue
        b = baseline[m]
        c = current[m]
        if b == 0:
            pct = 0.0
        else:
            pct = (c - b) / b
        status = "OK" if pct <= threshold else "REGRESSION"
        if pct > threshold:
            regressions.append(m)
        lines.append(f"  {m}: {b:.3f} -> {c:.3f} ({pct:+.1%}) {status}")
    ok = len(regressions) == 0
    if not ok:
        lines.append(f"\nFAILED: regressions in {', '.join(regressions)}")
    else:
        lines.append("\nPASSED: no regressions above threshold")
    return ok, "\n".join(lines)


def check_median(baseline: dict, currents: list[dict], threshold: float = 0.20) -> tuple[bool, str]:
    """Compare median of multiple current runs against baseline."""
    median_current = {}
    for m in METRICS:
        vals = [c[m] for c in currents if m in c]
        if vals:
            median_current[m] = statistics.median(vals)
    return check(baseline, median_current, threshold)


def load_jsonl(path: str) -> list[dict]:
    """Load JSONL results file into list of dicts."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="SQAC performance gate")
    ap.add_argument("--baseline", required=True, help="baseline JSON file")
    ap.add_argument("--current", required=True, help="current results JSONL file")
    ap.add_argument("--threshold", type=float, default=0.20, help="regression threshold (default 0.20)")
    args = ap.parse_args(argv)

    with open(args.baseline, encoding="utf-8") as f:
        baseline = json.load(f)

    rows = load_jsonl(args.current)
    if not rows:
        print("ERROR: no results in current file")
        return 1

    # Use the last row (most recent run) or median if multiple
    if len(rows) == 1:
        ok, report = check(baseline, rows[0], args.threshold)
    else:
        ok, report = check_median(baseline, rows, args.threshold)

    print(report)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
