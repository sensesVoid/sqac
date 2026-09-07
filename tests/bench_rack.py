#!/usr/bin/env python3
"""Whole-system stress test: the SQAC memory lifecycle at scale.

This is NOT a unit test of rack.py or the graduation pass in isolation. It
runs the entire system as one machine, end to end, at growing size:

    teach (facts/docs routed by kind + skills packed via the real skill
           pipeline)
        -> live session (ContextOffloader: rolling window, evictions)
        -> graduate (session bucket -> durable facts cartridge in the rack)
        -> recall (exact / fuzzy / grouped / back-reference)
        -> persist (save everything) -> reload (fresh rack + offloader)
        -> re-query from disk

and, at EVERY scale, verifies correctness AND measures latency.

Honest expectations (this repo's design, README-disclosed):
  FLAT   write(teach), session observe/offload, exact recall, per-exchange
         graduation cost — all O(1) per op.
  LINEAR fuzzy/semantic scans and rack search are O(n): the lexical tier is
         an XOR+popcount scan over every stored entry, and the rack cannot
         know which cartridge holds the answer, so an exact hit in one file
         still pays full scans on the others. Graduation is one linear pass
         over source exchanges + target facts. Cartridge bytes -> file size.

The script CLASSIFIES each op from the measured growth ratio and FAILS
(exit 1) if any correctness gate breaks at scale.

    python tests/bench_rack.py              # default sweep (semantic if model)
    python tests/bench_rack.py --no-semantic
    python tests/bench_rack.py --big        # push toward 100K entries (lexical)
    python tests/bench_rack.py --quick
    python tests/bench_rack.py --scales 1000,5000,10000,25000,50000

Run from the project root so `sqac.models` resolves.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import random
import re
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqac.offloader import ContextOffloader
from sqac.rack import CartridgeRack
from sqac.skills import load_skills, pack_cartridge, validate
from sqac.static_encoder import StaticSimHashEncoder

FAIL: list[str] = []

SEED = 12345


def check(gate_ok: bool, msg: str) -> None:
    """Record a correctness-gate result; failures are collected for exit 1."""
    if not gate_ok:
        FAIL.append(msg)
    print(f"      {'ok  ' if gate_ok else 'FAIL'} {msg}")


def classify(ratio: float, expected: str) -> str:
    """Label a measured growth ratio vs its design expectation."""
    if expected == "flat":
        return "FLAT" if ratio <= 2.0 else f"NOT-FLAT (x{ratio:.1f})"
    return f"LINEAR ~x{ratio:.1f}"


# ── synthetic domain generation ────────────────────────────────────────────
# Markers are REAL tokens drawn from the model's own vocabulary (seeded, so
# the sweep is reproducible). Each synthetic item embeds its integer id into
# its key AND a per-item word triple, so no two items ever share an exact
# lookup key and every back-reference probe carries distinctive content.

def _word_pool(min_size: int = 9000) -> list[str]:
    enc = StaticSimHashEncoder()
    vocab = enc._weights[3]
    plain = [
        w for w in vocab.keys()
        if w[0].isalpha() and "##" not in w and len(w) >= 5 and w.isalpha()
        and w not in ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")
    ]
    rng = random.Random(SEED)
    seen: set[str] = set()
    pool: list[str] = []
    while len(pool) < min_size:
        w = plain[rng.randrange(len(plain))]
        if w not in seen:
            seen.add(w)
            pool.append(w)
    L = len(pool)
    while L > 16 and (math.gcd(7, L) != 1 or math.gcd(13, L) != 1):
        L -= 1
    return pool[:L]


WORDS = _word_pool()
L = len(WORDS)


def rule_words(i: int) -> tuple[str, str, str]:
    return WORDS[i % L], WORDS[(i * 7) % L], WORDS[(i * 13) % L]


def make_fact(i: int) -> tuple[str, str]:
    w1, w2, w3 = rule_words(i)
    content = (
        f"Rule {i}: the {w1} {w2} {w3} release procedure requires review "
        f"step {i % 97} and signoff from owner-{i % 50}."
    )
    key = f"{w1} {w2} {w3} rule {i}"
    return content, key


def make_doc(i: int) -> tuple[str, str]:
    w1, w2, w3 = rule_words(i)
    content = (
        f"Doc {i}: {w1}-{w2} playbook covers the {w3} migration ticket "
        f"{i % 113} with rollback checklist and runbook links."
    )
    key = f"{w1} {w2} {w3} doc {i}"
    return content, key


def make_skill(i: int) -> dict:
    w1, w2, w3 = rule_words(i)
    return {
        "name": f"skill-{i}",
        "domain": "logic" if i % 2 else "debugging",
        "difficulty": "medium",
        "pattern": f"{w1}-{w2}-{w3}-pattern",
        "content": (
            f"SKILL skill-{i}: to decompose a {w1} {w2} problem with a {w3} "
            f"dependency, separate the inputs, label each component {i % 7}, "
            f"and inspect the imbalance to find the defective unit. "
            f"Verify by re-folding the {w2} path."
        ),
        "keys": [
            f"when you have {w1} {w2} items in task {i} and one batch is heavier",
            f"bags of {w3} material from task {i} where exactly one unit is defective",
            f"{w1} {w2} {w3} task {i} detect the single bad component by weighing",
            f"split the {w3} inputs of task {i} and compare against the {w2} baseline",
            f"find one defective {w2} among many identical {w1} samples in task {i}",
        ],
    }


# ── session (the real ContextOffloader) ────────────────────────────────────

def session_exchange(i: int) -> tuple[str, str]:
    """One salient exchange whose answer carries unique markers."""
    w1, w2, w3 = rule_words(i)
    question = f"what was the cause of the inc-{i} {w1} {w3} failure we saw earlier?"
    answer = (
        f"The inc-{i} {w1} {w3} failure is a 504 timeout after {300 + i % 97}s in "
        f"the {w1} {w2} layer; root cause: {w3} service slowed; "
        f"decision: bump the {w1} step timeout to 600s."
    )
    return question, answer


def session_blocks(n_exchanges: int) -> list[tuple[str, str]]:
    """Salient exchanges with unique markers + recurring chitchat.

    5 of every 6 exchanges carry distinctive error/decision markers (will
    graduate); 1 in 6 is pure chitchat (salience ~0, must NOT graduate).
    """
    turns: list[tuple[str, str]] = []
    for i in range(n_exchanges):
        if i % 6 == 5:
            turns.append(("user", "hey, how are you?"))
            turns.append(("assistant", "Doing great, ready to help!"))
            continue
        q, a = session_exchange(i)
        turns.append(("user", q))
        turns.append(("assistant", a))
    return turns


def run_session(off: ContextOffloader, n_exchanges: int) -> tuple[int, float]:
    evictions = 0
    t0 = time.perf_counter()
    for role, text in session_blocks(n_exchanges):
        if off.observe(role, text) is not None:
            evictions += 1
    off.offload()
    per_turn_ms = (time.perf_counter() - t0) * 1000.0 / max(1, n_exchanges * 2)
    return evictions, per_turn_ms


def session_probe(i: int) -> str:
    """A back-reference that shares DISTINCTIVE content with exchange i."""
    w1, _, w3 = rule_words(i)
    return f"what was the cause of that {w1} {w3} outage we discussed?"


# ── scale stages ───────────────────────────────────────────────────────────

def stage_teach(tmp: str, n_facts: int, n_docs: int, n_skills: int, semantic: bool):
    """Fill three cartridges: facts+docs via kind-routing, skills via the
    real YAML -> validate -> pack pipeline. Returns (rack, stats)."""
    rack = CartridgeRack(
        tmp,
        routes={"fact": "facts", "doc": "docs", "skill": "skills"},
        default="facts",
        semantic=semantic,
    )
    rack.create("facts", description="stress facts")
    rack.create("docs", description="stress docs")

    t0 = time.perf_counter()
    for i in range(n_facts):
        c, k = make_fact(i)
        rack.write_routed(c, key=k, kind="fact")
    for i in range(n_docs):
        c, k = make_doc(i)
        rack.write_routed(c, key=k, kind="doc")
    teach_ms = (time.perf_counter() - t0) * 1000.0 / max(1, n_facts + n_docs)

    t0 = time.perf_counter()
    skill_stats = {"n": n_skills, "errors": 0, "n_entries": 0}
    if n_skills:
        skills = [make_skill(i) for i in range(n_skills)]
        sf = os.path.join(tmp, "skills.json")
        with open(sf, "w") as j:
            json.dump(skills, j, ensure_ascii=False)
        objs = load_skills(sf)
        issues = validate(objs)
        errors = [i for i in issues if i.severity == "error"]
        skill_stats["errors"] = len(errors)
        if not errors:
            pack = os.path.join(tmp, "skills.sqac")
            with contextlib.redirect_stdout(io.StringIO()):
                pack_cartridge(objs, pack, name="stress-skills", semantic=semantic)
            rack.register("skills", path=pack)
            skill_stats["n_entries"] = sum(len(s.keys) for s in objs)
        else:
            FAIL.append(f"skill validation failed with {len(errors)} errors")
    pack_ms = (time.perf_counter() - t0) * 1000.0 / max(1, n_skills)

    return rack, {
        "teach_ms": round(teach_ms, 4),
        "pack_ms": round(pack_ms, 2),
        "n_facts": n_facts,
        "n_docs": n_docs,
        "n_skills": n_skills,
        "sizes": _cartridge_sizes(tmp, ["facts", "docs", "skills"]),
    }


def _cartridge_sizes(tmp: str, names: list[str]) -> dict[str, float]:
    out = {}
    for n in names:
        p = Path(tmp) / f"{n}.sqac"
        out[n] = round(p.stat().st_size / 1e6, 3) if p.exists() else 0.0
    return out


def recall_gates(rack: CartridgeRack, off: ContextOffloader,
                 n_facts: int, n_skills: int, strict: bool) -> dict:
    """Correctness gates + latency measurements on the whole rack + session."""
    # exact: sampled keys, expect conf 1.0 + correct content at every scale.
    # Two paths measured: rack-wide (scans sibling cartridges = linear) and
    # kind-hinted (O(1) dict hit in the right cartridge). The correctness
    # gate uses the rack-wide call: a caller that cannot name the cartridge
    # still gets the right answer, at the cross-cartridge scan cost.
    step = max(1, n_facts // 200)
    samples = list(range(0, n_facts, step))
    keys = [make_fact(i)[1] for i in samples]
    _warm(rack, keys[:2])
    exact_correct = 0
    t0 = time.perf_counter()
    for k in keys:
        hits = rack.search(k, top_k=1)
        i = int(re.search(r"rule (\d+)", k).group(1))
        exact_correct += bool(hits and hits[0].confidence == 1.0 and f"Rule {i}" in hits[0].content)
    exact_ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(keys))

    t0 = time.perf_counter()
    for k in keys:
        rack.search(k, top_k=1, kind="fact")
    exact_kind_ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(keys))
    check(len(samples) > 0 and exact_correct == len(keys),
          f"exact recall {exact_correct}/{len(keys)} (conf 1.0, correct content)")

    # fuzzy paraphrase probes (reversed-word shared vocab) — per-op timing
    fstep = max(1, n_facts // 25)
    probes = [f"{' '.join(make_fact(i)[1].split()[::-1])}" for i in range(0, n_facts, fstep)]
    _warm(rack, probes[:2])
    t0 = time.perf_counter()
    for q in probes:
        rack.search(q, top_k=1)
    fuzzy_ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(probes))

    # session back-reference recall (real offloader memory)
    ref_correct = ref_total = 0
    cands = [i for i in range(30) if i % 6 != 5][:20]
    for i in cands:
        hits = off.recall_detailed(session_probe(i), top_k=1)
        ref_total += 1
        m = re.search(r"inc-(\d+)\b", hits[0]["content"].lower()) if hits else None
        ref_correct += bool(m and int(m.group(1)) == i)
    if strict:
        check(ref_total > 0 and ref_correct == ref_total,
              f"back-reference recall {ref_correct}/{ref_total} (top-1 right exchange)")
    else:
        print(f"      info back-reference recall {ref_correct}/{ref_total} "
              f"(no semantic tier; not gated)")

    # skill grouped retrieval: top-1 group == the intended skill
    if n_skills > 0:
        g_correct = g_total = 0
        for i in range(0, n_skills, max(1, n_skills // 20)):
            probe = make_skill(i)["keys"][1]
            hits = rack.search_grouped(probe, group_key="skill", recall=40, kind="skill")
            g_total += 1
            g_correct += bool(hits and hits[0].meta.get("skill") == f"skill-{i}")
        check(g_total > 0 and g_correct == g_total,
              f"skill grouped top-1 {g_correct}/{g_total}")
    else:
        print("      info no skills at this scale")

    # fail-safe: garbage query must NOT yield a confident hit, at scale
    junk = [h for h in rack.search("zucchini flavored quantum unicorn telephony")]
    check(not junk, "fail-safe: garbage query -> no confident hit")

    return {"exact_ms": round(exact_ms, 4), "exact_kind_ms": round(exact_kind_ms, 4),
            "exact_ok": exact_correct, "exact_n": len(keys),
            "fuzzy_ms": round(fuzzy_ms, 2),
            "ref": ref_correct, "ref_n": ref_total, "g_ok": g_correct, "g_n": g_total}


def _warm(rack: CartridgeRack, queries: list[str]) -> None:
    """Prime numpy bitmap caches so first-query timing isn't startup bias."""
    for q in queries:
        if queries:
            rack.search(q, top_k=1)


def stage_graduate(rack: CartridgeRack, off: ContextOffloader,
                   n_exchanges: int, strict: bool) -> dict:
    t0 = time.perf_counter()
    r1 = rack.graduate(off, target_name="facts", threshold=0.40)
    pass_ms = (time.perf_counter() - t0) * 1000.0
    check(r1["promoted"] > 0, f"graduation promotes {r1['promoted']} salient exchanges")

    t0 = time.perf_counter()
    r2 = rack.graduate(off, target_name="facts", threshold=0.40)
    rerun_ms = (time.perf_counter() - t0) * 1000.0
    check(r2["promoted"] == 0, "graduation rerun idempotent (2nd pass promotes 0)")

    # graded facts are searchable rack-wide under kind=fact
    probe_idx = [i for i in range(20) if i % 6 != 5][:10]
    found = 0
    for i in probe_idx:
        w1, _, w3 = rule_words(i)
        hits = rack.search(f"inc-{i} {w1} {w3} 504 timeout", top_k=1, kind="fact")
        if hits and "504" in hits[0].content and f"inc-{i}" in hits[0].content \
                and w1 in hits[0].content:
            found += 1
    msg = f"graduated facts searchable rack-wide (kind=fact) {found}/{len(probe_idx)}"
    if strict:
        check(found == len(probe_idx), msg)
    else:
        print(f"      info {msg} (no semantic tier; not gated)")

    return {"pass_ms": round(pass_ms, 2), "rerun_ms": round(rerun_ms, 2),
            "promoted": r1["promoted"]}


def stage_persist(rack: CartridgeRack, off: ContextOffloader, tmp: Path,
                  semantic: bool) -> None:
    t0 = time.perf_counter()
    rack.save()
    off.save()
    save_ms = (time.perf_counter() - t0) * 1000.0

    # reload into a FRESH rack + fresh offloader, then re-answer from disk
    fresh = CartridgeRack(tmp,
                          routes={"fact": "facts", "doc": "docs", "skill": "skills"},
                          default="facts", semantic=semantic)
    for n in ("facts", "docs", "skills"):
        p = tmp / f"{n}.sqac"
        if p.exists():
            fresh.register(n)
    fresh_off = ContextOffloader(tmp / "session.sqac", window=8, semantic=semantic)

    _, k0 = make_fact(0)
    hits = fresh.search(k0, top_k=1)
    w1, _, w3 = rule_words(0)
    back = fresh_off.recall_detailed(session_probe(0), top_k=1)
    survived = bool(hits and hits[0].confidence == 1.0 and "Rule 0" in hits[0].content)
    back_ok = bool(back and re.search(r"inc-0\b", back[0]["content"].lower()) and w1 in back[0]["content"])
    check(survived, f"fact survives save/reload and re-answers from disk ({save_ms:.0f}ms save)")
    check(back_ok, "session bucket survives save/reload (back-reference still answers)")


# ── runner ─────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="whole-system SQAC stress test")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--big", action="store_true")
    ap.add_argument("--no-semantic", action="store_true")
    ap.add_argument("--exchanges", type=int, default=2000,
                    help="session exchanges at the smallest scale (scaled up)")
    ap.add_argument("--scales", type=str, default="",
                    help="explicit comma-separated scale list (overrides presets)")
    args = ap.parse_args()

    semantic = bool(StaticSimHashEncoder.available()) and not args.no_semantic
    strict = semantic
    if args.no_semantic:
        print("semantic tier disabled (--no-semantic); lexical+exact only; "
              "back-reference/graduation-search gates become informational\n")
    elif not semantic:
        print("WARNING: static semantic model unavailable; semantic tier off\n")

    if args.scales:
        scales = [int(s) for s in args.scales.split(",") if s.strip()]
    elif args.quick:
        scales = [1000, 5000]
    elif args.big:
        scales = [25000, 50000, 100000]
        if not args.no_semantic:
            print("--big forces lexical-only (semantic 25K+ would take ~45min)\n")
            semantic = False
            strict = False
    else:
        scales = [1000, 5000, 10000, 25000] if semantic else [1000, 5000, 10000, 25000, 50000]

    ex_cap = 4000 if args.big else 6000
    print(f"word pool: {len(WORDS)} unique in-vocab marker tokens (seed {SEED}), "
          f"{'semantic' if semantic else 'lexical+exact'} tier")
    print("SQAC WHOLE-SYSTEM STRESS — every scale runs the full lifecycle:")
    print("facts/docs kind-routed -> skills yaml-validated+packed -> live")
    print("offloader session -> graduation -> recall -> persist -> reload\n")
    print(f"{'scale':>9} | {'teach/op':>9} {'sess/op':>8} {'exact':>7} {'fuzzy':>8} "
          f"{'backref':>9} {'grad ms':>8} {'promo':>7} {'rerun':>7}")
    print("-" * 118)

    first: dict = {}
    last: dict = {}
    for n in scales:
        n_facts = n * 2 // 3
        n_docs = n - n_facts
        n_skills = max(1, n // 40)
        n_exch = min(max(args.exchanges, (n_facts + n_docs) // 4), ex_cap)
        with tempfile.TemporaryDirectory(prefix="sqac_stress_") as td:
            t0 = time.perf_counter()
            tmp = Path(td)

            rack, teach = stage_teach(str(tmp), n_facts, n_docs, n_skills, semantic)

            off = ContextOffloader(tmp / "session.sqac", window=8, semantic=semantic)
            evicted, sess_per_ms = run_session(off, n_exch)
            n_entries_bucket = off.stats()["entries"]

            rec = recall_gates(rack, off, n_facts, n_skills, strict)
            grad = stage_graduate(rack, off, n_exch, strict)
            stage_persist(rack, off, tmp, semantic)
            wall = time.perf_counter() - t0

            last = {"teach": teach["teach_ms"], "sess": sess_per_ms,
                    "exact": rec["exact_ms"], "exact_kind": rec["exact_kind_ms"],
                    "fuzzy": rec["fuzzy_ms"], "grad": grad["pass_ms"],
                    "promo": grad["promoted"], "bytes_mb": sum(teach["sizes"].values())}
            if not first:
                first = dict(last)

            print(f"{n:>8,} | {teach['teach_ms']:>8.3f} {sess_per_ms:>7.3f} "
                  f"{rec['exact_ms']:>6.3f} {rec['fuzzy_ms']:>7.2f} "
                  f"{rec['ref']}/{rec['ref_n']:>8}  {grad['pass_ms']:>7.1f} "
                  f"{grad['promoted']:>6,} {grad['rerun_ms']:>6.1f}")
        print(f"      -> lives: {n_facts + n_docs:,} facts+docs, {n_skills:,} skills | "
              f"session: {n_exch:,} exchanges -> {n_entries_bucket:,} bucket entries "
              f"({evicted} window-evicted) | wall {wall:.0f}s | disk: "
              + ", ".join(f"{k}={v}MB" for k, v in teach["sizes"].items()))

    # ── verdicts: measured growth ratio from first -> last scale ──────────
    def grow(k):
        a, b = first.get(k), last.get(k)
        return (b / a) if (a and b) else float("nan")

    print("\n=== verdict (measured ratio, first -> last scale) ===")
    px_first = first["grad"] / max(1, first["promo"])
    px_last = last["grad"] / max(1, last["promo"])
    rows = [
        ("teach write/op", grow("teach"), "flat", "O(1) per fact/doc"),
        ("session observe/offload /op", grow("sess"), "flat", "O(1) per turn"),
        ("exact recall /op (rack-wide)", grow("exact"), "linear",
         "no kind hint: pays cross-cartridge scans, then dict hit"),
        ("exact recall /op (kind-hinted)", grow("exact_kind"), "flat",
         "O(1) dict hit in the right cartridge"),
        ("fuzzy scan /op", grow("fuzzy"), "linear", "O(n) XOR+popcount, README-disclosed"),
        ("graduation (total)", grow("grad"), "linear", "one pass: source exchanges + target facts"),
        ("graduation per-exchange", px_last / px_first, "flat",
         f"{px_first:.2f} -> {px_last:.2f} ms/promoted"),
        ("cartridge bytes", grow("bytes_mb"), "linear", "~B/fact + payloads, expected"),
    ]
    for name, r, exp, note in rows:
        print(f"  {name:28} {r:6.2f}x  {classify(r, exp):14}  {note}")

    print(f"\nlargest scale: {scales[-1]:,} durable facts+docs, "
          f"{last['bytes_mb']:.1f} MB on disk across cartridges, "
          f"session up to {ex_cap:,} exchanges")

    if FAIL:
        print("\nSTRESS TEST FAILED — correctness gates broken at the last scale:")
        for f in FAIL[-12:]:
            print(f"  - {f}")
        print(f"  ({len(FAIL)} gate failures recorded in total)")
        return 1
    print("\nSTRESS TEST PASSED — all correctness gates held at every scale")
    return 0


if __name__ == "__main__":
    sys.exit(main())