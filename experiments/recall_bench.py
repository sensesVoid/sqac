"""recall_bench.py — per-tier retrieval diagnosis (PR 3).

Report-first recall benchmark for the SQAC retrieval stack:

- Seeded synthetic multilingual corpora (English + Spanish) with paraphrase
  pairs and code identifiers, each entry carrying a ground-truth key.
- Per-tier metrics vs ground truth: exact / fuzzy / semantic recall@1/3/5,
  confidence distributions, and the gate decision (would `search_or_none`
  with the default threshold answer?).
- Maintainer real-query cases live in ``CASES`` at the bottom — add failed
  queries here and re-run; each case names the ground-truth entry and the
  query class it belongs to.
- Output: a triage table (which tier fails which query class) as JSON +
  Markdown. There is deliberately **no pass/fail bar** (S7) — the table is
  the deliverable; follow-up fix PRs are informed by it.

Usage:
    python experiments/recall_bench.py                 # full report
    python experiments/recall_bench.py --json out.json # also write JSON
    python experiments/recall_bench.py --semantic      # enable semantic tier
    python experiments/recall_bench.py --seed 42 --n 200

Hermetic by default: ``SQAC_SEMANTIC=0`` (lexical + exact only, no model
downloads); pass ``--semantic`` to measure the semantic tier too.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Ensure the repo root is importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("SQAC_SEMANTIC", "0")

from sqac.store import SqacStore  # noqa: E402

DEFAULT_SEED = 20261123
DEFAULT_N = 200          # entries per corpus
DEFAULT_THRESHOLD = 0.60  # mirrors SqacStore.search_or_none default gate


# ── corpora ──────────────────────────────────────────────────────────────────

# Each corpus: (name, language, list of (content, key, query, query_class)).
# `key` is the ground-truth key used to grade the hit; `query` is what the
# bench asks; paraphrase queries deliberately differ in wording from `content`.

_EN_FACTS = [
    ("Team Atlas maintains the payments service", "payments ownership",
     "who owns the payments service", "factual"),
    ("Deployments target ARM64 only", "deploy target arch",
     "what architecture do we deploy to", "factual"),
    ("The staging cluster runs Kubernetes 1.29", "staging k8s version",
     "which kubernetes version is on staging", "factual"),
    ("Primary database is PostgreSQL 15 with streaming replication", "primary db engine",
     "what database do we run in production", "factual"),
    ("Feature flags are managed by Unleash", "feature flag tool",
     "which tool controls feature flags", "factual"),
    ("API gateway is Kong, deployed per region", "api gateway product",
     "what api gateway do we use", "factual"),
    ("CI runs on GitHub Actions with self-hosted runners", "ci provider",
     "where does continuous integration run", "factual"),
    ("Observability stack: Prometheus, Grafana, Loki", "observability stack",
     "what tools do we use for metrics and logs", "factual"),
    ("Secrets live in HashiCorp Vault with AWS KMS auto-unseal", "secrets backend",
     "where are application secrets stored", "factual"),
    ("The monorepo uses pnpm workspaces with Turborepo caching", "monorepo tooling",
     "how is the monorepo built and cached", "factual"),
    ("On-call rotation is weekly, handoff Mondays 10:00 UTC", "oncall handoff",
     "when does the on-call rotation hand over", "procedural"),
    ("Rollback procedure: redeploy the previous image tag via ArgoCD", "rollback procedure",
     "how do we roll back a bad deploy", "procedural"),
    ("Incident severity levels: SEV1 pages immediately, SEV3 next business day", "severity policy",
     "what are the incident severity levels", "procedural"),
    ("Database migrations run via Flyway before app rollout", "migration tool",
     "how are schema migrations applied", "procedural"),
    ("New services must register in the service catalog within one sprint", "service registration",
     "what must a new service do first", "procedural"),
    ("Postmortems are blameless and published within 48 hours", "postmortem policy",
     "what is the postmortem policy", "procedural"),
    ("Change freeze from Dec 20 to Jan 5 except SEV1 fixes", "change freeze window",
     "when is the holiday change freeze", "procedural"),
    ("Load tests use k6 in the perf environment before launch", "load testing tool",
     "which tool runs load tests", "procedural"),
    ("Access reviews happen quarterly via the IAM dashboard", "access review cadence",
     "how often are access rights reviewed", "procedural"),
    ("Disaster recovery drill is simulated twice a year in us-east-2", "dr drill",
     "how often do we test disaster recovery", "procedural"),
]

_ES_FACTS = [
    ("El equipo Atlas mantiene el servicio de pagos", "pagos dueno",
     "quien es el dueno del servicio de pagos", "factual"),
    ("Los despliegues son solo para ARM64", "arquitectura despliegue",
     "a que arquitectura desplegamos", "factual"),
    ("El cluster de staging ejecuta Kubernetes 1.29", "k8s version staging",
     "que version de kubernetes hay en staging", "factual"),
    ("La base de datos principal es PostgreSQL 15", "motor base de datos",
     "que base de datos usamos en produccion", "factual"),
    ("Los secretos viven en HashiCorp Vault", "backend secretos",
     "donde se guardan los secretos de la aplicacion", "factual"),
    ("El pipeline de CI corre en GitHub Actions", "proveedor ci",
     "donde corre la integracion continua", "factual"),
    ("La rotacion de guardia es semanal, cambio los lunes", "guardia relevo",
     "cuando se entrega la guardia", "procedural"),
    ("El procedimiento de rollback es redesplegar la imagen anterior", "rollback procedimiento",
     "como revertimos un despliegue defectuoso", "procedural"),
    ("Las migraciones se aplican con Flyway antes del rollout", "herramienta migraciones",
     "como se aplican las migraciones de esquema", "procedural"),
    ("El postmortem es sin culpables y se publica en 48 horas", "politica postmortem",
     "cual es la politica de postmortems", "procedural"),
    ("La congelacion de cambios es del 20 de diciembre al 5 de enero", "ventana congelacion",
     "cuando es la congelacion de cambios de fin de ano", "procedural"),
    ("Las pruebas de carga usan k6 en el entorno de rendimiento", "herramienta pruebas carga",
     "que herramienta ejecuta las pruebas de carga", "procedural"),
]

_CODE_IDS = [
    ("Class PaymentService retries failed charges three times with backoff", "PaymentService",
     "PaymentService retry behavior", "code"),
    ("def recalculate_invoice_totals(invoice) sums line items plus tax", "recalculate_invoice_totals",
     "recalculate_invoice_totals definition", "code"),
    ("The AuthMiddleware validates JWTs and refreshes expired tokens", "AuthMiddleware",
     "AuthMiddleware purpose", "code"),
    ("Config loader reads YAML from the config/ directory with env overrides", "load_config",
     "load_config behavior", "code"),
    ("RateLimiter uses a token bucket per API key, 100 rps burst 200", "RateLimiter",
     "RateLimiter algorithm and limits", "code"),
    ("Migration 0042 adds a unique index on users.email", "migration_0042",
     "migration 0042 change", "code"),
    ("Function parse_retry_after parses HTTP date or delta-seconds headers", "parse_retry_after",
     "parse_retry_after supported formats", "code"),
    ("The Saga coordinator compensates completed steps on failure", "SagaCoordinator",
     "SagaCoordinator compensation", "code"),
    ("HealthCheckHandler pings dependencies and aggregates status codes", "HealthCheckHandler",
     "HealthCheckHandler checks", "code"),
    ("build_docker_image tags images with git SHA and build timestamp", "build_docker_image",
     "build_docker_image tagging", "code"),
    ("CircuitBreaker opens after 50% failures over 20 requests", "CircuitBreaker",
     "CircuitBreaker open condition", "code"),
    ("The FeatureFlagEvaluator falls back to a default variant when the flag is missing", "FeatureFlagEvaluator",
     "FeatureFlagEvaluator default behavior", "code"),
    ("parse_duration accepts Go-style strings like 1h30m and 500ms", "parse_duration",
     "parse_duration accepted units", "code"),
    ("UserRepository caches lookups in Redis with a 60s TTL", "UserRepository",
     "UserRepository caching", "code"),
    ("The WebhookDispatcher retries with exponential backoff up to 5 attempts", "WebhookDispatcher",
     "WebhookDispatcher retry policy", "code"),
]


@dataclass
class Corpus:
    name: str
    language: str
    entries: list[tuple[str, str, str, str]] = field(default_factory=list)


def build_corpora(rng: random.Random, n: int) -> list[Corpus]:
    """Seeded corpora. The curated entries come first; when `n` exceeds the
    curated size, synthetic filler is generated by recombining templates so
    the store grows without changing the graded set."""
    corpora = [
        Corpus("en-facts", "en", list(_EN_FACTS)),
        Corpus("es-facts", "es", list(_ES_FACTS)),
        Corpus("code-ids", "code", list(_CODE_IDS)),
    ]
    templates = [
        ("Service {name} exposes a {protocol} API on port {port}", "service {name} api",
         "service {name} protocol and port", "synthetic"),
        ("The {team} team owns the {component} component", "owner of {component}",
         "who owns the {component}", "synthetic"),
        ("Alert {alert} fires when {metric} exceeds {threshold}", "alert {alert} condition",
         "when does alert {alert} fire", "synthetic"),
    ]
    names = ["billing", "catalog", "checkout", "fulfillment", "gateway", "identity",
             "inventory", "ledger", "notify", "search", "shipping", "tax"]
    teams = ["Atlas", "Nova", "Orion", "Lyra", "Vega", "Rigel"]
    components = ["auth", "router", "scheduler", "exporter", "indexer", "cache"]
    protocols = ["REST", "gRPC", "GraphQL", "WebSocket"]
    ports = [8080, 8443, 9090, 9091, 9200]
    alerts = ["HighLatency", "QueueBacklog", "DiskPressure", "CertExpiry", "ErrorSpike"]
    metrics = ["p99 latency", "queue depth", "disk usage", "error rate", "memory"]
    thresholds = ["500ms", "10k", "85%", "1%", "2GB"]

    for ci, corpus in enumerate(corpora):
        while len(corpus.entries) < n:
            t = rng.choice(templates)
            if t[0].startswith("Service"):
                name = rng.choice(names)
                corpus.entries.append((
                    t[0].format(name=name, protocol=rng.choice(protocols), port=rng.choice(ports)),
                    t[1].format(name=name),
                    t[2].format(name=name),
                    t[3],
                ))
            elif t[0].startswith("The"):
                team, comp = rng.choice(teams), rng.choice(components)
                corpus.entries.append((
                    t[0].format(team=team, component=comp),
                    t[1].format(component=comp),
                    t[2].format(component=comp),
                    t[3],
                ))
            else:
                alert, metric, thr = rng.choice(alerts), rng.choice(metrics), rng.choice(thresholds)
                corpus.entries.append((
                    t[0].format(alert=alert, metric=metric, threshold=thr),
                    t[1].format(alert=alert),
                    t[2].format(alert=alert),
                    t[3],
                ))
    return corpora


# ── bench core ───────────────────────────────────────────────────────────────

@dataclass
class TierStats:
    name: str
    hits: int = 0
    graded: int = 0          # queries with a ground-truth key
    recall1: int = 0
    recall3: int = 0
    recall5: int = 0
    gate_pass: int = 0
    gate_total: int = 0
    confs: list[float] = field(default_factory=list)

    def report(self) -> dict[str, Any]:
        r = lambda a, b: (a / b) if b else 0.0
        return {
            "tier": self.name,
            "graded": self.graded,
            "gate_total": self.gate_total,
            "queries": self.graded,
            "hits": self.hits,
            "recall@1": round(r(self.recall1, self.graded), 3),
            "recall@3": round(r(self.recall3, self.graded), 3),
            "recall@5": round(r(self.recall5, self.graded), 3),
            "gate_pass": f"{self.gate_pass}/{self.gate_total}",
            "gate_rate": round(r(self.gate_pass, self.gate_total), 3),
            "conf_mean": round(statistics.fmean(self.confs), 3) if self.confs else 0.0,
            "conf_median": round(statistics.median(self.confs), 3) if self.confs else 0.0,
            "conf_min": round(min(self.confs), 3) if self.confs else 0.0,
            "conf_max": round(max(self.confs), 3) if self.confs else 0.0,
        }


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def run_corpus(store: SqacStore, corpus: Corpus, threshold: float,
               enrich: bool = False) -> dict[str, Any]:
    """Grade one corpus: for every entry, search its paraphrase query and
    check the ground-truth key is in top-k (by key_norm)."""
    tiers = {"exact": TierStats("exact"), "fuzzy": TierStats("fuzzy"),
             "semantic": TierStats("semantic")}
    # key_norm -> entry idx, from the store itself (what search dedupes on)
    key_to_idx = {}
    for i, e in enumerate(store._entries):
        if not e.get("deleted"):
            key_to_idx.setdefault(_norm(e["key_norm"]), i)

    failures: list[dict[str, str]] = []
    for content, key, query, qclass in corpus.entries:
        if not key.strip():
            continue  # zero-anchor keys are excluded by design
        gt_idx = key_to_idx.get(_norm(key))
        if gt_idx is None:
            failures.append({"case": "missing-ground-truth", "query": query, "class": qclass})
            continue
        hits = store.search(query, top_k=5)
        found_modes = {h.mode for h in hits}
        hit_keys = {_norm(h.meta.get("key_norm", "")) for h in hits}
        ok = _norm(key) in hit_keys
        # per-tier attribution: a hit's mode is the tier that produced it
        for t in ("exact", "fuzzy", "semantic"):
            ts = tiers[t]
            ts.graded += 1
            if t in found_modes:
                ts.hits += 1
                rank = next((r for r, h in enumerate(hits, 1) if h.mode == t), None)
                if rank == 1:
                    ts.recall1 += 1
                if rank and rank <= 3:
                    ts.recall3 += 1
                if rank and rank <= 5:
                    ts.recall5 += 1
        # gate decision: would search_or_none(threshold) return the right key?
        top = hits[0] if hits else None
        ts_gate_ok = bool(top and top.confidence >= threshold and _norm(top.meta.get("key_norm", "")) == _norm(key))
        for t in ("exact", "fuzzy", "semantic"):
            ts = tiers[t]
            ts.gate_total += 1
            if t in found_modes and ts_gate_ok:
                ts.gate_pass += 1
        if top:
            for t in ("exact", "fuzzy", "semantic"):
                if t in found_modes:
                    tiers[t].confs.append(top.confidence)
        if not ok:
            failures.append({"case": "miss", "query": query, "class": qclass,
                             "expected_key": key,
                             "got": [h.meta.get("key_norm", "")[:40] for h in hits[:3]]})
    return {
        "corpus": corpus.name,
        "language": corpus.language,
        "entries": len(corpus.entries),
        "enrich_keys": enrich,
        "tiers": {t: s.report() for t, s in tiers.items()},
        "failures": failures,
    }


def run_negatives(store: SqacStore, corpora: list[Corpus]) -> dict[str, Any]:
    """False-positive sweep: for every entry, the top hit against a query that
    does NOT belong to it. Tells us where the wrong-key confidence band sits,
    so a gate threshold can be chosen to separate right from wrong."""
    all_keys = {}
    for c in corpora:
        for _content, key, _q, _cl in c.entries:
            all_keys.setdefault(_norm(key), key)
    queries = []
    for c in corpora:
        for _content, _key, q, _cl in c.entries:
            queries.append((c.name, q))
    wrong_confs: list[float] = []
    wrong_hits = 0
    for corpus_name, q in queries:
        hits = store.search(q, top_k=1)
        if not hits:
            continue
        top = hits[0]
        wrong_confs.append(top.confidence)
        if top.meta.get("key_norm") not in all_keys:
            wrong_hits += 1
    wrong_confs.sort()
    return {
        "queries": len(queries),
        "queries_with_hits": len(wrong_confs),
        "top1_wrong_key": wrong_hits,
        "conf_mean": round(statistics.fmean(wrong_confs), 3) if wrong_confs else 0.0,
        "conf_median": round(statistics.median(wrong_confs), 3) if wrong_confs else 0.0,
        "conf_min": round(min(wrong_confs), 3) if wrong_confs else 0.0,
        "conf_max": round(max(wrong_confs), 3) if wrong_confs else 0.0,
        "p90": wrong_confs[int(len(wrong_confs) * 0.9)] if wrong_confs else 0.0,
        "p99": wrong_confs[min(len(wrong_confs) - 1, int(len(wrong_confs) * 0.99))] if wrong_confs else 0.0,
    }


def run_margin_probe(store: SqacStore, corpora: list[Corpus], gate: float,
                     margins: list[float]) -> list[dict[str, Any]]:
    """Prototype for the tier-rescue follow-up: when the top hit is below the
    gate, admit the next candidate anyway and accept it if it leads the
    runner-up by >= margin. Sweeps margin values to find the safe operating
    point (max correct rescues before any wrong-key acceptance)."""
    lowered = max(0.0, gate - 0.15)
    probes: list[dict[str, Any]] = []
    for margin in margins:
        recovered = false_accept = below = 0
        for c in corpora:
            for _content, key, q, _cl in c.entries:
                if not key.strip():
                    continue
                hits = store.search(q, top_k=2)
                if hits and hits[0].confidence >= gate:
                    continue
                below += 1
                wide = _search_at(store, q, lowered)
                if not wide:
                    continue
                lead = wide[0][0] - (wide[1][0] if len(wide) > 1 else 0.0)
                if lead >= margin:
                    if _norm(wide[0][1]) == _norm(key):
                        recovered += 1
                    else:
                        false_accept += 1
        probes.append({"min_margin": margin, "below_gate": below,
                       "recovered": recovered, "false_accept": false_accept})
    return probes


def _search_at(store: SqacStore, query: str, threshold: float) -> list[tuple[float, str]]:
    """Raw (sim, key_norm) candidates at a given threshold, bypassing the
    store's configured fuzzy threshold (bench-only helper)."""
    saved = store.fuzzy_threshold
    store.fuzzy_threshold = threshold
    try:
        hits = store.search(query, top_k=2)
    finally:
        store.fuzzy_threshold = saved
    return [(h.confidence, _norm(h.meta.get("key_norm", ""))) for h in hits]


def run_tier_fusion(store: SqacStore, corpora: list[Corpus], gate: float) -> dict[str, Any]:
    """Measure max-fusion (lexical ∪ semantic): a key is rescued if it is
    top-1 in ANY tier, with the fused confidence = max over tiers. This is
    the classic multi-view retrieval fix — report how many gate misses it
    recovers and what the fused confidence band looks like."""
    if not store.semantic:
        return {"skipped": "semantic tier disabled"}
    recovered = 0
    below = 0
    for c in corpora:
        for _content, key, q, _cl in c.entries:
            if not key.strip():
                continue
            hits = store.search(q, top_k=1)
            if hits and hits[0].confidence >= gate:
                continue
            below += 1
            lex = _search_at(store, q, max(0.0, gate - 0.15))
            sem = _search_at(store, q, max(0.0, gate - 0.15))
            if not lex or not sem:
                continue
            fused_top = max(lex[0][0], sem[0][0])
            if fused_top >= gate and (lex[0][1] == _norm(key) or sem[0][1] == _norm(key)):
                recovered += 1
    return {"below_gate": below, "recovered_by_fusion": recovered}


def run_maintainer_cases(store: SqacStore, threshold: float) -> list[dict[str, Any]]:
    """Maintainer real failed queries (S7). Each case: (query, expected key,
    query class). Add rows here when a real query misses; the bench grades
    them like any other case."""
    out = []
    for case in CASES:
        hits = store.search(case["query"], top_k=5)
        ok = any(_norm(h.meta.get("key_norm", "")) == _norm(case["expected_key"]) for h in hits)
        top = hits[0] if hits else None
        out.append({
            "case": "maintainer",
            "query": case["query"],
            "class": case["query_class"],
            "expected_key": case["expected_key"],
            "hit": ok,
            "top_key": (top.meta.get("key_norm", "")[:40] if top else ""),
            "top_conf": round(top.confidence, 3) if top else 0.0,
            "gated": bool(top and top.confidence >= threshold),
        })
    return out


# ── triage report ────────────────────────────────────────────────────────────

def triage(results: list[dict[str, Any]], maintainer: list[dict[str, Any]]) -> dict[str, Any]:
    """Which tier fails which query class (the deliverable)."""
    by_class: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "hit": 0})
    tier_class: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for res in results:
        for f in res["failures"]:
            if f["case"] != "miss":
                continue
            c = f["class"]
            by_class[c]["total"] += 1
            for t in ("exact", "fuzzy", "semantic"):
                tier_class[t][c] += 1
    rows = []
    for c, s in sorted(by_class.items()):
        rows.append({
            "query_class": c,
            "misses": s["total"],
            "fails_by_tier": {t: tier_class[t][c] for t in ("exact", "fuzzy", "semantic")},
        })
    missed_queries = [f for res in results for f in res["failures"] if f["case"] == "miss"]
    return {"by_query_class": rows, "missed_queries": missed_queries,
            "maintainer_cases": maintainer}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SQAC per-tier recall benchmark (PR 3)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="entries per corpus")
    ap.add_argument("--semantic", action="store_true",
                    help="enable the semantic tier (downloads the model)")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help="gate threshold for the confidence check")
    ap.add_argument("--json", metavar="PATH", default=None, help="write the JSON report")
    ap.add_argument("--md", metavar="PATH", default=None, help="write the Markdown report")
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    corpora = build_corpora(rng, args.n)
    store = SqacStore(semantic=args.semantic)

    results = []
    for corpus in corpora:
        # fresh store per corpus so cross-corpus hits don't pollute grading
        store = SqacStore(semantic=args.semantic)
        for content, key, _q, _c in corpus.entries:
            store.add(content, key=key, kind="fact" if corpus.language != "code" else "skill")
        results.append(run_corpus(store, corpus, args.threshold))
        print(f"  {corpus.name}: {len(corpus.entries)} entries graded", file=sys.stderr)

    # enriched-key variant: same corpora, fuzzy-tier vectors fold in content
    # anchors (SqacStore.add(enrich_keys=True)); exact tier unchanged.
    results_enriched = []
    for corpus in corpora:
        store = SqacStore(semantic=args.semantic)
        for content, key, _q, _c in corpus.entries:
            store.add(content, key=key, kind="fact" if corpus.language != "code" else "skill",
                      enrich_keys=True)
        results_enriched.append(run_corpus(store, corpus, args.threshold, enrich=True))
        print(f"  {corpus.name} (enriched): graded", file=sys.stderr)

    # maintainer cases run against a combined store (curated entries only)
    mstore = SqacStore(semantic=args.semantic)
    for corpus in corpora:
        for content, key, _q, _c in corpus.entries:
            if corpus.name in ("en-facts", "es-facts", "code-ids") and _c != "synthetic":
                mstore.add(content, key=key, kind="fact" if corpus.language != "code" else "skill")
    maintainer = run_maintainer_cases(mstore, args.threshold)
    negatives = run_negatives(mstore, corpora)
    margin = run_margin_probe(mstore, corpora, args.threshold, [0.02, 0.05, 0.08, 0.12])
    fusion = run_tier_fusion(mstore, corpora, args.threshold)

    report = {
        "meta": {
            "seed": args.seed,
            "entries_per_corpus": args.n,
            "semantic": args.semantic,
            "threshold": args.threshold,
            "sqac_version": __import__("sqac").__version__,
        },
        "results": results,
        "results_enriched": results_enriched,
        "triage": triage(results, maintainer),
        "triage_enriched": triage(results_enriched, maintainer),
        "negatives": negatives,
        "margin_probe": margin,
        "tier_fusion": fusion,
    }

    # ── console report ──
    print("\n=== Per-tier recall (top-5 hits, graded by ground-truth key) ===")
    print("note: 'exact' stays 0 by design — bench queries are paraphrases, not keys")
    print("columns: baseline vs enrich_keys=True (fuzzy vectors + content anchors)")
    for res, res_en in zip(results, results_enriched):
        print(f"\n[{res['corpus']}] {res['language']} — {res['entries']} entries")
        for t, s in res["tiers"].items():
            s_en = res_en["tiers"][t]
            print(f"  {t:8s} recall@1 {s['recall@1']:.3f} -> {s_en['recall@1']:.3f}  "
                  f"@3 {s['recall@3']:.3f} -> {s_en['recall@3']:.3f}  "
                  f"@5 {s['recall@5']:.3f} -> {s_en['recall@5']:.3f}  "
                  f"gate {s['gate_pass'].split('/')[0]}/{s['gate_total']} -> "
                  f"{s_en['gate_pass'].split('/')[0]}/{s_en['gate_total']}  "
                  f"conf med {s['conf_median']} -> {s_en['conf_median']}")
    print("\n=== Triage: misses by query class (which tier fails which query class) ===")
    for row in report["triage"]["by_query_class"]:
        print(f"  {row['query_class']:12s} misses={row['misses']:3d}  "
              f"fails_by_tier={row['fails_by_tier']}")
    if report["triage"].get("missed_queries"):
        print("\n=== Missed queries (top-3 closest key shown) ===")
        for f in report["triage"]["missed_queries"]:
            print(f"  [{f['class']:10s}] {f['query']!r} -> expected {f['expected_key']!r}, "
                  f"got {f.get('got')}")
    if maintainer:
        print("\n=== Maintainer real-query cases ===")
        for m in maintainer:
            print(f"  [{'HIT ' if m['hit'] else 'MISS'}] {m['class']:10s} {m['query']!r} "
                  f"(top={m['top_key']!r} conf={m['top_conf']} gated={m['gated']})")
    print("\n=== False-positive band (top-1 against all entries, incl. wrong keys) ===")
    print(f"  wrong-key top-1: {negatives['top1_wrong_key']}/{negatives['queries_with_hits']} "
          f"conf mean {negatives['conf_mean']} med {negatives['conf_median']} "
          f"p90 {negatives['p90']} p99 {negatives['p99']} max {negatives['conf_max']}")
    print("\n=== Margin-rescue sweep (below-gate queries, runner-up admitted) ===")
    for m in margin:
        print(f"  min_margin {m['min_margin']}: {m['recovered']}/{m['below_gate']} "
              f"recovered, {m['false_accept']} false accepts")
    print("\n=== Tier fusion (lexical + semantic, max-fused) ===")
    if fusion.get("skipped"):
        print(f"  skipped: {fusion['skipped']} (pass --semantic to measure)")
    else:
        print(f"  {fusion['recovered_by_fusion']}/{fusion['below_gate']} gate misses recovered")

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    if args.md:
        md = ["# Recall benchmark (PR 3)", "",
              f"- seed {args.seed}, n={args.n}/corpus, semantic={args.semantic}, "
              f"gate threshold {args.threshold}", ""]
        md.append("| corpus | tier | recall@1 | recall@3 | recall@5 | gate pass/total | conf med |")
        md.append("|---|---|---|---|---|---|---|")
        for res, res_en in zip(results, results_enriched):
            for t, s in res["tiers"].items():
                s_en = res_en["tiers"][t]
                gp = s["gate_pass"].split("/")[0]
                gp_en = s_en["gate_pass"].split("/")[0]
                md.append(f"| {res['corpus']} | {t} | {s['recall@1']} | {s['recall@3']} | "
                          f"{s['recall@5']} | {gp}/{s['gate_total']} | {s['conf_median']} |")
                md.append(f"| {res['corpus']} (enriched) | {t} | {s_en['recall@1']} | "
                          f"{s_en['recall@3']} | {s_en['recall@5']} | {gp_en}/{s_en['gate_total']} | "
                          f"{s_en['conf_median']} |")
        md.append("")
        md.append("## Triage — misses by query class")
        md.append("")
        md.append("| query class | misses | fails by tier |")
        md.append("|---|---|---|")
        for row in report["triage"]["by_query_class"]:
            md.append(f"| {row['query_class']} | {row['misses']} | {row['fails_by_tier']} |")
        if maintainer:
            md.append("")
            md.append("## Maintainer real-query cases")
            for m in maintainer:
                md.append(f"- [{'HIT' if m['hit'] else 'MISS'}] {m['class']}: {m['query']!r}")
        md.append("")
        md.append("## False-positive band (gate threshold evidence)")
        md.append("")
        md.append(f"- wrong-key top-1: {negatives['top1_wrong_key']}/{negatives['queries_with_hits']}, "
                  f"conf mean {negatives['conf_mean']}, med {negatives['conf_median']}, "
                  f"p90 {negatives['p90']}, p99 {negatives['p99']}, max {negatives['conf_max']}")
        Path(args.md).write_text("\n".join(md) + "\n", encoding="utf-8")
        print(f"wrote {args.md}")
    return 0


# Maintainer real-query cases (S7): (query, expected ground-truth key, class).
# Seed rows mirror the curated corpora; replace with real failed queries.
CASES: list[dict[str, str]] = [
    {"query": "who maintains the billing service", "expected_key": "payments ownership", "query_class": "factual"},
    {"query": "qué motor de base de datos usamos", "expected_key": "motor base de datos", "query_class": "factual"},
    {"query": "PaymentService retry policy", "expected_key": "PaymentService", "query_class": "code"},
    {"query": "como revertimos un despliegue malo", "expected_key": "rollback procedimiento", "query_class": "procedural"},
]


if __name__ == "__main__":
    raise SystemExit(main())