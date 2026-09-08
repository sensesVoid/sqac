"""Dynamic Memory Sparsification (DMS) — staleness/decay policy for SQAC memory.

The capacity benchmark (`experiments/CAPACITY_BENCHMARK.md`) showed a single
cartridge is fast far beyond normal use: sweet spot 25K entries (~1.1M tokens,
27ms scan, 132MB RSS), practical ceiling 100K (~4.5M tokens, 108ms). DMS is the
*eviction* layer that keeps memory sparse and fast when you push toward that
ceiling — and, per the KV-eviction literature, selective forgetting can
*improve* generation by suppressing attention dilution (Make Each Token Count /
DBTrimKV), so DMS is not just memory management, it is a quality knob.

Policy surface:
    utility(exchange) = salience * exp(-lambda_decay * age) + alpha * access_count

    - `salience`  — existing cheap importance (salient markers + questions).
    - exp(-lambda_decay*age) — TTL aging: cold-but-valuable tails survive
      (bimodal reuse in Qwen-Bailian traces: hot short-cycle + cold long-tail).
    - access_count — each recall bumps utility (ARC-like promotion; protects
      repeatedly-accessed exchanges from premature eviction).

Three tiers:
    live      — turns in the window buffer (always retained, no cost)
    bucket    — offloaded KIND_TURN entries (evictable, recallable)
    durable   — graduated facts (KIND_FACT, exempt from eviction)

Sparsification:
    when total *basket* entries exceed `budget`, keep the top `keep_ratio`
    by utility in the bucket and *demote* the long tail tx the durable store
    (they remain recallable, but leave the hot basket so scans stay short).
    `demote_threshold` sets the absolute floor under which an entry is a
    candidate for removal once it passes its TTL.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .store import KIND_FACT, KIND_TURN, SqacStore

# Time base for the TTL half-life when nothing extrinsic is passed in.
# Matches the bimodal KV reuse finding: give the cold tail ~hours, not minutes.
_AGE_FACTOR = 1.0 / 60.0  # convert elapsed seconds -> minutes for the decay term


@dataclass
class UtilityWeights:
    """Tunables for the utility scoring function."""

    lambda_decay: float = 0.10   # hourly decay on the e^(-lambda*t) TTL term
    alpha: float = 0.15          # weight per recall access
    salience: float = 1.0        # weight on base salience
    demote_threshold: float = 0.10  # floor below which TTL-expired entries are removable
    keep_ratio: float = 0.80     # fraction of the basket to keep when over budget


def utility(
    salience: float,
    age_minutes: float,
    access_count: int,
    w: Optional[UtilityWeights] = None,
) -> float:
    """Score one exchange. Higher = more valuable to keep in the hot basket."""
    w = w or UtilityWeights()
    if age_minutes < 0:
        age_minutes = 0.0
    age_hours = age_minutes / 60.0
    return (
        w.salience * salience * math.exp(-w.lambda_decay * age_hours)
        + w.alpha * access_count
    )


@dataclass
class DMSRecord:
    """Bookkeeping for one exchange under DMS."""

    xid: int
    salience: float
    created_at: float
    access_count: int = 0
    tier: str = "bucket"      # "live" | "bucket" | "durable"
    score: float = 0.0        # last computed utility


class DMS:
    """Drives sparsification over an offloader's store."""

    def __init__(
        self,
        budget: int = 25_000,
        weights: Optional[UtilityWeights] = None,
        now: Optional[Callable[[], float]] = None,
    ):
        self.budget = max(1, budget)
        self.weights = weights or UtilityWeights()
        self.now = now or time.time
        self._records: dict[int, DMSRecord] = {}

    # ── write path ──────────────────────────────────────────────────────

    def register(self, xid: int, salience: float, tier: str = "bucket") -> None:
        """Book a new exchange with the DMS."""
        self._records[xid] = DMSRecord(
            xid=xid,
            salience=salience,
            created_at=self.now(),
            tier=tier,
        )

    def touch(self, xid: int) -> None:
        """Bump the access_count for a recalled exchange."""
        r = self._records.get(xid)
        if r is not None:
            r.access_count += 1

    def promote(self, xid: int) -> None:
        """Move an exchange to the durable tier (exempt from sparsification)."""
        r = self._records.get(xid)
        if r is not None:
            r.tier = "durable"

    # ── scoring / eviction ──────────────────────────────────────────────

    def score(self, xid: int) -> float:
        r = self._records.get(xid)
        if r is None:
            return 0.0
        age = (self.now() - r.created_at) / 60.0  # minutes
        s = utility(r.salience, age, r.access_count, self.weights)
        r.score = s
        return s

    def over_budget(self) -> bool:
        n = sum(
            1 for r in self._records.values()
            if r.tier == "bucket"   # live + durable don't count against the basket
        )
        return n > self.budget

    def evict_list(self, top_k: int | None = None) -> list[int]:
        """Rank bucket-tier records by score (ascending = most evictable).

        Returns xids from the *bottom* of the ranking up to `top_k` (or all
        that are both expired and below the demote threshold). The caller
        decides between (a) soft-evict to the durable store or (b) drop.
        """
        cands = [r for r in self._records.values() if r.tier == "bucket"]
        if not cands:
            return []
        scored = [(self.score(r.xid), r.xid) for r in cands]
        scored.sort()  # ascending score = least useful first
        if top_k is not None:
            return [xid for _, xid in scored[:top_k]]
        return [
            xid for score, xid in scored
            if score < self.weights.demote_threshold
        ]

    def stats(self) -> dict[str, Any]:
        tiers = {"live": 0, "bucket": 0, "durable": 0}
        total_access = 0
        for r in self._records.values():
            tiers[r.tier] = tiers.get(r.tier, 0) + 1
            total_access += r.access_count
        return {
            "records": len(self._records),
            "tiers": tiers,
            "budget": self.budget,
            "over_budget": self.over_budget(),
            "total_access_count": total_access,
            "weights": {
                "lambda_decay": self.weights.lambda_decay,
                "alpha": self.weights.alpha,
                "demote_threshold": self.weights.demote_threshold,
                "keep_ratio": self.weights.keep_ratio,
            },
        }


# ── integration helper: rank entries already in a store ────────────────────

def rank_store(
    store: SqacStore,
    dms: DMS,
    kind: int | str | None = None,
) -> list[dict[str, Any]]:
    """Return all bucket-tier entries ranked by DMS utility (ascending).

    Useful to drive a compaction/eviction pass over a persisted cartridge.
    Each item: {idx, xid, score, salience, access_count, age_minutes, tier}.
    """
    out: list[dict[str, Any]] = []
    for idx, e in enumerate(store._entries):
        if e.get("deleted"):
            continue
        k = e.get("kind")
        if kind is not None and k != (kind if isinstance(kind, int) else _kind_int(kind)):
            continue
        meta = e.get("meta", {})
        xid = meta.get("exchange", idx)
        rec = dms._records.get(xid) or DMSRecord(
            xid=xid,
            salience=meta.get("salience", 0.0),
            created_at=meta.get("created_at", 0.0),
            access_count=meta.get("access_count", 0),
            tier=meta.get("tier", "bucket"),
        )
        out.append(
            {
                "idx": idx,
                "xid": xid,
                "score": dms.score(xid) if xid in dms._records else 0.0,
                "salience": rec.salience,
                "access_count": rec.access_count,
                "age_minutes": (dms.now() - rec.created_at) / 60.0 if rec.created_at else 0.0,
                "tier": rec.tier,
            }
        )
    out.sort(key=lambda d: d["score"])
    return out


def _kind_int(kind) -> int:
    from .store import resolve_kind

    return resolve_kind(kind)
