#!/usr/bin/env python3
"""SQAC HTTP API server.

    python -m sqac.server --dir ./memory --port 8420
    python -m sqac.cli serve --dir ./memory --port 8420 --api-key sk-xxx
"""
from __future__ import annotations
import argparse, hmac, html, os, re, threading, time
from collections import Counter, deque
from pathlib import Path
from typing import Optional
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from .format import compact_cartridge, read_cartridge, _HAS_LZ4
from .store import SqacStore, _HAS_SIMD
from ._version import __version__
from .rack import CartridgeRack
from .offloader import ContextOffloader
import sqac.agent as _agent_mod

# Load the graph visualization HTML at import time
_GRAPH_HTML_PATH = Path(__file__).parent / "graph_view.html"
_GRAPH_HTML = _GRAPH_HTML_PATH.read_text(encoding="utf-8") if _GRAPH_HTML_PATH.exists() else "<h1>graph_view.html not found</h1>"

_api_key = os.environ.get("SQAC_API_KEY", "")
SESSION_FLUSH_TURNS = int(os.environ.get("SQAC_SESSION_FLUSH_TURNS", "10"))
SESSION_FLUSH_SECS = float(os.environ.get("SQAC_SESSION_FLUSH_SECS", "60"))
AGENT_ENCRYPTED_MAGIC = b"SQAE"
# Mirror sqac_mcp.DEFAULT_ROUTES (kept local so the HTTP server never imports the MCP layer)
_RACK_ROUTES = {"fact": "facts", "doc": "docs", "skill": "skills"}


def _make_rack(d):
    return CartridgeRack(
        directory=d, routes=dict(_RACK_ROUTES), default="facts",
        semantic=True, min_confidence=0.60, auto_load=True,
        max_entries=25_000, folder_routing=True,
    )
_security = APIKeyHeader(name="X-API-Key", auto_error=False)
_start_time = time.time()

# ── Metrics counters ────────────────────────────────────────────────────
_metrics_lock = threading.Lock()
_state_lock = threading.Lock()          # guards lazy _ensure_state() init
_teach_locks: dict[str, threading.Lock] = {}   # per-cartridge write serialization
_req_counts: Counter = Counter()        # endpoint -> count
_req_latencies: deque[float] = deque(maxlen=1000)
_req_errors: Counter = Counter()        # status_code -> count
_search_latencies: deque[float] = deque(maxlen=1000)
_teach_count = 0
_compact_count = 0

def _verify_key(key: str = Security(_security)):
    if _api_key and not hmac.compare_digest(key or "", _api_key):
        raise HTTPException(status_code=401, detail="invalid API key")
    return key

def _teach_lock_for(name: str) -> threading.Lock:
    with _metrics_lock:
        lock = _teach_locks.get(name)
        if lock is None:
            lock = _teach_locks[name] = threading.Lock()
        return lock

def _cartridge_path(name: str) -> Path:
    """Resolve a cartridge name to a file, refusing path traversal."""
    if "/" in name or "\\" in name or name in ("", ".", ".."):
        raise HTTPException(400, f"invalid cartridge name: {name!r}")
    base = _state["dir"].resolve()
    path = (base / f"{name}.sqac").resolve()
    if base not in path.parents:
        raise HTTPException(400, f"invalid cartridge name: {name!r}")
    return path

# ── agent cartridge helpers (HTTP parity, v1.3) ─────────────────────────

_AGENT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def _agent_dir() -> Path:
    # resolved lazily so tests can point the agent dir elsewhere
    return Path(getattr(_agent_mod, "AGENT_DIR", Path.home() / ".sqacm" / "agents"))


def _agent_path(name: str) -> Path:
    """Resolve an agent cartridge path, refusing traversal."""
    if not name or not _AGENT_NAME_RE.match(name) or ".." in name:
        raise HTTPException(400, f"invalid agent name: {name!r}")
    base = _agent_dir().resolve()
    p = (base / f"{name}.sqac").resolve()
    if p.parent != base:
        raise HTTPException(400, f"invalid agent name: {name!r}")
    return p


def _agent_meta(path: Path) -> dict:
    """Header-only agent metadata; encrypted (SQAE) agents expose nothing."""
    try:
        raw = path.read_bytes()
    except OSError:
        raw = b""
    base = {"name": path.stem, "role": "", "encrypted": True,
            "model_profile_id": None, "secret_ref": None, "secret_refs": [],
            "allowed_tools": [], "sync_policy": "never", "created": None}
    if raw.startswith(AGENT_ENCRYPTED_MAGIC):
        return dict(base)
    try:
        cart = read_cartridge(str(path))
        info = (cart.header.extra.get("agent") or {}) if isinstance(cart.header.extra, dict) else {}
        base.update({
            "encrypted": bool(cart.header.extra.get("encrypted")),
            "name": info.get("name") or path.stem,
            "role": info.get("role") or "",
            "model_profile_id": info.get("model_profile_id"),
            "secret_ref": info.get("secret_ref"),
            "secret_refs": list(info.get("secret_refs") or []),
            "allowed_tools": list(info.get("allowed_tools") or []),
            "sync_policy": (cart.header.extra.get("sync_policy") or "never"),
            "created": info.get("created"),
            "updated": info.get("updated"),
        })
        return base
    except Exception:
        return dict(base)


VALID_SYNC_POLICIES = {"write_through", "never", "local_only"}

# Module-level state: lazily initialized on first request from env vars
_state: dict = {}

def _ensure_state():
    if _state:
        return
    with _state_lock:
        if _state:
            return
        d = Path(os.environ.get("SQAC_DIR", ".")).resolve()
        d.mkdir(parents=True, exist_ok=True)
        _state["dir"] = d
        _state["rack"] = None
        _state["offloader"] = None
        dp = d / "memory.sqac"
        _state["default_store"] = SqacStore.load(dp) if dp.exists() else SqacStore()
        if not dp.exists():
            _state["default_store"].save(dp)
        if os.environ.get("SQAC_RACK"):
            _state["rack"] = _make_rack(d)
        sp = os.environ.get("SQAC_SESSION")
        if sp:
            sp_path = Path(sp)
            if not sp_path.is_absolute():
                sp_path = d / sp
            _state["offloader"] = ContextOffloader(sp_path)

def _get_store(name: Optional[str] = None) -> SqacStore:
    _ensure_state()
    if name == "session" and _state.get("offloader"):
        return _state["offloader"]._store
    if _state.get("rack") and name:
        return _state["rack"][name]
    s = _state.get("default_store")
    if s is not None:          # stores are __len__-able: an EMPTY store is falsy
        return s
    raise HTTPException(404, "no cartridge loaded")

app = FastAPI(title="SQAC Server", version=__version__)

class SearchRequest(BaseModel):
    query: str; top_k: int = Field(3, ge=1, le=50); kind: Optional[str] = None
    cartridge: Optional[str] = None; threshold: Optional[float] = None
class TeachRequest(BaseModel):
    content: str; key: Optional[str] = None; kind: Optional[str] = None
    source: str = "api"; cartridge: Optional[str] = None
class CompactRequest(BaseModel):
    cartridge: Optional[str] = None
class RackWriteRequest(BaseModel):
    content: str; key: Optional[str] = None; kind: Optional[str] = None; source: str = "api"
class SessionObserveRequest(BaseModel):
    role: str = "user"; text: str
class SessionRecallRequest(BaseModel):
    query: str; top_k: int = Field(2, ge=1, le=10)
class GraphRequest(BaseModel):
    threshold: float = Field(0.55, ge=0.0, le=1.0)
    max_edges: int = Field(200, ge=1, le=2000)
    cartridge: Optional[str] = None
class HitResponse(BaseModel):
    content: str; confidence: float; source: str; mode: str; meta: dict = {}
class AgentCreateRequest(BaseModel):
    name: str
    role: str = ""
    model_profile_id: Optional[str] = None
    secret_ref: Optional[str] = None
    secret_refs: Optional[list[str]] = None
    allowed_tools: Optional[list[str]] = None
    sync_policy: str = "never"
class AgentUpdateRequest(BaseModel):
    role: Optional[str] = None
    model_profile_id: Optional[str] = None
    secret_ref: Optional[str] = None
    secret_refs: Optional[list[str]] = None
    allowed_tools: Optional[list[str]] = None
    sync_policy: Optional[str] = None
class SessionForgetRequest(BaseModel):
    n: Optional[int] = Field(None, ge=1)
    all_turns: bool = False
class SessionGraduateRequest(BaseModel):
    target_cartridge: Optional[str] = None
    threshold: float = Field(0.40, ge=0.0, le=1.0)

@app.middleware("http")
async def _track_metrics(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start
    endpoint = request.url.path
    with _metrics_lock:
        _req_counts[endpoint] += 1
        _req_latencies.append(elapsed)
        if response.status_code >= 400:
            _req_errors[str(response.status_code)] += 1
    return response

@app.get("/health")
def health():
    return {"status": "ok", "simd": _HAS_SIMD, "lz4": _HAS_LZ4, "uptime_s": round(time.time()-_start_time, 1)}

@app.get("/stats")
def stats(cartridge: Optional[str] = None, _=Depends(_verify_key)):
    return _get_store(cartridge).stats()

@app.post("/search")
def search_endpoint(req: SearchRequest, _=Depends(_verify_key)):
    t0 = time.time()
    store = _get_store(req.cartridge)
    if req.threshold is not None:
        # Per-request threshold: apply WITHOUT leaking into the shared store
        # for every subsequent request (that mutation also raced concurrently).
        old = store.fuzzy_threshold
        store.fuzzy_threshold = req.threshold
        try:
            hits = store.search(req.query, top_k=req.top_k, kind=req.kind)
        finally:
            store.fuzzy_threshold = old
    else:
        hits = store.search(req.query, top_k=req.top_k, kind=req.kind)
    latency = (time.time() - t0) * 1000
    with _metrics_lock:
        _search_latencies.append(latency)
    return {"hits": [HitResponse(content=h.content, confidence=round(h.confidence,4), source=h.source, mode=h.mode, meta=h.meta).model_dump() for h in hits], "count": len(hits)}

@app.post("/teach")
def teach(req: TeachRequest, _=Depends(_verify_key)):
    global _teach_count
    _ensure_state()
    name = req.cartridge or "memory"
    path = _cartridge_path(name)
    r = _state.get("rack")
    if r and req.cartridge is None and req.kind:
        # kind-aware teach: route into the rack (fact->facts, skill->skills, doc->docs)
        with _teach_lock_for("__rack__"):
            target = r._target_for(req.kind)
            r.write_routed(req.content, key=req.key, source=req.source, kind=req.kind)
            r.save()
        with _metrics_lock:
            _teach_count += 1
        entries = len(r._require(target)) if target in r else 0
        return {"ok": True, "written": target, "cartridge": target,
                "entries": entries, "path": str(_cartridge_path(target))}
    store = _get_store(req.cartridge)
    with _teach_lock_for(name):
        store.add(req.content, key=req.key, source=req.source, kind=req.kind)
        store.save(path)
    with _metrics_lock:
        _teach_count += 1
    return {"ok": True, "entries": len(store), "path": str(path)}

@app.post("/compact")
def compact_endpoint(req: CompactRequest, _=Depends(_verify_key)):
    global _compact_count
    _ensure_state()
    name = req.cartridge or "memory"
    path = _cartridge_path(name)
    if not path.exists(): raise HTTPException(404, f"cartridge not found: {name}")
    with _metrics_lock:
        _compact_count += 1
    return compact_cartridge(path)

@app.get("/cartridges")
def list_cartridges(_=Depends(_verify_key)):
    _ensure_state()
    r = _state.get("rack"); d = _state["dir"]
    return {"cartridges": r.names() if r else sorted(p.stem for p in d.glob("*.sqac")), "rack": str(d) if r else None}

def _enriched_hit(h):
    """HitResponse fields plus the v1.3 HTTP-parity surface:
    cartridge/key/kind at top level, tier preserved in meta."""
    meta = dict(h.meta)
    return {
        "content": h.content,
        "confidence": round(h.confidence, 4),
        "source": h.source,
        "mode": h.mode,
        "tier": h.mode,  # exact | fuzzy | semantic
        "cartridge": meta.get("cartridge"),
        "key": meta.get("key_norm"),
        "kind": meta.get("kind"),
        "meta": meta,
    }


def _searched(r, cartridges=None):
    """The stems a rack search will actually touch (incl. shards)."""
    targets = r._collect_search_targets(cartridges)
    return [t for t in targets if t in r._stores]


@app.post("/rack/search")
def rack_search(req: SearchRequest, _=Depends(_verify_key)):
    _ensure_state()
    r = _state.get("rack")
    if not r: raise HTTPException(400, "no rack mounted")
    hits = r.search(req.query, top_k=req.top_k, kind=req.kind)
    out = [_enriched_hit(h) for h in hits]
    return {"hits": out, "count": len(out), "searched": _searched(r, None)}

@app.post("/rack/write")
def rack_write(req: RackWriteRequest, _=Depends(_verify_key)):
    _ensure_state()
    r = _state.get("rack")
    if not r: raise HTTPException(400, "no rack mounted")
    r.write_routed(req.content, key=req.key, source=req.source, kind=req.kind); r.save()
    return {"ok": True, "cartridges": r.names()}

@app.post("/session/observe")
def session_observe(req: SessionObserveRequest, _=Depends(_verify_key)):
    _ensure_state()
    o = _state.get("offloader")
    if not o: raise HTTPException(400, "no session offloader")
    xid = o.observe(req.role, req.text)
    if o.should_flush(SESSION_FLUSH_TURNS, SESSION_FLUSH_SECS):
        o.save()
    return {"ok": True, "exchange_offloaded": xid, "buffer_size": len(o._buffer)}

@app.post("/session/recall")
def session_recall(req: SessionRecallRequest, _=Depends(_verify_key)):
    _ensure_state()
    o = _state.get("offloader")
    if not o: raise HTTPException(400, "no session offloader")
    text = o.recall(req.query, top_k=req.top_k)
    return {"text": text, "hit": bool(text)}

@app.post("/graph")
def graph_endpoint(req: GraphRequest, _=Depends(_verify_key)):
    """Compute pairwise VSA similarity and return nodes + edges for 3D visualization."""
    store = _get_store(req.cartridge)
    alive = [i for i, e in enumerate(store._entries) if not e.get("deleted")]
    if not alive:
        return {"nodes": [], "edges": [], "dims": store.dims}

    n = len(alive)
    import numpy as np

    # Get key bit matrices for similarity computation
    kmat = store._bitmatrix("keys")  # (total, D) uint8
    cmat = store._bitmatrix("ckeys")

    # Extract alive vectors
    kvecs = kmat[alive]  # (n, D)
    cvecs = cmat[alive]

    # --- Derive 3D positions from VSA vectors ---
    # Take first 24 bytes (192 bits) → 3 groups of 64 bits → 3 floats in [-1, 1]
    pos = np.zeros((n, 3), dtype=np.float32)
    for dim in range(3):
        start = dim * 24
        end = start + 24
        chunk = kvecs[:, start:end].view(np.uint64)  # treat 8 bytes as uint64
        # Normalize to [-1, 1] using bit population count
        counts = np.unpackbits(chunk.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1).reshape(n, 3)
        pos[:, dim] = (counts.mean(axis=1) / 8.0) * 2.0 - 1.0

    # Scale positions for better spacing
    pos *= 15.0

    # --- Compute pairwise similarity ---
    # XOR all pairs → hamming distance → similarity
    # For efficiency: compute key similarity matrix
    # kvecs is (n, D) as uint8 bits. Use broadcasting.
    # Actually, let's use a smarter approach: pack into bits and use XOR
    kbits = np.unpackbits(kvecs, axis=1).astype(np.int8)  # (n, D)
    cbits = np.unpackbits(cvecs, axis=1).astype(np.int8)

    # Compute similarity matrix: 1 - hamming/dims
    # (n, 1, D) XOR (1, n, D) → (n, n, D) is too big for large n
    # Instead: use dot product trick. sim = 1 - (n_different / D)
    # n_different = D - (kbits[i] == kbits[j]).sum()
    # = D - (kbits[i] * kbits[j] + (1-kbits[i])*(1-kbits[j])).sum()
    # For binary {0,1}: (a == b) = a*b + (1-a)*(1-b) = 1 - a - b + 2ab
    # So: sim = 1 - (D - sum(1 - a - b + 2ab))/D = sum(1 - a - b + 2ab)/D
    # = (D - sum(a) - sum(b) + 2*sum(ab))/D
    # For chunks: compute in blocks to avoid O(n^2 * D) memory

    edges = []
    if n <= 1:
        pass
    else:
        # Compute similarity in blocks to manage memory
        block_size = min(500, n)
        sim_threshold = req.threshold
        all_edges = []

        for i_start in range(0, n, block_size):
            i_end = min(i_start + block_size, n)
            ki = kbits[i_start:i_end]  # (bi, D)
            ci = cbits[i_start:i_end]

            for j_start in range(i_start, n, block_size):
                j_end = min(j_start + block_size, n)
                kj = kbits[j_start:j_end]  # (bj, D)
                cj = cbits[j_start:j_end]

                # Key similarity: 1 - hamming/D
                # hamming = (ki[:, None, :] != kj[None, :, :]).sum(axis=2)
                # For binary: XOR then popcount
                # sim = 1 - (ki @ (1-kj.T) + (1-ki) @ kj.T) / D
                D = kbits.shape[1]
                # Key sim
                sum_ki = ki.sum(axis=1)  # (bi,)
                sum_kj = kj.sum(axis=1)  # (bj,)
                dot_kk = ki.astype(np.int32) @ kj.astype(np.int32).T  # (bi, bj)
                sim_k = (D - sum_ki[:, None] - sum_kj[None, :] + 2 * dot_kk) / D

                # Content sim
                sum_ci = ci.sum(axis=1)
                sum_cj = cj.sum(axis=1)
                dot_cc = ci.astype(np.int32) @ cj.astype(np.int32).T
                sim_c = (D - sum_ci[:, None] - sum_cj[None, :] + 2 * dot_cc) / D

                # Combined: max of key and content similarity
                sim = np.maximum(sim_k, sim_c)

                # Extract edges above threshold
                for li in range(sim.shape[0]):
                    for lj in range(sim.shape[1]):
                        gi = i_start + li  # global index in alive array
                        gj = j_start + lj  # global index in alive array
                        if gi >= gj:
                            continue  # skip self and duplicates
                        s = float(sim[li, lj])
                        if s >= sim_threshold:
                            all_edges.append((gi, gj, s))

        # Sort by similarity descending, keep top max_edges
        all_edges.sort(key=lambda e: -e[2])
        edges = all_edges[:req.max_edges]

    # --- Build node list ---
    nodes = []
    kind_colors = {"fact": "#58a6ff", "skill": "#f0883e", "doc": "#3fb950", "turn": "#bc8cff", "generic": "#8b949e"}
    cartridge_name = getattr(store, "_name", None) or req.cartridge or "memory"
    for i, idx in enumerate(alive):
        entry = store._entries[idx]
        kind = entry.get("kind", "generic")
        if isinstance(kind, int):
            from .store import KIND_NAMES
            kind = KIND_NAMES.get(kind, "generic")
        nodes.append({
            "id": i,
            "key": entry.get("key_norm", entry.get("key", "")),
            "content": entry["content"][:100],
            "kind": kind,
            "source": entry.get("source", ""),
            "color": kind_colors.get(kind, "#8b949e"),
            "position": [round(float(pos[i, 0]), 3), round(float(pos[i, 1]), 3), round(float(pos[i, 2]), 3)],
            "is_cartridge": False,
        })

    # --- Add cartridge as center node ---
    cartridge_id = n
    nodes.append({
        "id": cartridge_id,
        "key": cartridge_name,
        "content": f"Cartridge: {cartridge_name} ({len(store)} entries, {n} alive)",
        "kind": "cartridge",
        "source": "cartridge",
        "color": "#f0e68c",
        "position": [0.0, 0.0, 0.0],
        "is_cartridge": True,
    })

    # Connect every entry to the cartridge center
    cartridge_edges = [{"source": i, "target": cartridge_id, "similarity": 1.0} for i in range(n)]
    all_edges_out = [{"source": e[0], "target": e[1], "similarity": round(e[2], 4)} for e in edges] + cartridge_edges

    return {
        "nodes": nodes,
        "edges": all_edges_out,
        "dims": store.dims,
        "total_entries": len(store),
        "alive_entries": n,
        "cartridge": cartridge_name,
    }


@app.get("/graph")
def graph_view(_=Depends(_verify_key)):
    """Serve the 3D graph visualization HTML."""
    return HTMLResponse(_GRAPH_HTML)


@app.get("/metrics")
def metrics(_=Depends(_verify_key)):
    """Prometheus-compatible metrics endpoint."""
    uptime = time.time() - _start_time
    lines = [
        "# HELP sqac_uptime_seconds Server uptime in seconds.",
        "# TYPE sqac_uptime_seconds gauge",
        f'sqac_uptime_seconds {uptime:.1f}',
        "# HELP sqac_requests_total Total requests per endpoint.",
        "# TYPE sqac_requests_total counter",
    ]
    with _metrics_lock:
        for ep, count in sorted(_req_counts.items()):
            safe_ep = ep.replace("/", "_").strip("_")
            lines.append(f'sqac_requests_total{{endpoint="{safe_ep}"}} {count}')
        lines += [
            "# HELP sqac_errors_total Total error responses by status.",
            "# TYPE sqac_errors_total counter",
        ]
        for code, count in sorted(_req_errors.items()):
            lines.append(f'sqac_errors_total{{status="{code}"}} {count}')
        lines += [
            "# HELP sqac_search_latency_ms Search latency in milliseconds.",
            "# TYPE sqac_search_latency_ms summary",
        ]
        if _search_latencies:
            sl = sorted(_search_latencies)
            n = len(sl)
            lines.append(f'sqac_search_latency_ms{{quantile="0.5"}} {sl[n//2]:.2f}')
            lines.append(f'sqac_search_latency_ms{{quantile="0.95"}} {sl[int(n*0.95)]:.2f}')
            lines.append(f'sqac_search_latency_ms{{quantile="0.99"}} {sl[int(n*0.99)]:.2f}')
            lines.append(f'sqac_search_latency_ms_sum {sum(sl):.2f}')
            lines.append(f'sqac_search_latency_ms_count {n}')
        lines += [
            "# HELP sqac_request_latency_ms Request latency in milliseconds.",
            "# TYPE sqac_request_latency_ms summary",
        ]
        if _req_latencies:
            rl = sorted(_req_latencies)
            n = len(rl)
            lines.append(f'sqac_request_latency_ms{{quantile="0.5"}} {rl[n//2]*1000:.2f}')
            lines.append(f'sqac_request_latency_ms{{quantile="0.95"}} {rl[int(n*0.95)]*1000:.2f}')
            lines.append(f'sqac_request_latency_ms{{quantile="0.99"}} {rl[int(n*0.99)]*1000:.2f}')
            lines.append(f'sqac_request_latency_ms_sum {sum(rl)*1000:.2f}')
            lines.append(f'sqac_request_latency_ms_count {n}')
        lines += [
            f"# HELP sqac_teach_total Total teach operations.",
            f"# TYPE sqac_teach_total counter",
            f"sqac_teach_total {_teach_count}",
            f"# HELP sqac_compact_total Total compact operations.",
            f"# TYPE sqac_compact_total counter",
            f"sqac_compact_total {_compact_count}",
        ]
    _ensure_state()
    store = _get_store()
    stats = store.stats()
    lines += [
        f"# HELP sqac_entries Current entry count.",
        f"# TYPE sqac_entries gauge",
        f'sqac_entries {{cartridge="memory"}} {stats["entries"]}',
        f"# HELP sqac_dims Vector dimensionality.",
        f"# TYPE sqac_dims gauge",
        f'sqac_dims {stats["dims"]}',
        f"# HELP sqac_simd_enabled Whether Rust SIMD is active.",
        f"# TYPE sqac_simd_enabled gauge",
        f'sqac_simd_enabled {1 if _HAS_SIMD else 0}',
        f"# HELP sqac_lz4_enabled Whether lz4 compression is active.",
        f"# TYPE sqac_lz4_enabled gauge",
        f'sqac_lz4_enabled {1 if _HAS_LZ4 else 0}',
    ]
    return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4; charset=utf-8")


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(_=Depends(_verify_key)):
    """Self-contained HTML dashboard — no external deps."""
    _ensure_state()
    store = _get_store()
    stats = store.stats()
    uptime = time.time() - _start_time
    # Get top entries for display
    hits = store.search("", top_k=20)
    # If empty query returns nothing, get all entries by searching with a wildcard-ish term
    if not hits:
        hits = store.search("the", top_k=20)
    rack = _state.get("rack")
    cartridges = rack.names() if rack else sorted(p.stem for p in _state["dir"].glob("*.sqac"))
    with _metrics_lock:
        total_requests = sum(_req_counts.values())
        total_errors = sum(_req_errors.values())
        search_p50 = (sorted(_search_latencies)[len(_search_latencies)//2] if _search_latencies else 0)
        search_p99 = (sorted(_search_latencies)[int(len(_search_latencies)*0.99)] if _search_latencies else 0)
    kinds_html = "".join(
        f'<span class="badge">{html.escape(str(k))}: {html.escape(str(v))}</span>'
        for k, v in sorted(stats.get("kinds", {}).items())
    )
    entries_html = ""
    for h in hits[:15]:
        tag = "exact" if h.mode == "exact" else "fuzzy"
        content_h = html.escape(h.content[:120] or "")
        source_h = html.escape(h.source or "")
        entries_html += f'<div class="entry"><span class="tag tag-{tag}">{tag}</span> '
        entries_html += f'<span class="conf">{h.confidence:.3f}</span> '
        entries_html += f'<span class="content">{content_h}</span>'
        if h.source:
            entries_html += f' <span class="source">{source_h}</span>'
        entries_html += '</div>\n'
    dashboard_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SQAC Dashboard</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0d1117;color:#c9d1d9;padding:24px;max-width:960px;margin:0 auto}}
h1{{font-size:1.5rem;margin-bottom:4px;color:#58a6ff}}
.subtitle{{color:#8b949e;font-size:.85rem;margin-bottom:20px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-bottom:20px}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px}}
.card .label{{font-size:.75rem;color:#8b949e;text-transform:uppercase;letter-spacing:.5px}}
.card .value{{font-size:1.6rem;font-weight:600;color:#c9d1d9;margin-top:4px}}
.card .value.green{{color:#3fb950}}
.card .value.blue{{color:#58a6ff}}
.badge{{display:inline-block;background:#1f6feb33;color:#58a6ff;border:1px solid #1f6feb;border-radius:12px;padding:2px 10px;font-size:.8rem;margin:2px}}
h2{{font-size:1.1rem;margin:20px 0 10px;color:#c9d1d9}}
.entry{{background:#161b22;border:1px solid #21262d;border-radius:6px;padding:8px 12px;margin-bottom:6px;font-size:.85rem;display:flex;align-items:center;gap:8px;flex-wrap:wrap}}
.entry .content{{flex:1;min-width:200px}}
.tag{{font-size:.7rem;padding:2px 6px;border-radius:4px;font-weight:600}}
.tag-exact{{background:#238636;color:#fff}}
.tag-fuzzy{{background:#1f6feb;color:#fff}}
.conf{{color:#8b949e;font-family:monospace;font-size:.8rem;min-width:40px}}
.source{{color:#8b949e;font-size:.75rem}}
.table{{width:100%;border-collapse:collapse;font-size:.85rem}}
.table th,.table td{{padding:6px 10px;text-align:left;border-bottom:1px solid #21262d}}
.table th{{color:#8b949e;font-weight:500}}
footer{{margin-top:24px;padding-top:12px;border-top:1px solid #21263d;color:#484f58;font-size:.75rem}}
</style>
</head>
<body>
<h1>⚡ SQAC Dashboard</h1>
<div class="subtitle">Real-time overview — {time.strftime('%Y-%m-%d %H:%M:%S')}</div>

<div class="grid">
  <div class="card"><div class="label">Entries</div><div class="value blue">{stats['entries']}</div></div>
  <div class="card"><div class="label">Dimensions</div><div class="value">{stats['dims']}</div></div>
  <div class="card"><div class="label">Uptime</div><div class="value green">{uptime/3600:.1f}h</div></div>
  <div class="card"><div class="label">Requests</div><div class="value">{total_requests}</div></div>
  <div class="card"><div class="label">Errors</div><div class="value">{total_errors}</div></div>
  <div class="card"><div class="label">Search p50</div><div class="value green">{search_p50:.1f}ms</div></div>
  <div class="card"><div class="label">Search p99</div><div class="value">{search_p99:.1f}ms</div></div>
  <div class="card"><div class="label">Cartridge</div><div class="value">{stats.get('encoder','')}</div></div>
</div>

<div class="grid">
  <div class="card">
    <div class="label">Features</div>
    <div style="margin-top:6px">
      {'<span class="badge">SIMD</span>' if stats.get('simd') else ''}
      {'<span class="badge">LZ4</span>' if stats.get('lz4') else ''}
      {'<span class="badge">Semantic</span>' if stats.get('semantic') else ''}
    </div>
  </div>
  <div class="card">
    <div class="label">Knowledge Kinds</div>
    <div style="margin-top:6px">{kinds_html or '<span class="badge">generic</span>'}</div>
  </div>
  <div class="card">
    <div class="label">Cartridges</div>
    <div style="margin-top:6px">{''.join(f'<span class="badge">{html.escape(c)}</span>' for c in cartridges)}</div>
  </div>
</div>

<h2>Entries</h2>
{entries_html or '<div class="entry"><span class="content">No entries yet — teach something with <code>sqac teach</code></span></div>'}

<h2>Endpoints</h2>
<table class="table">
<tr><th>Method</th><th>Path</th><th>Description</th></tr>
<tr><td>GET</td><td>/health</td><td>Health check (no auth)</td></tr>
<tr><td>GET</td><td>/stats</td><td>Cartridge statistics</td></tr>
<tr><td>GET</td><td>/metrics</td><td>Prometheus metrics</td></tr>
<tr><td>GET</td><td>/dashboard</td><td>This dashboard</td></tr>
<tr><td>POST</td><td>/search</td><td>Search the memory</td></tr>
<tr><td>POST</td><td>/teach</td><td>Teach a new fact</td></tr>
<tr><td>POST</td><td>/compact</td><td>Remove tombstones</td></tr>
<tr><td>GET</td><td>/cartridges</td><td>List cartridges</td></tr>
<tr><td>POST</td><td>/rack/search</td><td>Search across rack</td></tr>
<tr><td>POST</td><td>/rack/write</td><td>Write with routing</td></tr>
<tr><td>POST</td><td>/graph</td><td>Graph data (JSON)</td></tr>
<tr><td>GET</td><td>/graph</td><td>3D graph visualization</td></tr>
</table>

<footer>SQAC v{__version__} — VSA memory cartridges for LLMs — <a href="/health" style="color:#58a6ff">Health</a> · <a href="/metrics" style="color:#58a6ff">Metrics</a> · <a href="/docs" style="color:#58a6ff">API Docs</a></footer>
</body></html>"""
    return HTMLResponse(dashboard_html)


def main(argv: list[str] | None = None) -> int:
    global _api_key
    ap = argparse.ArgumentParser(prog="sqac.server")
    ap.add_argument("--dir", default=None, help="cartridge directory (default: env SQAC_DIR or .)")
    ap.add_argument("--port", type=int, default=8420)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--rack", action="store_true")
    ap.add_argument("--session", default=None)
    args = ap.parse_args(argv)
    # Only override env vars when explicitly provided via CLI
    if args.dir is not None:
        os.environ["SQAC_DIR"] = str(Path(args.dir).resolve())
    if args.rack:
        os.environ["SQAC_RACK"] = "1"
    if args.session:
        os.environ["SQAC_SESSION"] = args.session
    if args.api_key:
        _api_key = args.api_key
    elif not os.environ.get("SQAC_API_KEY"):
        _api_key = ""  # no key required if neither CLI nor env
    d = Path(os.environ.get("SQAC_DIR", ".")).resolve()
    print(f"SQAC server on {args.host}:{args.port} | dir={d} | SIMD={_HAS_SIMD} | LZ4={_HAS_LZ4}")
    import uvicorn; uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0

# ── agents (HTTP parity, Tier 1A) ───────────────────────────────────

def _agent_list():
    d = _agent_dir()
    out = []
    if d.exists():
        for p in sorted(d.glob("*.sqac")):
            if p.name.startswith("."):
                continue
            out.append(_agent_meta(p))
    return out


@app.get("/agents")
def agents_list(_=Depends(_verify_key)):
    return {"agents": _agent_list()}


@app.get("/agents/{name}")
def agents_get(name: str, _=Depends(_verify_key)):
    p = _agent_path(name)
    if not p.exists():
        raise HTTPException(404, f"agent not found: {name}")
    return _agent_meta(p)


@app.post("/agents")
def agents_create(req: AgentCreateRequest, _=Depends(_verify_key)):
    name = req.name.strip()
    p = _agent_path(name)
    if p.exists():
        raise HTTPException(409, f"agent already exists: {name}")
    if req.sync_policy not in VALID_SYNC_POLICIES:
        raise HTTPException(400, f"invalid sync_policy: {req.sync_policy!r}")
    try:
        _agent_mod.create_agent(
            name, req.role, semantic=False,
            model_profile_id=req.model_profile_id,
            secret_ref=req.secret_ref, secret_refs=req.secret_refs,
            allowed_tools=req.allowed_tools, sync_policy=req.sync_policy)
    except Exception as exc:
        raise HTTPException(400, f"create failed: {exc}")
    return {"ok": True, **_agent_meta(p)}


@app.put("/agents/{name}")
def agents_update(name: str, req: AgentUpdateRequest, _=Depends(_verify_key)):
    p = _agent_path(name)
    if not p.exists():
        raise HTTPException(404, f"agent not found: {name}")
    if req.sync_policy is not None and req.sync_policy not in VALID_SYNC_POLICIES:
        raise HTTPException(400, f"invalid sync_policy: {req.sync_policy!r}")
    is_enc = p.read_bytes().startswith(AGENT_ENCRYPTED_MAGIC)
    try:
        ag = _agent_mod.load_agent(name, semantic=False)
    except Exception as exc:
        if is_enc:
            raise HTTPException(422, f"agent is encrypted; it cannot be opened here: {exc}")
        raise HTTPException(400, f"load failed: {exc}")
    if req.role is not None:
        ag.role = req.role
    binding = {}
    if req.model_profile_id is not None:
        binding["model_profile_id"] = req.model_profile_id
    if req.secret_ref is not None:
        binding["secret_ref"] = req.secret_ref
    if req.secret_refs is not None:
        binding["secret_refs"] = req.secret_refs
    if req.allowed_tools is not None:
        binding["allowed_tools"] = req.allowed_tools
    if req.sync_policy is not None:
        binding["sync_policy"] = req.sync_policy
    if binding:
        ag.set_binding(**binding)   # persists internally
    else:
        ag.save()
    return {"ok": True, **_agent_meta(p)}


@app.delete("/agents/{name}")
def agents_delete(name: str, hard: bool = False, _=Depends(_verify_key)):
    p = _agent_path(name)
    if not p.exists():
        raise HTTPException(404, f"agent not found: {name}")
    if hard:
        p.unlink()
        return {"ok": True, "deleted": name}
    trash = p.parent / ".trash"
    trash.mkdir(parents=True, exist_ok=True)
    dest = trash / f"{name}-{int(time.time())}.sqac"
    p.rename(dest)
    try:
        active_ptr = Path(getattr(_agent_mod, "ACTIVE_AGENT_PATH", ""))
        if active_ptr.exists() and active_ptr.read_text(encoding="utf-8").strip() == name:
            active_ptr.unlink()
    except OSError:
        pass
    return {"ok": True, "tombstoned": name, "trash": str(dest)}


# ── rack introspection + reload (Tier 1B) ──────────────────────────

@app.get("/rack")
def rack_state(_=Depends(_verify_key)):
    _ensure_state()
    r = _state.get("rack")
    if not r:
        raise HTTPException(400, "no rack mounted")
    mounted = []
    for n in r.names():
        s = r.stats(n)
        entries = s.get("entries", 0)
        mounted.append({
            "name": n,
            "entries": entries,
            "kinds": s.get("kinds", {}),
            "percent_full": round(100.0 * entries / max(1, r.max_entries), 1),
        })
    return {"mounted": mounted, "count": len(mounted),
            "max_entries": r.max_entries, "default": r.default}


@app.get("/rack/capabilities")
def rack_capabilities(_=Depends(_verify_key)):
    _ensure_state()
    r = _state.get("rack")
    if not r:
        raise HTTPException(400, "no rack mounted")
    routes = {str(k): v for k, v in (r.routes or {}).items()}
    return {
        "kinds": sorted(set(routes) | {"generic"}),
        "routes": routes,
        "default": r.default,
        "max_entries": r.max_entries,
        "shard_at": r.max_entries,
    }


@app.post("/reload")
def reload_rack(_=Depends(_verify_key)):
    _ensure_state()
    old = _state.get("rack")
    had_rack = old is not None or bool(os.environ.get("SQAC_RACK"))
    if not had_rack:
        return {"ok": True, "mounted": [], "note": "no rack to reload"}
    if old is not None:
        try:
            old.save()
        except Exception:
            pass
    d = _state["dir"]
    _state["rack"] = _make_rack(d)
    r = _state["rack"]
    return {"ok": True, "mounted": r.names(), "directory": str(d)}


# ── rack search trace (Tier 1D) ─────────────────────────────────────

@app.post("/rack/trace")
def rack_trace(req: SearchRequest, _=Depends(_verify_key)):
    _ensure_state()
    r = _state.get("rack")
    if not r:
        raise HTTPException(400, "no rack mounted")
    hits = r.search(req.query, top_k=req.top_k, kind=req.kind)
    out = []
    for h in hits:
        hit = _enriched_hit(h)
        hit["tier"] = h.mode
        hit["latency_ms"] = h.meta.get("latency_ms")
        out.append(hit)
    return {"hits": out, "count": len(out), "searched": _searched(r, None)}


# ── session lifecycle (Tier 1E) ──────────────────────────────────────

@app.get("/session/stats")
def session_stats(_=Depends(_verify_key)):
    _ensure_state()
    o = _state.get("offloader")
    if not o:
        raise HTTPException(400, "no session offloader")
    return {
        "entries": len(o._store),
        "offloaded_exchanges": o._xid,
        "turns": o._turn_count,
        "live_buffer": len(o._buffer),
        "window": o.window,
        "kinds": o._store.stats().get("kinds", {}),
    }


@app.post("/session/graduate")
def session_graduate(req: SessionGraduateRequest, _=Depends(_verify_key)):
    _ensure_state()
    o = _state.get("offloader")
    if not o:
        raise HTTPException(400, "no session offloader")
    r = _state.get("rack")
    if not r:
        raise HTTPException(400, "no rack mounted")
    target = req.target_cartridge or r.default or "facts"
    if target not in r:
        r.create(target)
    report = r.graduate(o, target_name=target, threshold=req.threshold)
    r.save()
    o.save()
    target = req.target_cartridge or r.default or "facts"
    return {"ok": True, "target": target, **(report if isinstance(report, dict) else {"report": report})}


@app.post("/session/forget")
def session_forget(req: SessionForgetRequest, _=Depends(_verify_key)):
    _ensure_state()
    o = _state.get("offloader")
    if not o:
        raise HTTPException(400, "no session offloader")
    from .store import KIND_TURN
    store = o._store
    entries = store._entries
    turn_idx = [i for i, e in enumerate(entries)
                if not e.get("deleted") and e.get("kind") == KIND_TURN]
    if req.all_turns:
        drop = turn_idx
    else:
        n = req.n or len(turn_idx)
        drop = turn_idx[:n]
    for i in drop:
        store.delete(i)
    buffer_cleared = False
    if req.all_turns:
        o._buffer.clear()
        buffer_cleared = True
    removed = len(drop)
    if removed:
        o.save()
    return {"ok": True, "removed": removed, "buffer_cleared": buffer_cleared,
            "remaining_turns": sum(1 for e in store._entries
                                    if not e.get("deleted") and e.get("kind") == KIND_TURN)}



if __name__ == "__main__":
    import sys; sys.exit(main())
