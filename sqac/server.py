#!/usr/bin/env python3
"""SQAC HTTP API server.

    python -m sqac.server --dir ./memory --port 8420
    python -m sqac.cli serve --dir ./memory --port 8420 --api-key sk-xxx
"""
from __future__ import annotations
import argparse, hmac, html, os, threading, time
from collections import Counter, deque
from pathlib import Path
from typing import Optional
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from .format import compact_cartridge, _HAS_LZ4
from .store import SqacStore, _HAS_SIMD
from .rack import CartridgeRack
from .offloader import ContextOffloader

# Load the graph visualization HTML at import time
_GRAPH_HTML_PATH = Path(__file__).parent / "graph_view.html"
_GRAPH_HTML = _GRAPH_HTML_PATH.read_text(encoding="utf-8") if _GRAPH_HTML_PATH.exists() else "<h1>graph_view.html not found</h1>"

_api_key = os.environ.get("SQAC_API_KEY", "")
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
            _state["rack"] = CartridgeRack(directory=d, semantic=True)
        sp = os.environ.get("SQAC_SESSION")
        if sp:
            _state["offloader"] = ContextOffloader(sp)

def _get_store(name: Optional[str] = None) -> SqacStore:
    _ensure_state()
    if _state.get("rack") and name:
        return _state["rack"][name]
    s = _state.get("default_store")
    if s:
        return s
    raise HTTPException(404, "no cartridge loaded")

app = FastAPI(title="SQAC Server", version="0.1.0")

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
class HitResponse(BaseModel):
    content: str; confidence: float; source: str; mode: str; meta: dict = {}

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
    store = _get_store(req.cartridge)
    name = req.cartridge or "memory"
    path = _cartridge_path(name)
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

@app.post("/rack/search")
def rack_search(req: SearchRequest, _=Depends(_verify_key)):
    _ensure_state()
    r = _state.get("rack")
    if not r: raise HTTPException(400, "no rack mounted")
    hits = r.search(req.query, top_k=req.top_k, kind=req.kind)
    return {"hits": [HitResponse(content=h.content, confidence=round(h.confidence,4), source=h.source, mode=h.mode, meta=h.meta).model_dump() for h in hits], "count": len(hits)}

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
    xid = o.observe(req.role, req.text); o.save()
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
    store = _get_store()
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
        })

    return {
        "nodes": nodes,
        "edges": [{"source": e[0], "target": e[1], "similarity": round(e[2], 4)} for e in edges],
        "dims": store.dims,
        "total_entries": len(store),
        "alive_entries": n,
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

<footer>SQAC v0.1.0 — VSA memory cartridges for LLMs — <a href="/health" style="color:#58a6ff">Health</a> · <a href="/metrics" style="color:#58a6ff">Metrics</a> · <a href="/docs" style="color:#58a6ff">API Docs</a></footer>
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

if __name__ == "__main__":
    import sys; sys.exit(main())
