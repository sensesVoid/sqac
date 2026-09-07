#!/usr/bin/env python3
"""SQAC HTTP API server.

    python -m sqac.server --dir ./memory --port 8420
    python -m sqac.cli serve --dir ./memory --port 8420 --api-key sk-xxx
"""
from __future__ import annotations
import argparse, os, time
from pathlib import Path
from typing import Optional
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from .format import compact_cartridge, _HAS_LZ4
from .store import SqacStore, _HAS_SIMD
from .rack import CartridgeRack
from .offloader import ContextOffloader

_api_key = os.environ.get("SQAC_API_KEY", "")
_security = APIKeyHeader(name="X-API-Key", auto_error=False)
_start_time = time.time()

def _verify_key(key: str = Security(_security)):
    if _api_key and key != _api_key:
        raise HTTPException(status_code=401, detail="invalid API key")
    return key

# Module-level state: lazily initialized on first request from env vars
_state: dict = {}

def _ensure_state():
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
class HitResponse(BaseModel):
    content: str; confidence: float; source: str; mode: str; meta: dict = {}

@app.get("/health")
def health():
    return {"status": "ok", "simd": _HAS_SIMD, "lz4": _HAS_LZ4, "uptime_s": round(time.time()-_start_time, 1)}

@app.get("/stats")
def stats(cartridge: Optional[str] = None, _=Depends(_verify_key)):
    return _get_store(cartridge).stats()

@app.post("/search")
def search_endpoint(req: SearchRequest, _=Depends(_verify_key)):
    store = _get_store(req.cartridge)
    if req.threshold is not None: store.fuzzy_threshold = req.threshold
    hits = store.search(req.query, top_k=req.top_k, kind=req.kind)
    return {"hits": [HitResponse(content=h.content, confidence=round(h.confidence,4), source=h.source, mode=h.mode, meta=h.meta).model_dump() for h in hits], "count": len(hits)}

@app.post("/teach")
def teach(req: TeachRequest, _=Depends(_verify_key)):
    _ensure_state()
    store = _get_store(req.cartridge)
    store.add(req.content, key=req.key, source=req.source, kind=req.kind)
    name = req.cartridge or "memory"
    path = _state["dir"] / f"{name}.sqac"
    store.save(path)
    # Reload from disk so subsequent searches see the new entry
    if not req.cartridge or req.cartridge == "memory":
        _state["default_store"] = SqacStore.load(path)
    return {"ok": True, "entries": len(store), "path": str(path)}

@app.post("/compact")
def compact_endpoint(req: CompactRequest, _=Depends(_verify_key)):
    _ensure_state()
    name = req.cartridge or "memory"
    path = _state["dir"] / f"{name}.sqac"
    if not path.exists(): raise HTTPException(404, f"cartridge not found: {name}")
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
    return {"text": o.recall(req.query, top_k=req.top_k), "hit": bool(o.recall(req.query, top_k=req.top_k))}

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
