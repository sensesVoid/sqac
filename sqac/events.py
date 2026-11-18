"""Event bus for cross-process observability (SSE + webhooks).

Thread-safe, in-process pub/sub with a bounded ring of recent events and
an optional outbound webhook. Events are **metadata only** — never
cartridge content, never secret values:

    {"event": "mem_write", "ts": 1731920000.0, "seq": 42,
     "key": "deploy-arch", "kind": "fact", "cartridge": "facts", "source": "mcp"}

SQAC_WEBHOOK_URL (or SQAC_WEBHOOK) — every event is POSTed there as JSON,
best-effort with a short timeout. Webhook failures never affect the caller.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import itertools
from collections import deque
from typing import Any, Optional
from urllib.request import urlopen, Request

MAX_HISTORY = 256
_SUBSCRIBER_QUEUE = 1024

_lock = threading.Lock()
_history: deque[dict[str, Any]] = deque(maxlen=MAX_HISTORY)
_subscribers: list[queue.Queue] = []
_seq = itertools.count(1)


def subscribe() -> queue.Queue:
    q: queue.Queue = queue.Queue(maxsize=_SUBSCRIBER_QUEUE)
    with _lock:
        _subscribers.append(q)
    return q


def unsubscribe(q: queue.Queue) -> None:
    with _lock:
        try:
            _subscribers.remove(q)
        except ValueError:
            pass


def backlog(limit: Optional[int] = None) -> list[dict[str, Any]]:
    with _lock:
        items = list(_history)
    return items[-limit:] if limit is not None else items


def _webhook_url() -> Optional[str]:
    return os.environ.get("SQAC_WEBHOOK_URL") or os.environ.get("SQAC_WEBHOOK") or None


def emit(event: str, **data: Any) -> dict[str, Any]:
    """Publish an event to history, subscribers, and (optionally) the webhook.

    Never raises — callers wiring this into write paths must not fail
    because observability did.
    """
    try:
        evt = {"event": event, "ts": time.time(), "seq": next(_seq), **data}
        with _lock:
            _history.append(evt)
            for q in list(_subscribers):
                try:
                    q.put_nowait(evt)
                except queue.Full:
                    pass
        url = _webhook_url()
        if url:
            threading.Thread(target=_post_webhook, args=(url, evt), daemon=True).start()
        return evt
    except Exception:
        return {}


def _post_webhook(url: str, evt: dict[str, Any]) -> None:
    try:
        body = json.dumps(evt, default=str).encode("utf-8")
        req = Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", "User-Agent": "sqac-events/1"},
            method="POST",
        )
        with urlopen(req, timeout=3) as resp:
            resp.read(256)
    except Exception:
        # webhook is best-effort: never propagate
        pass
