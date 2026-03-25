"""
Shared utilities for traceable checkpoint saving.

Goal:
- Every run gets a unique `run_id` based on timestamp + cfg fingerprint.
- Save checkpoints under `checkpoints/stage*/history/<run_id>/`.
- Also write/update `checkpoints/stage*/latest/` fixed paths for convenience.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from typing import Any, Dict, Optional


def _json_default(obj: Any) -> Any:
    # Make common non-JSON types serializable.
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return str(obj)


def cfg_fingerprint(cfg: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> str:
    payload = {"cfg": cfg, "extra": extra or {}}
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=_json_default)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def make_run_id(prefix: str, cfg: Optional[Dict[str, Any]] = None, extra: Optional[Dict[str, Any]] = None) -> str:
    # Include microseconds to reduce the chance of collisions when runs start quickly.
    ts = _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    fp = cfg_fingerprint(cfg or {}, extra=extra)
    return f"{prefix}_{ts}_{fp[:10]}"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, data: Any) -> None:
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)

