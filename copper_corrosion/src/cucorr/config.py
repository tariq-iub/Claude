"""YAML configuration loading with deep-merge overrides and stable hashing."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict[str, Any]:
    path = Path(path) if path else PROJECT_ROOT / "configs" / "default.yaml"
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return deep_merge(cfg, overrides or {})


def resolve(cfg: dict, key: str) -> Path:
    """Resolve ``cfg['paths'][key]`` relative to the project root."""
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else PROJECT_ROOT / p


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]
