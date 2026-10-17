"""Загрузка ml/detect/config.yaml (+ необязательный override через env SENTINEL_DETECT_CONFIG)."""
from __future__ import annotations

import copy
import functools
import os
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(__file__).with_name("config.yaml")


def deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: ожидался YAML-словарь")
    return data


@functools.lru_cache(maxsize=8)
def _load_cached(override_path: str) -> dict[str, Any]:
    cfg = _read_yaml(CONFIG_PATH)
    if override_path:
        cfg = deep_merge(cfg, _read_yaml(Path(override_path)))
    return cfg


def load_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Возвращает копию конфига; overrides (dict) применяются поверх (удобно в тестах/тюнинге)."""
    cfg = copy.deepcopy(_load_cached(os.environ.get("SENTINEL_DETECT_CONFIG", "")))
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def label_threshold(table: dict[str, float], label: str) -> float:
    return float(table.get(label, table.get("default", 0.5)))


def env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")
