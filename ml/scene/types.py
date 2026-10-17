"""Совместимое расширение SceneState (без изменения замороженного контракта).

``SceneStateEx`` — подкласс ``SceneState``: потребители, читающие только ``t``/``signals``, работают
как раньше (isinstance(x, SceneState) истинно). Дополнительно на каждый сигнал:

* ``conf[name]``       — 0..1, насколько значение подтверждено наблюдением (0 = неизвестно);
* ``known[name]``      — conf ≥ порога: значение можно использовать как «да»/«нет»;
* ``visibility[name]`` — 0..1, доля видимости объекта/ROI на кадре (1 − окклюзия людьми);
* ``provenance[name]`` — откуда значение: det | roi | geom | traffic | track | pose | det_fallback |
  derived | prior | none (через «+», если источников несколько);
* ``observed``         — сырые наблюдаемые признаки вне SIGNALS (например unknown_person_in_cashier_area);
* ``quality``          — 0..1 качество кадра (резкость/экспозиция).

Правило: «не обнаружили» ≠ «отсутствует». Неизвестное значение в ``signals`` остаётся
консервативным (не тянет к нарушению), а ``known[name] = False`` говорит, что это не факт.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ._contract import SceneState

KNOWN_CONF = 0.5  # conf ≥ KNOWN_CONF → значение подтверждено наблюдением


@dataclass
class SceneStateEx(SceneState):
    conf: dict[str, float] = field(default_factory=dict)
    known: dict[str, bool] = field(default_factory=dict)
    visibility: dict[str, float] = field(default_factory=dict)
    provenance: dict[str, str] = field(default_factory=dict)
    observed: dict[str, float] = field(default_factory=dict)
    quality: float = 1.0


def signal_value(state: Any, name: str, min_conf: float = KNOWN_CONF) -> Optional[float]:
    """Значение сигнала или None, если оно не подтверждено наблюдением (для правил A09/A12)."""
    val = getattr(state, "signals", {}).get(name)
    if val is None:
        return None
    conf = getattr(state, "conf", None)
    if conf is None:  # обычный SceneState без расширения — доверяем как есть
        return float(val)
    return float(val) if float(conf.get(name, 0.0)) >= min_conf else None


def is_known(state: Any, name: str, min_conf: float = KNOWN_CONF) -> bool:
    return signal_value(state, name, min_conf) is not None


def state_to_dict(s: Any) -> dict[str, Any]:
    d: dict[str, Any] = {"t": float(s.t), "signals": {k: float(v) for k, v in s.signals.items()}}
    for k in ("conf", "visibility", "observed"):
        if hasattr(s, k):
            d[k] = {kk: float(vv) for kk, vv in getattr(s, k).items()}
    if hasattr(s, "known"):
        d["known"] = {kk: bool(vv) for kk, vv in s.known.items()}
    if hasattr(s, "provenance"):
        d["provenance"] = dict(s.provenance)
    if hasattr(s, "quality"):
        d["quality"] = float(s.quality)
    return d


def state_from_dict(d: dict[str, Any]) -> SceneStateEx:
    return SceneStateEx(
        t=float(d["t"]),
        signals={k: float(v) for k, v in d.get("signals", {}).items()},
        conf={k: float(v) for k, v in d.get("conf", {}).items()},
        known={k: bool(v) for k, v in d.get("known", {}).items()},
        visibility={k: float(v) for k, v in d.get("visibility", {}).items()},
        provenance=dict(d.get("provenance", {})),
        observed={k: float(v) for k, v in d.get("observed", {}).items()},
        quality=float(d.get("quality", 1.0)),
    )
