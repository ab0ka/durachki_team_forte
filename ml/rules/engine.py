"""Движок правил: ряды сигналов → события-нарушения (``Event``, source="rules").

Конвейер для каждого правила из ``rules.yaml``:

1. ``strength`` (нечёткое выражение 0..1) × ``gate`` (булево условие) по всему ряду;
2. сглаживание (медиана/среднее в окне ``smooth.window_s``) — убирает одиночные вспышки;
3. гистерезис ``on``/``off`` → маска «правило активно»;
4. дебаунс: склейка сегментов с разрывом ≤ ``merge_gap_s``, отбрасывание короче ``min_duration_s``;
5. событие: ``t_peak`` по политике (``onset`` | ``onset_plus`` | ``max`` | ``center``),
   ``confidence`` от силы и длительности (монотонно по длительности), факты по шаблонам,
   ``bboxes`` участников на ``t_peak`` из треков, ``extra.trace`` — почему сработало.

Затем по всем правилам: слияние пересекающихся событий одного типа, «уступка» типов
(``yield_to``), NMS одного типа в окне ``nms_window_s``, порог уверенности, лимит числа событий.
"""
from __future__ import annotations

import math
import string
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .catalog import resolve_catalog, type_info
from .config import RuleSpec, RulesConfig, load_rules_config
from .expr import EvalContext
from .signals import SignalSeries, build_series, derive_from_tracks, polygon_bbox, track_box_at

try:
    from contracts.sentinel_contracts import Event  # noqa
except ImportError:  # пока ветка A01 не смерджена
    from ._contract_fallback import Event  # noqa

__all__ = ["run_rules", "evaluate", "RulesReport", "SIGNAL_RU", "hysteresis_mask", "mask_to_segments",
           "merge_segments", "confidence_for"]

SOURCE = "rules"

# Человекочитаемые названия сигналов — для trace/UI («почему сработало»).
SIGNAL_RU: dict[str, str] = {
    "cashier_present": "кассир в кадре",
    "cashier_in_workplace": "кассир на рабочем месте",
    "n_persons": "число людей в кадре",
    "outsider_in_cashier_area": "посторонний в зоне кассира",
    "collector_present": "инкассатор в кадре",
    "safe_open": "сейф открыт",
    "door_open": "дверь открыта",
    "cash_visible": "наличные видны",
    "cash_bag_visible": "инкассаторская сумка видна",
    "phone_in_hand": "телефон в руке",
    "phone_to_ear": "телефон у уха",
    "eating_drinking": "приём пищи/напитков",
    "hands_hidden": "руки скрыты от камеры",
    "cashier_facing_away": "кассир спиной к камере",
    "workplace_unattended_with_cash": "рабочее место с наличными без присмотра",
    "handover_motion": "движение передачи",
    "motion_energy": "интенсивность движения",
}


# --------------------------------------------------------------------------- helpers

def _r(x: float, nd: int = 3) -> float:
    x = float(x)
    return round(x, nd) if math.isfinite(x) else 0.0


def smooth(x: np.ndarray, dt: float, window_s: float, method: str) -> np.ndarray:
    if method == "none" or window_s <= 0 or x.shape[0] < 3:
        return x
    k = int(round(window_s / max(dt, 1e-6)))
    if k < 2:
        return x
    if k % 2 == 0:
        k += 1
    k = min(k, x.shape[0] if x.shape[0] % 2 == 1 else x.shape[0] - 1)
    if k < 3:
        return x
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(xp, k)
    if method == "median":
        return np.median(win, axis=1)
    return win.mean(axis=1)


def hysteresis_mask(x: np.ndarray, on: float, off: float) -> np.ndarray:
    """Включаемся при x ≥ on, держимся пока x ≥ off."""
    mask = np.zeros(x.shape[0], dtype=bool)
    active = False
    for i, v in enumerate(x):
        active = bool(v >= off) if active else bool(v >= on)
        mask[i] = active
    return mask


def mask_to_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Сегменты подряд идущих True как пары индексов (включительно)."""
    segs: list[tuple[int, int]] = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            segs.append((start, i - 1))
            start = None
    if start is not None:
        segs.append((start, int(mask.shape[0]) - 1))
    return segs


def merge_segments(times: np.ndarray, segs: list[tuple[int, int]], gap_s: float) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for s in segs:
        if out and float(times[s[0]] - times[out[-1][1]]) <= gap_s + 1e-9:
            out[-1] = (out[-1][0], s[1])
        else:
            out.append(s)
    return out


def confidence_for(rule: RuleSpec, mean_strength: float, max_strength: float, duration: float,
                   n_frames: int = 0, severity_factor: float = 1.0) -> tuple[float, dict[str, float]]:
    """Уверенность события; неубывающая по ``duration`` при фиксированной силе."""
    dur_term = 1.0 - math.exp(-max(0.0, duration) / rule.conf_tau_s)
    if rule.conf_expr is not None:
        from .expr import evaluate_scalar

        raw = evaluate_scalar(rule.conf_expr, {
            "mean_strength": mean_strength, "max_strength": max_strength, "duration": duration,
            "dur_term": dur_term, "base": rule.conf_base, "w_strength": rule.conf_w_strength,
            "w_duration": rule.conf_w_duration, "tau_s": rule.conf_tau_s, "n_frames": float(n_frames),
        })
        terms = {"expr": _r(raw)}
    else:
        s_term = rule.conf_w_strength * float(np.clip(mean_strength, 0.0, 1.0))
        d_term = rule.conf_w_duration * dur_term
        raw = rule.conf_base + s_term + d_term
        terms = {"base": _r(rule.conf_base), "strength": _r(s_term), "duration": _r(d_term)}
    factor = rule.weight * severity_factor
    conf = float(np.clip(raw * factor, rule.conf_min, rule.conf_max))
    terms["factor"] = _r(factor)
    return conf, terms


class _SafeFormatter(string.Formatter):
    """Только простые имена из контекста: без атрибутов/индексов; пропуски → «?»."""

    def get_field(self, field_name: str, args: Sequence[Any], kwargs: Mapping[str, Any]) -> Any:
        if not field_name.isidentifier():
            return "?", field_name
        return kwargs.get(field_name, "?"), field_name

    def format_field(self, value: Any, format_spec: str) -> str:
        try:
            return format(value, format_spec)
        except (TypeError, ValueError):
            return str(value)


_FMT = _SafeFormatter()


def format_fact(template: str, ctx: Mapping[str, Any]) -> str:
    try:
        return _FMT.vformat(template, (), dict(ctx))
    except (ValueError, IndexError, KeyError):
        return template


# --------------------------------------------------------------------------- report

@dataclass
class RulesReport:
    events: list[Any]
    series: SignalSeries
    strengths: dict[str, np.ndarray] = field(default_factory=dict)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    catalog_source: str = "fallback"
    config: Optional[RulesConfig] = None

    def risk_curve(self, step_s: float = 0.0) -> list[tuple[float, float]]:
        """Кривая риска по правилам: максимум силы активных правил в каждый момент."""
        t = self.series.times
        if t.shape[0] == 0:
            return []
        risk = np.zeros(t.shape[0])
        for arr in self.strengths.values():
            np.maximum(risk, arr, out=risk)
        pts = [(float(_r(ti, 2)), float(_r(ri))) for ti, ri in zip(t, np.clip(risk, 0, 1))]
        if step_s > 0 and len(pts) > 1:
            out, last = [], -1e9
            for p in pts:
                if p[0] - last >= step_s - 1e-9:
                    out.append(p)
                    last = p[0]
            return out
        return pts


# --------------------------------------------------------------------------- core

def _bboxes_at(rule: RuleSpec, t: float, tracks: Sequence[Any], zones: Sequence[Any],
               cfg: RulesConfig) -> tuple[dict[str, tuple[float, float, float, float]], list[int]]:
    boxes: dict[str, tuple[float, float, float, float]] = {}
    ids: list[int] = []
    for spec in rule.highlight:
        kind, _, value = spec.partition(":")
        if kind == "zone":
            for z in zones or ():
                name = getattr(z, "name", None) if not isinstance(z, Mapping) else z.get("name")
                poly = getattr(z, "polygon", None) if not isinstance(z, Mapping) else z.get("polygon")
                if name == value and poly:
                    bb = polygon_bbox(poly)
                    if bb:
                        boxes[f"zone:{name}"] = tuple(_r(v, 4) for v in bb)  # type: ignore[assignment]
            continue
        for tr in tracks or ():
            label = getattr(tr, "label", None) if not isinstance(tr, Mapping) else tr.get("label")
            role = getattr(tr, "role", None) if not isinstance(tr, Mapping) else tr.get("role")
            tid = getattr(tr, "track_id", None) if not isinstance(tr, Mapping) else tr.get("track_id")
            if kind == "role" and not (label == "person" and role == value):
                continue
            if kind == "label" and label != value:
                continue
            if kind not in ("role", "label"):
                continue
            bb = track_box_at(tr, t, cfg.bbox_tol_s)
            if bb is None:
                continue
            key = f"{role if kind == 'role' else label}#{tid}"
            boxes[key] = tuple(_r(min(1.0, max(0.0, v)), 4) for v in bb)  # type: ignore[assignment]
            if tid is not None:
                try:
                    ids.append(int(tid))
                except (TypeError, ValueError):
                    pass
            if len(boxes) >= cfg.max_bboxes:
                break
        if len(boxes) >= cfg.max_bboxes:
            break
    return boxes, sorted(set(ids))


def _signal_stats(series: SignalSeries, names: Sequence[str], i0: int, i1: int, ip: int) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for nm in sorted(names):
        arr = series.signals.get(nm)
        if arr is None or arr.shape[0] == 0:
            continue
        seg = arr[i0:i1 + 1]
        out[nm] = {"at_peak": _r(arr[ip]), "mean": _r(seg.mean()), "min": _r(seg.min()), "max": _r(seg.max())}
    return out


def _peak_time(rule: RuleSpec, times: np.ndarray, strength: np.ndarray, i0: int, i1: int) -> float:
    t_start, t_end = float(times[i0]), float(times[i1])
    if rule.t_peak_policy == "onset":
        return t_start
    if rule.t_peak_policy == "max":
        return float(times[i0 + int(np.argmax(strength[i0:i1 + 1]))])
    if rule.t_peak_policy == "center":
        return (t_start + t_end) / 2.0
    return float(np.clip(t_start + rule.t_peak_offset_s, 0.0, max(t_end, t_start)))


def _rule_events(rule: RuleSpec, series: SignalSeries, strength: np.ndarray, tracks: Sequence[Any],
                 zones: Sequence[Any], catalog: Mapping[str, Any], cfg: RulesConfig,
                 candidates: list[dict[str, Any]]) -> list[Any]:
    times = series.times
    mask = hysteresis_mask(strength, rule.on, rule.off)
    segs = merge_segments(times, mask_to_segments(mask), rule.merge_gap_s)
    info = type_info(catalog, rule.type_id)
    sev_factor = float(cfg.severity_factor.get(info["severity"], 1.0))
    missing = sorted(n for n in rule.names if n in series.defaulted)
    derived = {n: series.derived[n] for n in rule.names if n in series.derived}
    events = []
    for i0, i1 in segs:
        t_start, t_end = float(times[i0]), float(times[i1])
        duration = t_end - t_start + series.dt
        cand = {"rule_id": rule.id, "type_id": rule.type_id, "t_start": _r(t_start, 2), "t_end": _r(t_end, 2),
                "duration_s": _r(duration, 2)}
        if duration + 1e-9 < rule.min_duration_s:
            cand["rejected"] = f"короче min_duration_s={rule.min_duration_s}"
            candidates.append(cand)
            continue
        seg = strength[i0:i1 + 1]
        mean_s, max_s = float(seg.mean()), float(seg.max())
        t_peak = _peak_time(rule, times, strength, i0, i1)
        ip = int(np.argmin(np.abs(times - t_peak)))
        conf, terms = confidence_for(rule, mean_s, max_s, duration, i1 - i0 + 1, sev_factor)
        names = sorted(set(rule.names) | {"n_persons"})
        stats = _signal_stats(series, names, i0, i1, ip)
        ctx: dict[str, Any] = {
            "t_start": t_start, "t_end": t_end, "t_peak": t_peak, "duration": duration,
            "strength": mean_s, "strength_max": max_s, "confidence": conf, "type_name": info["name"],
        }
        for nm, st in stats.items():
            ctx[nm] = st["at_peak"]
            ctx[f"{nm}_mean"] = st["mean"]
            ctx[f"{nm}_min"] = st["min"]
            ctx[f"{nm}_max"] = st["max"]
        facts = [format_fact(tpl, ctx) for tpl in rule.facts]
        conditions = []
        for expr_obj in (rule.strength, rule.gate):
            if expr_obj is None:
                continue
            for c in expr_obj.comparisons:
                v = series.signals.get(c.name)
                val = float(v[ip]) if v is not None and v.shape[0] else float("nan")
                conditions.append({"signal": c.name, "label": SIGNAL_RU.get(c.name, c.name), "op": c.op,
                                   "threshold": c.threshold, "value": _r(val), "ok": c.check(val)})
        bboxes, track_ids = _bboxes_at(rule, t_peak, tracks, zones, cfg)
        trace = {
            "rule_id": rule.id,
            "type_id": rule.type_id,
            "comment": rule.comment,
            "strength_expr": rule.strength.source,
            "gate_expr": rule.gate.source if rule.gate is not None else None,
            "thresholds": {"on": rule.on, "off": rule.off, "min_duration_s": rule.min_duration_s,
                           "merge_gap_s": rule.merge_gap_s, "smooth_s": rule.smooth_s},
            "t_peak_policy": rule.t_peak_policy,
            "t_peak_offset_s": rule.t_peak_offset_s,
            "strength": {"mean": _r(mean_s), "max": _r(max_s), "at_peak": _r(strength[ip])},
            "duration_s": _r(duration, 2),
            "signals": {nm: dict(st, label=SIGNAL_RU.get(nm, nm)) for nm, st in stats.items()},
            "conditions": conditions,
            "confidence_terms": terms,
            "missing_signals": missing,
            "derived_signals": derived,
        }
        ev = Event(
            type_id=rule.type_id,
            t_start=_r(t_start, 2),
            t_end=_r(t_end, 2),
            t_peak=_r(t_peak, 2),
            confidence=_r(conf),
            source=SOURCE,
            facts=facts,
            bboxes=bboxes,
            track_ids=track_ids,
            extra={
                "rule_id": rule.id,
                "rule_ids": [rule.id],
                "type_name": info["name"],
                "severity": info["severity"],
                "in_catalog": info["in_catalog"],
                "trace": trace,
            },
        )
        cand["accepted"] = True
        cand["confidence"] = ev.confidence
        cand["t_peak"] = ev.t_peak
        candidates.append(cand)
        events.append(ev)
    events.sort(key=lambda e: -e.confidence)
    return events[: rule.max_events]


def _overlap(a: Any, b: Any, gap: float) -> bool:
    return a.t_start <= b.t_end + gap and b.t_start <= a.t_end + gap


def _merge_same_type(events: list[Any], cfg: RulesConfig) -> list[Any]:
    by_type: dict[str, list[Any]] = {}
    for ev in events:
        by_type.setdefault(ev.type_id, []).append(ev)
    out: list[Any] = []
    for type_id, evs in by_type.items():
        evs.sort(key=lambda e: e.t_start)
        groups: list[list[Any]] = []
        for ev in evs:
            if groups and any(_overlap(ev, g, cfg.same_type_merge_gap_s) for g in groups[-1]):
                groups[-1].append(ev)
            else:
                groups.append([ev])
        for g in groups:
            if len(g) == 1:
                out.append(g[0])
                continue
            primary = max(g, key=lambda e: e.confidence)
            first = min(g, key=lambda e: (e.t_start, e.t_peak))
            rule_ids = []
            for e in g:
                for rid in e.extra.get("rule_ids", []):
                    if rid not in rule_ids:
                        rule_ids.append(rid)
            facts: list[str] = []
            for e in sorted(g, key=lambda e: e.t_start):
                for f in e.facts:
                    if f not in facts:
                        facts.append(f)
            bboxes = dict(first.bboxes)
            for k, v in primary.bboxes.items():
                bboxes.setdefault(k, v)
            n_extra = len(rule_ids) - 1
            cap = max(e.extra.get("trace", {}).get("confidence_terms", {}).get("cap", 0.97) for e in g)
            conf = min(cap, primary.confidence + cfg.corroboration_bonus * max(0, n_extra))
            extra = dict(primary.extra)
            extra["rule_ids"] = rule_ids
            extra["merged"] = [{"rule_id": e.extra.get("rule_id"), "t_start": e.t_start, "t_end": e.t_end,
                                "t_peak": e.t_peak, "confidence": e.confidence} for e in g]
            extra["traces"] = [e.extra.get("trace") for e in g if e is not primary]
            merged = Event(
                type_id=type_id,
                t_start=min(e.t_start for e in g),
                t_end=max(e.t_end for e in g),
                t_peak=first.t_peak,
                confidence=_r(conf),
                source=SOURCE,
                facts=facts,
                bboxes=bboxes,
                track_ids=sorted({t for e in g for t in e.track_ids}),
                extra=extra,
            )
            out.append(merged)
    return out


def _apply_yield(events: list[Any], cfg: RulesConfig) -> None:
    rules = {r.id: r for r in cfg.rules}
    for ev in events:
        rule = rules.get(ev.extra.get("rule_id", ""))
        if rule is None or not rule.yield_to:
            continue
        winners = [o for o in events if o is not ev and o.type_id in rule.yield_to
                   and _overlap(ev, o, 0.0)]
        if winners:
            ev.confidence = _r(ev.confidence * rule.yield_factor)
            ev.extra["yielded_to"] = sorted({o.type_id for o in winners})


def _nms(events: list[Any], window_s: float) -> list[Any]:
    kept: list[Any] = []
    for ev in sorted(events, key=lambda e: -e.confidence):
        clash = next((k for k in kept if k.type_id == ev.type_id and abs(k.t_peak - ev.t_peak) < window_s), None)
        if clash is None:
            kept.append(ev)
        else:
            clash.extra.setdefault("suppressed", []).append(
                {"rule_id": ev.extra.get("rule_id"), "t_peak": ev.t_peak, "confidence": ev.confidence})
    return kept


def evaluate(states: Sequence[Any], tracks: Sequence[Any] = (), zones: Sequence[Any] = (),
             catalog: Optional[Mapping[str, Any]] = None, config: Optional[RulesConfig] = None) -> RulesReport:
    """Полный прогон правил с диагностикой (для replay, UI и интегратора)."""
    cfg = config or load_rules_config()
    cat, cat_src = resolve_catalog(catalog)
    warnings = list(cfg.warnings)
    needed: set[str] = set()
    for r in cfg.enabled_rules:
        needed |= r.names
    series = build_series(states, needed, defaults=cfg.signal_defaults, default_value=cfg.default_value,
                          ffill_max_s=cfg.ffill_max_s, merge_policy=cfg.merge_policy)
    report = RulesReport(events=[], series=series, warnings=warnings, catalog_source=cat_src, config=cfg)
    if series.n == 0:
        warnings.append("нет состояний сцены — правила не применялись")
        return report
    if cfg.derive_from_tracks and tracks:
        absent = {n for n in needed if n not in series.present}
        if absent:
            try:
                derived = derive_from_tracks(series.times, tracks, zones, presence_tol_s=cfg.presence_tol_s)
            except Exception as exc:  # pragma: no cover - страховка
                derived = {}
                warnings.append(f"производные сигналы из треков: {exc}")
            for name, arr in derived.items():
                if name in absent:
                    series.signals[name] = arr
                    series.derived[name] = "tracks"
                    series.defaulted.discard(name)
    ctx = EvalContext(times=series.times, signals=series.signals, default=cfg.default_value)
    events: list[Any] = []
    for rule in cfg.enabled_rules:
        try:
            s = np.clip(rule.strength(ctx), 0.0, 1.0)
            if rule.gate is not None:
                s = np.where(rule.gate(ctx) > 0.5, s, 0.0)
            s = np.clip(smooth(s, series.dt, rule.smooth_s, rule.smooth_method), 0.0, 1.0)
            report.strengths[rule.id] = s
            if all(n in series.defaulted for n in rule.strength.names) and rule.strength.names:
                warnings.append(f"{rule.id}: нет ни одного входного сигнала ({', '.join(sorted(rule.strength.names))})")
                continue
            events.extend(_rule_events(rule, series, s, tracks, zones, cat, cfg, report.candidates))
        except Exception as exc:  # одно сломанное правило не валит остальные
            warnings.append(f"{rule.id}: ошибка вычисления: {exc}")
    events = _merge_same_type(events, cfg)
    _apply_yield(events, cfg)
    events = _nms(events, cfg.nms_window_s)
    events = [e for e in events if e.confidence >= cfg.min_confidence]
    events.sort(key=lambda e: -e.confidence)
    events = events[: cfg.max_events]
    events.sort(key=lambda e: (e.t_peak, e.type_id))
    report.events = events
    return report


def run_rules(states: Sequence[Any], tracks: Sequence[Any], zones: Sequence[Any],
              catalog: Optional[Mapping[str, Any]], config: Optional[RulesConfig] = None) -> list[Any]:
    """Публичная функция модуля (сигнатура контракта v1 + необязательный ``config``)."""
    return evaluate(states, tracks, zones, catalog, config).events
