"""Полнота анализа: meta.coverage (доля длительности, реально проанализированная стадией) и
meta.completeness в формате contracts/extensions.py (A01).

Принципы:
* coverage считается по фактическим таймстемпам (кадры клипа, состояния сцены, точки риск-кривой),
  а не по флагу «стадия отработала»; провал между кадрами больше ``max_gap_s`` — непокрытое время;
* неизвестно ≠ 0 и ≠ 1: если стадия не сообщает, что именно она смотрела (vlm_discover), coverage=None;
* неполный анализ (сбой обязательной стадии ИЛИ coverage обязательной стадии < min_required)
  никогда не превращается в no_violation.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

from .config import cfg_get

# Статусы стадий в терминах contracts.extensions.STAGE_STATUSES
_STATUS_MAP = {"ok": "ok", "cached": "ok", "error": "failed", "crashed": "failed", "missing": "failed",
               "timeout": "timeout", "skipped": "skipped", "disabled": "skipped", "pending": "skipped",
               "cancelled": "failed"}

# Стадии, чьё покрытие = покрытие кадров клипа, если стадия завершилась успешно (контракт: стадия
# обрабатывает клип целиком, а выход содержит только найденное — пустой кадр выхода не оставляет).
_CLIP_WIDE = ("detect", "track", "pose")


def interval_coverage(times: Iterable[float], duration: Optional[float], max_gap_s: float = 2.0) -> Optional[float]:
    """Доля [0, duration], покрытая таймстемпами: кадр в t представляет [t, t + шаг), где шаг —
    медианный собственный шаг выборки, но не больше max_gap_s (редкая выборка ≠ полное покрытие)."""
    ts = sorted({float(t) for t in times if t is not None})
    if not duration or duration <= 0:
        return None
    if not ts:
        return 0.0
    if len(ts) == 1:
        win = max_gap_s
    else:
        diffs = sorted(b - a for a, b in zip(ts, ts[1:]) if b > a)
        step = diffs[len(diffs) // 2] if diffs else max_gap_s
        win = min(step, max_gap_s)
    covered = 0.0
    cur_a, cur_b = None, None
    for t in ts:
        a, b = max(0.0, t), min(duration, t + win)
        if b <= a:
            continue
        if cur_b is None or a > cur_b:
            if cur_b is not None:
                covered += cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    if cur_b is not None:
        covered += cur_b - cur_a
    return round(min(1.0, covered / duration), 4)


def reference_duration(clip_info: Mapping[str, Any], probe: Mapping[str, Any]) -> Optional[float]:
    d = float(clip_info.get("duration") or 0.0)
    if d <= 0:
        d = float(probe.get("duration") or 0.0)
    return d or None


def stage_coverage(records: Mapping[str, Any], outputs: Mapping[str, Any], clip_info: Mapping[str, Any],
                   probe: Mapping[str, Any], cfg: Mapping) -> dict[str, Optional[float]]:
    """coverage по стадиям: 0..1 или None (неизвестно)."""
    max_gap = float(cfg_get(cfg, "coverage.max_gap_s", 2.0))
    tol = float(cfg_get(cfg, "coverage.container_tolerance", 0.15))
    duration = reference_duration(clip_info, probe)
    out: dict[str, Optional[float]] = {}

    ingest = interval_coverage(clip_info.get("times") or [], duration, max_gap)
    container = float(probe.get("duration") or 0.0)
    if ingest is not None and duration and container > duration * (1.0 + tol):
        ingest = round(ingest * duration / container, 4)  # декодер дошёл не до конца файла
    out["ingest"] = ingest if _ok(records.get("ingest")) else 0.0

    for stage in _CLIP_WIDE:
        out[stage] = ingest if _ok(records.get(stage)) else (None if stage not in records else 0.0)
    for stage, idx in (("scene", 0), ("temporal", 0)):
        rec = records.get(stage)
        if not _ok(rec):
            out[stage] = None if rec is None else 0.0
            continue
        values = (outputs.get(stage) or (None,))[idx]
        if stage == "temporal":
            times = [t for t, _r in (values or [])]
        else:
            times = [getattr(s, "t", None) for s in (values or [])]
        out[stage] = interval_coverage(times, duration, max_gap) if times else (0.0 if duration else None)
        if out[stage] is not None and ingest is not None:
            out[stage] = min(out[stage], ingest)
    out["rules"] = out.get("scene") if _ok(records.get("rules")) else (None if "rules" not in records else 0.0)
    out["vlm_verify"] = None   # проверяет окна кандидатов, а не всё видео: см. meta.vlm_verified
    out["vlm_discover"] = None  # модуль не сообщает просмотренные окна → неизвестно
    out["fusion"] = 1.0 if _ok(records.get("fusion")) else (None if "fusion" not in records else 0.0)
    return out


def _ok(rec: Any) -> bool:
    return rec is not None and getattr(rec, "status", None) in ("ok", "cached")


def coverage_gate(coverage: Mapping[str, Optional[float]], required: Iterable[str], cfg: Mapping) -> list[str]:
    """Причины неполноты по покрытию обязательных стадий (пусто — покрытие достаточное)."""
    if not cfg_get(cfg, "coverage.gate", True):
        return []
    need = float(cfg_get(cfg, "coverage.min_required", 0.9))
    reasons = []
    for stage in required:
        if stage == "fusion":
            continue
        cov = coverage.get(stage)
        if cov is None:
            reasons.append(f"{stage}: покрытие неизвестно")
        elif cov < need:
            reasons.append(f"{stage}: покрыто {cov:.0%} длительности < {need:.0%}")
    return reasons


def completeness(records: Mapping[str, Any], required: list[str], complete: bool, duration: Optional[float],
                 analyzed_until: Optional[float], reason: Optional[str], mock: bool) -> dict:
    """meta.completeness в формате contracts.extensions.Completeness.to_dict()."""
    stages = {}
    for name, rec in records.items():
        st = _STATUS_MAP.get(getattr(rec, "status", "skipped"), "failed")
        if st == "ok" and (mock or getattr(rec, "origin", "") == "builtin-mock"):
            st = "mock"
        stages[name] = st
    return {"complete": bool(complete), "analyzed_until_sec": analyzed_until, "duration_sec": duration,
            "stages": stages, "required_stages": list(required), "reason": reason}
