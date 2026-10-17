"""Совместимые дополнения к контракту v1 (v1.1-additive, владелец A01).

Dataclass-ы v1 в ``sentinel_contracts.py`` НЕ меняются. Всё новое живёт в уже существующих
расширяемых полях — ``AnalysisResult.meta`` и ``Event.extra`` — и имеет значения по умолчанию,
поэтому модули, которые об этих ключах не знают, продолжают работать. См. ``contracts/MIGRATION.md``.

Ключи ``AnalysisResult.meta`` (все необязательные):

* ``completeness`` — насколько анализ полон (``Completeness.to_dict()``). Нет ключа -> «нет сведений»
  (трактуется как полный — совместимость со старыми модулями).
* ``provenance``  — откуда результат: версия пайплайна/контракта, mock-режим, модели (``Provenance``).
* ``duration_sec`` — длительность видео (для проверки таймкодов).
* ``warnings``    — list[str], предупреждения для человека.
* ``violation_extra`` — list[dict], выровнен по индексу с ``result.violations`` (provenance нарушения,
  t_peak-trace и т.п.) — попадает только в UI-report.
* ``progress`` / ``stage`` / ``error`` — прогресс/стадия/текст ошибки задачи.

``VERDICT_STATES`` — состояние вердикта для UI: ``unknown`` — анализ не дал права на бинарный вердикт
(failed / неполный анализ / нет вердикта); жюри в этом случае получает ``status=failed, verdict=null``.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional

__all__ = [
    "META_COMPLETENESS",
    "META_PROVENANCE",
    "META_DURATION",
    "META_WARNINGS",
    "META_VIOLATION_EXTRA",
    "STAGE_STATUSES",
    "VERDICT_STATES",
    "UNKNOWN_TYPE_ID",
    "Completeness",
    "Provenance",
    "analysis_complete",
    "video_duration",
    "violation_extra",
]

META_COMPLETENESS = "completeness"
META_PROVENANCE = "provenance"
META_DURATION = "duration_sec"
META_WARNINGS = "warnings"
META_VIOLATION_EXTRA = "violation_extra"

StageStatus = Literal["ok", "skipped", "failed", "timeout", "mock"]
STAGE_STATUSES: tuple[str, ...] = ("ok", "skipped", "failed", "timeout", "mock")
VerdictState = Literal["violation", "no_violation", "unknown", "pending"]
VERDICT_STATES: tuple[str, ...] = ("violation", "no_violation", "unknown", "pending")
UNKNOWN_TYPE_ID = "V99"  # нарушение без известного типа -> catch-all из norms/catalog.yaml


@dataclass
class Completeness:
    """Полнота анализа. ``complete`` ставит оркестратор (A03): ядро анализа покрыло всё видео.

    Пропуск необязательной стадии (например, VLM по бюджету) полноту не нарушает —
    перечислите обязательные стадии в ``required_stages``.
    """

    complete: bool = True
    analyzed_until_sec: Optional[float] = None
    duration_sec: Optional[float] = None
    stages: dict[str, str] = field(default_factory=dict)          # stage -> STAGE_STATUSES
    required_stages: list[str] = field(default_factory=list)
    reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_meta(cls, meta: Mapping[str, Any] | None) -> Optional["Completeness"]:
        raw = (meta or {}).get(META_COMPLETENESS)
        if isinstance(raw, bool):
            return cls(complete=raw)
        if not isinstance(raw, Mapping):
            return None
        known = {k: raw[k] for k in ("complete", "analyzed_until_sec", "duration_sec", "stages",
                                      "required_stages", "reason") if k in raw}
        return cls(**known)

    def is_complete(self) -> bool:
        if not self.complete:
            return False
        for st in self.required_stages:
            if self.stages.get(st, "ok") in ("failed", "timeout", "skipped"):
                return False
        return True


@dataclass
class Provenance:
    """Происхождение результата (для UI, отчёта и раздела «Архитектура/локальность»)."""

    pipeline_version: str = ""
    contract_version: str = "1"
    mock: bool = False
    models: list[dict[str, Any]] = field(default_factory=list)   # как в <модуль>/models.json
    sources: list[str] = field(default_factory=list)            # rules | temporal | vlm | fusion

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analysis_complete(meta: Mapping[str, Any] | None) -> Optional[bool]:
    """True/False, если пайплайн сообщил полноту; None — сведений нет (совместимость: считать полным)."""
    c = Completeness.from_meta(meta)
    return None if c is None else c.is_complete()


def video_duration(meta: Mapping[str, Any] | None) -> Optional[float]:
    """Длительность видео из meta (``duration_sec`` / ``duration`` / ``completeness.duration_sec``)."""
    meta = meta or {}
    candidates = [meta.get(META_DURATION), meta.get("duration")]
    comp = meta.get(META_COMPLETENESS)
    if isinstance(comp, Mapping):
        candidates.append(comp.get("duration_sec"))
    for c in candidates:
        try:
            f = float(c)
        except (TypeError, ValueError):
            continue
        if f == f and f not in (float("inf"), float("-inf")) and f > 0:
            return f
    return None


def violation_extra(meta: Mapping[str, Any] | None, index: int) -> dict[str, Any]:
    """Доп. сведения о нарушении ``index`` из ``meta['violation_extra']`` (или {})."""
    extra = (meta or {}).get(META_VIOLATION_EXTRA)
    if isinstance(extra, list) and 0 <= index < len(extra) and isinstance(extra[index], Mapping):
        return dict(extra[index])
    return {}
