"""Совместимые дополнения к контракту v1 (v1.1-additive, владелец A01).

Dataclass-ы v1 в ``sentinel_contracts.py`` НЕ меняются. Всё новое живёт в расширяемых полях
(``AnalysisResult.meta``, ``Event.extra``, ``SceneState.signals``) и имеет значения по умолчанию.
Подробности и что делать модулям — ``contracts/MIGRATION.md`` и ``contracts/COMPAT.md``.

Ключи ``AnalysisResult.meta`` (все необязательные, кроме оговорённого):

* ``coverage`` — полнота анализа (``Coverage.to_dict()`` или просто доля 0..1). **Без подтверждённого покрытия
  вердикт «нет нарушения» не выдаётся** — ответ ``failed`` с причиной (неполный анализ ≠ «нарушений нет»).
* ``mock`` (bool, так пишет A03) / ``provenance.mock`` — результат получен на mock-бэкенде. Production сам на mock
  не переключается: такой результат допустим только при явном ``SENTINEL_MOCK=1``.
* ``stages`` / ``required_stages`` / ``complete`` — как пишет A03 (``meta.stages[имя].status``).
* ``duration_sec`` (или ``video.duration``) — длительность для проверки таймкодов.
* ``warnings`` — list[str]; ``violation_extra`` — list[dict] по индексу нарушения (только UI-report).

Неизвестно ≠ нет: если сигнал не наблюдался (человек/объект не виден, темно, перекрыто), его НЕТ в
``SceneState.signals`` (или он в ``unknown_signals`` события), а не 0.0. ``signal_value`` возвращает ``None``.
"""
from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

__all__ = [
    "META_COVERAGE",
    "META_PROVENANCE",
    "META_DURATION",
    "META_WARNINGS",
    "META_VIOLATION_EXTRA",
    "OK_STAGE_STATUSES",
    "VERDICT_STATES",
    "TYPE_REGISTRY",
    "UNKNOWN_TYPE_ID",
    "EXTRA_SIGNALS",
    "EXTRA_LABELS",
    "Coverage",
    "Provenance",
    "compute_coverage",
    "coverage_problem",
    "is_mock_result",
    "mock_allowed",
    "signal_value",
    "video_duration",
    "violation_extra",
]

META_COVERAGE = "coverage"
META_PROVENANCE = "provenance"
META_DURATION = "duration_sec"
META_WARNINGS = "warnings"
META_VIOLATION_EXTRA = "violation_extra"

OK_STAGE_STATUSES: frozenset[str] = frozenset({"ok", "done", "cached"})
VERDICT_STATES: tuple[str, ...] = ("violation", "no_violation", "unknown", "pending")
UNKNOWN_TYPE_ID = "V99"

# Реестр типов (ключи norms/catalog.yaml). Названия — запасные, источник истины — каталог A13.
TYPE_REGISTRY: dict[str, str] = {
    "V01": "Посторонний в кассовом помещении",
    "V02": "Кассир покинул рабочее место, оставив ценности без присмотра",
    "V03": "Сейф открыт и оставлен без присмотра",
    "V04": "Дверь кассового помещения открыта",
    "V05": "Использование мобильного телефона при работе с ценностями",
    "V06": "Приём пищи или напитков на рабочем месте с наличными",
    "V07": "Нарушение порядка приёма-передачи инкассаторской сумки",
    "V08": "Операции с наличными вне обзора камеры",
    "V09": "Личные вещи на кассовом столе",
    "V99": "Иное подозрительное действие",
}

# Дополнительный словарь (v1.1-additive): сигналы и метки сверх замороженных SIGNALS/LABELS.
EXTRA_SIGNALS: tuple[str, ...] = ("personal_items_on_desk", "scene_visibility", "second_outsider")
EXTRA_LABELS: tuple[str, ...] = ("personal_item",)


@dataclass
class Coverage:
    """Полнота анализа видео. Считает оркестратор (A03), проверяет сериализатор перед «нет нарушения»."""

    ratio: float = 0.0                       # доля длительности, реально проанализированная ядром (без unknown)
    analyzed_sec: float = 0.0
    duration_sec: float = 0.0
    unknown_spans: list[list[float]] = field(default_factory=list)  # [[t0, t1], ...] не видно/не декодировано
    stages: dict[str, str] = field(default_factory=dict)            # stage -> status
    required: list[str] = field(default_factory=list)
    complete: bool = False                   # обязательные стадии отработали на всём видео
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Provenance:
    """Происхождение результата (UI, отчёт, раздел «Архитектура/локальность»)."""

    pipeline_version: str = ""
    contract_version: str = "1"
    mock: bool = False
    models: list[dict[str, Any]] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _f(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v and v not in (float("inf"), float("-inf")) else None


def _merge_spans(spans: Iterable[Iterable[float]], lo: float, hi: float) -> list[list[float]]:
    clean = sorted([max(lo, a), min(hi, b)] for a, b in ((_f(s[0]), _f(s[1])) for s in spans)
                   if a is not None and b is not None and b > a)
    out: list[list[float]] = []
    for a, b in clean:
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def compute_coverage(duration_sec: float, analyzed_until_sec: float, *, unknown_spans: Iterable = (),
                     stages: Mapping[str, Any] | None = None, required: Iterable[str] = (),
                     source: str = "pipeline") -> dict[str, Any]:
    """Готовый ``meta['coverage']`` для оркестратора.

    ``stages`` — как ``meta['stages']`` у A03 (значение — dict со ``status`` или строка статуса).
    """
    dur = max(0.0, _f(duration_sec) or 0.0)
    until = min(dur, max(0.0, _f(analyzed_until_sec) or 0.0))
    unk = _merge_spans(unknown_spans, 0.0, until)
    unknown_total = sum(b - a for a, b in unk)
    analyzed = max(0.0, until - unknown_total)
    st = {k: (v.get("status") if isinstance(v, Mapping) else v) for k, v in (stages or {}).items()}
    st = {k: str(v) for k, v in st.items() if v is not None}
    req = list(required)
    # «mock» тут считается отработавшей стадией; допустим ли mock — решает coverage_problem(allow_mock)
    req_ok = all(st.get(r) in OK_STAGE_STATUSES or st.get(r) == "mock" for r in req)
    ratio = analyzed / dur if dur > 0 else 0.0
    return Coverage(ratio=round(ratio, 4), analyzed_sec=round(analyzed, 3), duration_sec=round(dur, 3),
                    unknown_spans=[[round(a, 3), round(b, 3)] for a, b in unk], stages=st, required=req,
                    complete=bool(req_ok and until >= dur - 1e-6), source=source).to_dict()


def mock_allowed() -> bool:
    """Mock-результаты допустимы только при явно включённом mock-режиме (тесты, демо)."""
    return os.environ.get("SENTINEL_MOCK", "").strip().lower() in ("1", "true", "yes", "on")


def is_mock_result(meta: Mapping[str, Any] | None) -> bool:
    meta = meta or {}
    if meta.get("mock") is True:
        return True
    prov = meta.get(META_PROVENANCE)
    if isinstance(prov, Mapping) and prov.get("mock") is True:
        return True
    for src in (meta.get("stages"), (meta.get(META_COVERAGE) or {}).get("stages")
                if isinstance(meta.get(META_COVERAGE), Mapping) else None):
        if isinstance(src, Mapping):
            for v in src.values():
                status = v.get("status") if isinstance(v, Mapping) else v
                if status == "mock":
                    return True
    return False


def coverage_problem(meta: Mapping[str, Any] | None, min_ratio: float, allow_mock: bool = False
                     ) -> Optional[str]:
    """None — полнота подтверждена и «нет нарушения» допустимо; иначе — причина для ``failed``."""
    meta = meta or {}
    if meta.get("complete") is False:
        return "оркестратор сообщил незавершённый анализ (meta.complete=false)"
    cov = meta.get(META_COVERAGE)
    if cov is None:
        return "полнота анализа не подтверждена: нет meta.coverage"
    if not isinstance(cov, Mapping):
        ratio = _f(cov)
        if ratio is None:
            return f"meta.coverage не число и не объект: {cov!r}"
        cov = {"ratio": ratio, "complete": True}
    ratio = _f(cov.get("ratio"))
    if ratio is None:
        return "meta.coverage.ratio не задан"
    if ratio + 1e-9 < min_ratio:
        return f"проанализировано {ratio:.0%} видео (< {min_ratio:.0%}): неизвестные участки ≠ «нарушений нет»"
    if cov.get("complete") is False:
        return "meta.coverage.complete=false"
    stages = cov.get("stages") or {}
    required = list(cov.get("required") or meta.get("required_stages") or [])
    top_stages = meta.get("stages") if isinstance(meta.get("stages"), Mapping) else {}
    for r in required:
        status = stages.get(r)
        if status is None and r in top_stages:
            status = top_stages[r].get("status") if isinstance(top_stages[r], Mapping) else top_stages[r]
        if status == "mock" and allow_mock:
            continue
        if status not in OK_STAGE_STATUSES:
            return f"обязательная стадия {r}: {status or 'нет статуса'}"
    return None


def signal_value(signals: Mapping[str, Any] | None, name: str) -> Optional[float]:
    """Значение сигнала или None, если он не наблюдался («неизвестно», не «нет»)."""
    if not signals or name not in signals:
        return None
    return _f(signals[name])


def video_duration(meta: Mapping[str, Any] | None) -> Optional[float]:
    """Длительность видео из meta: ``duration_sec`` / ``duration`` / ``video.duration`` / ``coverage.duration_sec``."""
    meta = meta or {}
    candidates = [meta.get(META_DURATION), meta.get("duration")]
    for key in ("video", META_COVERAGE):
        sub = meta.get(key)
        if isinstance(sub, Mapping):
            candidates += [sub.get("duration"), sub.get("duration_sec")]
    for c in candidates:
        f = _f(c)
        if f is not None and f > 0:
            return f
    return None


def violation_extra(meta: Mapping[str, Any] | None, index: int) -> dict[str, Any]:
    """Доп. сведения о нарушении ``index`` из ``meta['violation_extra']`` (или {})."""
    extra = (meta or {}).get(META_VIOLATION_EXTRA)
    if isinstance(extra, list) and 0 <= index < len(extra) and isinstance(extra[index], Mapping):
        return dict(extra[index])
    return {}
