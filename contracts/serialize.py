"""Сериализация ``AnalysisResult`` -> JSON для API (владелец A01). Два сериализатора:

1. ``result_to_jury_api(result)`` — ответ ``GET /api/v1/jobs/{job_id}`` для жюри: РОВНО обязательная схема ТЗ,
   без предположения, что дополнительные поля будут приняты::

       {"job_id", "status", "verdict",
        "violations": [{"type", "timecode_sec", "description",
                        "evidence_frames": [{"timestamp_sec", "image_base64"}]}]}

   Ссылка на норму (первая из ``Violation.rules``, т.е. из norms/catalog.yaml) дописывается в ``description``,
   если её там ещё нет — других полей для нормы в схеме ТЗ нет.

2. ``result_to_ui_report(result)`` — расширенный отчёт для UI/отладки: всё из (1) + ``type_id, confidence,
   t_start, t_end, rules[], rule{}`` у нарушения и ``progress, stage, error, verdict_state, analysis_complete,
   warnings, candidates, timeline, meta`` на верхнем уровне.

``result_to_api(result)`` — прежнее имя из контракта; по умолчанию ``profile="jury"`` (строгая схема),
``profile="ui"`` — расширенный отчёт.

Политика вердикта (одна для обоих сериализаторов, ``resolve_outcome``):

* ``processing`` и ``failed`` — ``verdict = null``, ``violations = []`` (никаких фиктивных бинарных вердиктов);
* ``done`` + ``violation`` — только если есть хотя бы одно валидное нарушение (конечный таймкод), иначе ``failed``;
* ``done`` + ``no_violation`` — только после успешного анализа: если ``meta.completeness`` сообщает неполный
  анализ -> ``failed``. Нарушения при вердикте ``no_violation`` жюри не отдаются (в UI — ``candidates``);
* ``done`` без вердикта — ``violation``, если есть валидные нарушения, иначе ``failed`` (не выдумываем ``no_violation``);
* таймкоды конечные, >= 0, не больше длительности (если известна), округлены до 0.1 с; NaN/Inf в ответе нет;
  нарушение без конечного таймкода (и без запасного — кадр/``t_start``) отбрасывается с предупреждением;
* кадры: только валидный base64 JPEG (сигнатуры FF D8 FF ... FF D9), без префикса ``data:``; жюри — не больше 3.

Модуль без тяжёлых зависимостей и без побочных эффектов при импорте.
"""
from __future__ import annotations

import base64
import binascii
import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Optional

from .extensions import META_WARNINGS, analysis_complete, video_duration, violation_extra
from .sentinel_contracts import AnalysisResult, EvidenceFrame, RuleRef, Violation

__all__ = [
    "API_REQUIRED_TOP",
    "API_REQUIRED_VIOLATION",
    "API_REQUIRED_FRAME",
    "DEFAULT_TYPE_NAMES",
    "MAX_JURY_FRAMES",
    "TIME_DECIMALS",
    "Outcome",
    "resolve_outcome",
    "result_to_api",
    "result_to_jury_api",
    "result_to_ui_report",
    "api_to_result",
    "minimal_api",
    "rule_to_api",
    "jpeg_b64_problem",
    "to_jsonable",
]

API_REQUIRED_TOP = ("job_id", "status", "verdict", "violations")
API_REQUIRED_VIOLATION = ("type", "timecode_sec", "description", "evidence_frames")
API_REQUIRED_FRAME = ("timestamp_sec", "image_base64")

TIME_DECIMALS = 1      # таймкоды — до 0.1 с
SCORE_DECIMALS = 3     # confidence / risk / progress
MAX_JURY_FRAMES = 3    # ТЗ: 1–3 кадра-доказательства

_STATUSES = ("processing", "done", "failed")
_VERDICTS = ("violation", "no_violation")

# Запасные русские названия — ТОЛЬКО если fusion не заполнил ``Violation.type``.
# Источник истины — norms/catalog.yaml (A13); здесь нет никаких ссылок на нормы.
DEFAULT_TYPE_NAMES: dict[str, str] = {
    "V01": "Посторонний в кассовом помещении",
    "V02": "Кассир покинул рабочее место, оставив ценности без присмотра",
    "V03": "Сейф открыт и оставлен без присмотра",
    "V04": "Дверь кассового помещения открыта",
    "V05": "Использование мобильного телефона при работе с ценностями",
    "V06": "Приём пищи или напитков на рабочем месте с наличными",
    "V07": "Нарушение порядка приёма-передачи инкассаторской сумки",
    "V08": "Операции с наличными вне обзора камеры",
    "V99": "Иное подозрительное действие",
}

_DATA_URI_MARK = ";base64,"


# ----------------------------------------------------------------------------- helpers

def _finite(x: Any) -> float | None:
    """float(x), если это конечное число, иначе None (NaN/Inf/None/мусор)."""
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _round_t(x: float) -> float:
    return max(0.0, round(x, TIME_DECIMALS)) + 0.0


def _score(x: Any) -> float:
    f = _finite(x)
    return 0.0 if f is None else min(1.0, max(0.0, round(f, SCORE_DECIMALS))) + 0.0


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Атрибут dataclass-а или ключ словаря — сериализатор принимает и то и другое."""
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _clean_b64(data: Any) -> str:
    """Чистый base64: bytes JPEG кодируем, у data URI отрезаем префикс, пробелы/переводы строк убираем."""
    if data is None:
        return ""
    if isinstance(data, (bytes, bytearray, memoryview)):
        raw = bytes(data)
        if raw[:3] == b"\xff\xd8\xff" or not raw.isascii():
            return base64.b64encode(raw).decode("ascii")
        data = raw.decode("ascii")
    s = str(data).strip()
    if s.startswith("data:") and _DATA_URI_MARK in s[:64]:
        s = s.split(_DATA_URI_MARK, 1)[1]
    if any(c.isspace() for c in s):
        s = "".join(s.split())
    return s


def jpeg_b64_problem(s: Any) -> Optional[str]:
    """None, если ``s`` — валидный base64 JPEG (SOI FF D8 FF в начале, EOI FF D9 в конце), иначе причина."""
    if not isinstance(s, str) or not s:
        return "пустой image_base64"
    if s.startswith("data:"):
        return "префикс data URI (нужен чистый base64)"
    try:
        raw = base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError):
        return "не декодируется как base64"
    if not raw.startswith(b"\xff\xd8\xff"):
        return "не JPEG (нет сигнатуры FF D8 FF)"
    if b"\xff\xd9" not in raw[-64:]:
        return "JPEG обрезан (нет маркера конца FF D9)"
    return None


def to_jsonable(obj: Any, _depth: int = 0) -> Any:
    """Рекурсивно приводит объект к JSON-совместимому виду (без импорта numpy/torch).

    NaN/Inf -> None; numpy-скаляры/массивы -> python; dataclass -> dict; set/tuple -> list; прочее -> str.
    """
    if _depth > 32:
        return str(obj)
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name), _depth + 1) for f in dataclasses.fields(obj)}
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        items = sorted(obj, key=repr) if isinstance(obj, (set, frozenset)) else obj
        return [to_jsonable(v, _depth + 1) for v in items]
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return base64.b64encode(bytes(obj)).decode("ascii")
    for conv in ("tolist", "item"):  # numpy / torch — утиная типизация
        fn = getattr(obj, conv, None)
        if callable(fn):
            try:
                return to_jsonable(fn(), _depth + 1)
            except Exception:  # noqa: BLE001
                pass
    return str(obj)


def rule_to_api(rule: RuleRef | Mapping[str, Any]) -> dict[str, Any]:
    clause = _get(rule, "clause")
    return {
        "document": str(_get(rule, "document", "") or ""),
        "clause": None if clause is None else str(clause),
        "url": str(_get(rule, "url", "") or ""),
        "quote": str(_get(rule, "quote", "") or ""),
        "verified": bool(_get(rule, "verified", False)),
    }


# ----------------------------------------------------------------------------- outcome policy

@dataclass
class Outcome:
    """Решение о статусе/вердикте + нормализованные нарушения (UI-формат) — общее для обоих сериализаторов."""

    job_id: str
    status: str                                   # processing | done | failed
    verdict: Optional[str]                        # violation | no_violation | None
    verdict_state: str                            # violation | no_violation | unknown | pending
    violations: list[dict[str, Any]] = field(default_factory=list)   # отдаются (жюри и UI)
    candidates: list[dict[str, Any]] = field(default_factory=list)   # только UI (при вердикте no_violation)
    error: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    analysis_complete: Optional[bool] = None


def _normalize_violation(v: Any, index: int, duration: Optional[float], meta: Mapping[str, Any],
                         warnings: list[str]) -> Optional[dict[str, Any]]:
    type_id = str(_get(v, "type_id", "") or "").strip()
    vtype = str(_get(v, "type", "") or "").strip() or DEFAULT_TYPE_NAMES.get(type_id, "") or "Возможное нарушение"

    frames_raw = list(_get(v, "evidence_frames") or [])
    frames: list[dict[str, Any]] = []
    for j, fr in enumerate(frames_raw):
        ts = _finite(_get(fr, "timestamp_sec"))
        b64 = _clean_b64(_get(fr, "image_base64"))
        problem = jpeg_b64_problem(b64)
        if ts is None or ts < 0:
            warnings.append(f"нарушение {index}, кадр {j}: нет конечного timestamp_sec — кадр отброшен")
            continue
        if problem:
            warnings.append(f"нарушение {index}, кадр {j}: {problem} — кадр отброшен")
            continue
        frames.append({"timestamp_sec": ts, "image_base64": b64})

    tc = _finite(_get(v, "timecode_sec"))
    if tc is None or tc < 0:
        fallback = next((f["timestamp_sec"] for f in frames), None)
        if fallback is None:
            fallback = _finite(_get(v, "t_start"))
        if fallback is None or fallback < 0:
            warnings.append(f"нарушение {index} ({type_id or vtype}): нет конечного таймкода — не отдаётся")
            return None
        warnings.append(f"нарушение {index}: timecode_sec не задан, взят {fallback:.1f} с из кадра/t_start")
        tc = fallback
    if duration is not None and tc > duration:
        warnings.append(f"нарушение {index}: timecode_sec={tc:.1f} > длительности {duration:.1f} — обрезан")
        tc = duration

    t_start = _finite(_get(v, "t_start"))
    t_end = _finite(_get(v, "t_end"))
    t_start = tc if t_start is None or t_start < 0 else t_start
    t_end = tc if t_end is None or t_end < 0 else t_end
    if t_end < t_start:
        t_start, t_end = t_end, t_start
    t_start, t_end = min(t_start, tc), max(t_end, tc)
    if duration is not None:
        t_end = min(t_end, duration)

    description = str(_get(v, "description", "") or "").strip()
    if not description:
        description = f"Признаки возможного нарушения ({vtype}) на {tc:.1f} с. Требуется проверка человеком."
    rules = [rule_to_api(r) for r in (_get(v, "rules") or [])]
    if not frames:
        warnings.append(f"нарушение {index}: нет валидных кадров-доказательств")

    out: dict[str, Any] = {
        "type": vtype,
        "timecode_sec": _round_t(tc),
        "description": description,
        "evidence_frames": [{"timestamp_sec": _round_t(f["timestamp_sec"]), "image_base64": f["image_base64"]}
                            for f in frames],
        "type_id": type_id,
        "confidence": _score(_get(v, "confidence", 0.0)),
        "t_start": _round_t(t_start),
        "t_end": _round_t(t_end),
        "rules": rules,
        "rule": rules[0] if rules else None,
    }
    extra = violation_extra(meta, index)
    if extra:
        out["extra"] = to_jsonable(extra)
    return out


def resolve_outcome(result: AnalysisResult | Mapping[str, Any]) -> Outcome:
    """Единая политика статуса/вердикта (см. docstring модуля)."""
    job_id = str(_get(result, "job_id", "") or "")
    status = str(_get(result, "status", "failed") or "failed")
    meta_raw = _get(result, "meta") or {}
    meta: Mapping[str, Any] = meta_raw if isinstance(meta_raw, Mapping) else {}
    warnings: list[str] = [str(w) for w in (meta.get(META_WARNINGS) or []) if w is not None] \
        if isinstance(meta.get(META_WARNINGS), list) else []
    complete = analysis_complete(meta)
    if status not in _STATUSES:
        warnings.append(f"недопустимый status={status!r} -> failed")
        return Outcome(job_id, "failed", None, "unknown", error=f"недопустимый статус {status!r}",
                       warnings=warnings, analysis_complete=complete)
    if status == "processing":
        return Outcome(job_id, "processing", None, "pending", warnings=warnings, analysis_complete=complete)
    if status == "failed":
        err = meta.get("error")
        return Outcome(job_id, "failed", None, "unknown", error=str(err) if err else "Анализ завершился с ошибкой",
                       warnings=warnings, analysis_complete=complete)

    duration = video_duration(meta)
    valid = []
    for i, v in enumerate(_get(result, "violations") or []):
        nv = _normalize_violation(v, i, duration, meta, warnings)
        if nv is not None:
            valid.append(nv)
    verdict = _get(result, "verdict")

    def failed(reason: str) -> Outcome:
        warnings.append(reason)
        return Outcome(job_id, "failed", None, "unknown", candidates=valid, error=reason, warnings=warnings,
                       analysis_complete=complete)

    if verdict == "violation":
        if not valid:
            return failed("verdict=violation, но нет ни одного нарушения с конечным таймкодом")
        return Outcome(job_id, "done", "violation", "violation", violations=valid, warnings=warnings,
                       analysis_complete=complete)
    if verdict == "no_violation":
        if complete is False:
            return failed("анализ неполный (meta.completeness) — вердикт no_violation не выдаётся")
        if valid:
            warnings.append("verdict=no_violation: найденные кандидаты не отдаются жюри (см. candidates)")
        return Outcome(job_id, "done", "no_violation", "no_violation", candidates=valid, warnings=warnings,
                       analysis_complete=complete)
    if verdict is not None:
        warnings.append(f"недопустимый verdict={verdict!r}")
    if valid:
        warnings.append("verdict не задан — выведен 'violation' по списку нарушений")
        return Outcome(job_id, "done", "violation", "violation", violations=valid, warnings=warnings,
                       analysis_complete=complete)
    return failed("анализ не выдал вердикт — no_violation не выдумываем")


# ----------------------------------------------------------------------------- serializers

def _jury_description(v: Mapping[str, Any]) -> str:
    desc = v["description"]
    rule = v.get("rule")
    if rule and rule.get("url") and rule["url"] not in desc:
        ref = rule.get("document") or ""
        if rule.get("clause"):
            ref = f"{ref}, {rule['clause']}" if ref else str(rule["clause"])
        desc = f"{desc.rstrip()} Норма: {ref + ' — ' if ref else ''}{rule['url']}"
    return desc


def result_to_jury_api(result: AnalysisResult | Mapping[str, Any], *, embed_rule_in_description: bool = True
                       ) -> dict:
    """Ответ ``GET /api/v1/jobs/{job_id}`` для жюри: ровно обязательные поля ТЗ, ничего лишнего."""
    oc = resolve_outcome(result)
    return {
        "job_id": oc.job_id,
        "status": oc.status,
        "verdict": oc.verdict,
        "violations": [
            {
                "type": v["type"],
                "timecode_sec": v["timecode_sec"],
                "description": _jury_description(v) if embed_rule_in_description else v["description"],
                "evidence_frames": [{"timestamp_sec": f["timestamp_sec"], "image_base64": f["image_base64"]}
                                    for f in v["evidence_frames"][:MAX_JURY_FRAMES]],
            }
            for v in oc.violations
        ],
    }


def result_to_ui_report(result: AnalysisResult | Mapping[str, Any], *, include_timeline: bool = True,
                        include_meta: bool = True) -> dict:
    """Расширенный отчёт для UI: обязательные поля ТЗ (те же значения, что у жюри) + дополнительные."""
    oc = resolve_outcome(result)
    meta_raw = _get(result, "meta") or {}
    meta: Mapping[str, Any] = meta_raw if isinstance(meta_raw, Mapping) else {}
    out: dict[str, Any] = {
        "job_id": oc.job_id,
        "status": oc.status,
        "verdict": oc.verdict,
        "violations": oc.violations,
        "verdict_state": oc.verdict_state,
        "analysis_complete": oc.analysis_complete,
        "candidates": oc.candidates,
        "warnings": oc.warnings,
    }
    if "progress" in meta:
        p = _finite(meta.get("progress"))
        out["progress"] = None if p is None else _score(p)
    if meta.get("stage") is not None:
        out["stage"] = str(meta["stage"])
    if oc.error:
        out["error"] = oc.error
    if include_timeline:
        timeline = []
        for point in _get(result, "timeline") or []:
            try:
                t, risk = _finite(point[0]), point[1]
            except (TypeError, IndexError, KeyError):
                continue
            if t is not None and t >= 0:
                timeline.append([_round_t(t), _score(risk)])
        out["timeline"] = timeline
    if include_meta:
        out["meta"] = to_jsonable(dict(meta))
    return out


def result_to_api(result: AnalysisResult | Mapping[str, Any], *, profile: str = "jury", **kwargs: Any) -> dict:
    """Имя из контракта v1. ``profile="jury"`` (по умолчанию) — строгая схема ТЗ; ``"ui"`` — UI-report."""
    if profile == "jury":
        return result_to_jury_api(result, **kwargs)
    if profile == "ui":
        return result_to_ui_report(result, **kwargs)
    raise ValueError(f"profile должен быть 'jury' или 'ui', получено {profile!r}")


def minimal_api(data: AnalysisResult | Mapping[str, Any]) -> dict:
    """Проекция на обязательные поля ТЗ: из API-dict (любого профиля) — как есть; из результата — jury."""
    if not isinstance(data, Mapping):
        return result_to_jury_api(data)
    return {
        "job_id": data.get("job_id", ""),
        "status": data.get("status", "failed"),
        "verdict": data.get("verdict"),
        "violations": [
            {
                "type": v.get("type", ""),
                "timecode_sec": v.get("timecode_sec", 0.0),
                "description": v.get("description", ""),
                "evidence_frames": [{"timestamp_sec": f.get("timestamp_sec", 0.0),
                                     "image_base64": f.get("image_base64", "")}
                                    for f in v.get("evidence_frames", []) or []],
            }
            for v in data.get("violations", []) or []
        ],
    }


# ----------------------------------------------------------------------------- from API

def _num(x: Any, default: float) -> float:
    f = _finite(x)
    return default if f is None else f


def _rule_from_api(d: Mapping[str, Any]) -> RuleRef:
    clause = d.get("clause")
    return RuleRef(document=str(d.get("document", "") or ""), clause=None if clause is None else str(clause),
                   url=str(d.get("url", "") or ""), quote=str(d.get("quote", "") or ""),
                   verified=bool(d.get("verified", False)))


def _violation_from_api(d: Mapping[str, Any]) -> Violation:
    rules_raw = d.get("rules")
    if not rules_raw and isinstance(d.get("rule"), Mapping):
        rules_raw = [d["rule"]]
    tc = _num(d.get("timecode_sec"), float("nan"))
    return Violation(
        type=str(d.get("type", "") or ""),
        timecode_sec=tc,
        description=str(d.get("description", "") or ""),
        evidence_frames=[EvidenceFrame(timestamp_sec=_num(f.get("timestamp_sec"), float("nan")),
                                       image_base64=str(f.get("image_base64", "")))
                         for f in (d.get("evidence_frames") or []) if isinstance(f, Mapping)],
        type_id=str(d.get("type_id", "") or ""),
        confidence=_num(d.get("confidence"), 0.0),
        t_start=_num(d.get("t_start"), tc),
        t_end=_num(d.get("t_end"), tc),
        rules=[_rule_from_api(r) for r in (rules_raw or []) if isinstance(r, Mapping)],
    )


def api_to_result(data: Mapping[str, Any]) -> AnalysisResult:
    """API-dict (любого профиля) -> ``AnalysisResult`` (для клиентов, eval и тестов).

    Верхнеуровневые ``progress/stage/error`` возвращаются в ``meta``, если их там нет.
    """
    status = data.get("status", "failed")
    verdict = data.get("verdict")
    meta = dict(data.get("meta") or {})
    for key in ("progress", "stage", "error"):
        if key in data and key not in meta:
            meta[key] = data[key]
    timeline: list[tuple[float, float]] = []
    for point in data.get("timeline") or []:
        try:
            timeline.append((_num(point[0], 0.0), _num(point[1], 0.0)))
        except (TypeError, IndexError, KeyError):
            continue
    return AnalysisResult(
        job_id=str(data.get("job_id", "") or ""),
        status=status if status in _STATUSES else "failed",
        verdict=verdict if verdict in _VERDICTS else None,
        violations=[_violation_from_api(v) for v in (data.get("violations") or []) if isinstance(v, Mapping)],
        timeline=timeline,
        meta=meta,
    )
