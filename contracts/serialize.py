"""Сериализация ``AnalysisResult`` -> JSON для API (владелец A01). Два сериализатора:

1. ``result_to_jury_api(result)`` — ответ ``GET /api/v1/jobs/{job_id}`` для жюри: РОВНО обязательная схема ТЗ,
   без предположения, что дополнительные поля будут приняты::

       {"job_id", "status", "verdict",
        "violations": [{"type", "timecode_sec", "description",
                        "evidence_frames": [{"timestamp_sec", "image_base64"}]}]}

2. ``result_to_ui_report(result)`` — расширенный отчёт для UI/отладки: всё из (1) + ``type_id, confidence,
   t_start, t_end, rules[], rule{}`` у нарушения и ``progress, stage, error, verdict_state, analysis_complete,
   coverage_problem, warnings, candidates, timeline, meta`` на верхнем уровне.

``result_to_api(result)`` — прежнее имя и поведение из контракта («схема ТЗ + доп. поля») = UI-report; на него
опираются A02/A03 (кэш)/A16 (eval). Ответ жюри — ``result_to_jury_api`` (или ``profile="jury"``).

Политика вердикта (одна для обоих сериализаторов, ``resolve_outcome``):

* ``processing`` и ``failed`` — ``verdict = null``, ``violations = []`` (никаких фиктивных бинарных вердиктов);
* mock-результат (``meta.mock`` / ``provenance.mock`` / стадия ``mock``) без явного ``SENTINEL_MOCK=1`` -> ``failed``:
  production сам на mock не переключается;
* ``done`` + ``violation`` — только если есть хотя бы одно нарушение с конечным таймкодом, иначе ``failed``;
* ``done`` + ``no_violation`` — только при подтверждённой полноте ``meta.coverage`` (``contracts.extensions``),
  иначе ``failed`` с причиной: неполный/упавший анализ ≠ «нарушений нет». Нарушения при ``no_violation`` жюри
  не отдаются (в UI — ``candidates``);
* ``done`` без вердикта — ``violation``, если есть валидные нарушения, иначе ``failed`` (``no_violation`` не выдумываем);
* таймкоды конечные, >= 0, не больше длительности, округлены до 0.1 с; если ``timecode_sec`` не конечен — берётся
  ``t_start`` (начало действия), затем первый кадр-доказательство; иначе нарушение отбрасывается с предупреждением;
* кадры: только валидный base64 JPEG (FF D8 FF ... FF D9), без префикса ``data:``; жюри — не больше ``jury.max_frames``.

Флаги — ``contracts/config.yaml`` (гипотезы по умолчанию выключены). Без побочных эффектов при импорте.
"""
from __future__ import annotations

import base64
import binascii
import dataclasses
import functools
import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .extensions import (
    META_COVERAGE,
    META_WARNINGS,
    TYPE_REGISTRY,
    coverage_problem,
    is_mock_result,
    mock_allowed,
    video_duration,
    violation_extra,
)
from .sentinel_contracts import AnalysisResult, EvidenceFrame, RuleRef, Violation

__all__ = [
    "API_REQUIRED_TOP",
    "API_REQUIRED_VIOLATION",
    "API_REQUIRED_FRAME",
    "DEFAULT_TYPE_NAMES",
    "MAX_JURY_FRAMES",
    "TIME_DECIMALS",
    "Outcome",
    "load_config",
    "resolve_outcome",
    "result_to_api",
    "result_to_jury_api",
    "result_to_ui_report",
    "api_to_result",
    "minimal_api",
    "rule_to_api",
    "jpeg_b64_problem",
    "to_jsonable",
    "canonical_json",
    "fingerprint",
]

API_REQUIRED_TOP = ("job_id", "status", "verdict", "violations")
API_REQUIRED_VIOLATION = ("type", "timecode_sec", "description", "evidence_frames")
API_REQUIRED_FRAME = ("timestamp_sec", "image_base64")

TIME_DECIMALS = 1      # таймкоды — до 0.1 с
SCORE_DECIMALS = 3     # confidence / risk / progress
MAX_JURY_FRAMES = 3    # ТЗ: 1–3 кадра-доказательства
PTS_TOLERANCE_S = 0.05  # кадр-доказательство не может быть позже конца видео (PTS)
EVIDENCE_WINDOW_S = 3.0  # кадр дальше этого от окна события — предупреждение в UI-report (жюри не меняется)

_STATUSES = ("processing", "done", "failed")
_VERDICTS = ("violation", "no_violation")
_DATA_URI_MARK = ";base64,"

# Запасные русские названия — ТОЛЬКО если fusion не заполнил ``Violation.type``. Нормы здесь не хранятся.
DEFAULT_TYPE_NAMES: dict[str, str] = dict(TYPE_REGISTRY)

_DEFAULT_CONFIG: dict[str, Any] = {
    "jury": {"max_frames": MAX_JURY_FRAMES, "embed_rule_in_description": False},
    "coverage": {"min_ratio_for_no_violation": 0.98},
}


# ----------------------------------------------------------------------------- config

@functools.lru_cache(maxsize=4)
def _load_config_cached(path: str) -> str:
    cfg = json.loads(json.dumps(_DEFAULT_CONFIG))
    p = Path(path)
    if p.is_file():
        try:
            import yaml  # лениво

            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001 — нет PyYAML / битый файл: значения по умолчанию
            data = {}
        for section, values in data.items() if isinstance(data, dict) else []:
            if isinstance(values, dict) and section in cfg:
                cfg[section].update(values)
    return json.dumps(cfg)


def load_config() -> dict[str, Any]:
    """``contracts/config.yaml`` (или ``SENTINEL_CONTRACTS_CONFIG``) поверх значений по умолчанию."""
    path = os.environ.get("SENTINEL_CONTRACTS_CONFIG") or str(Path(__file__).with_name("config.yaml"))
    return json.loads(_load_config_cached(path))


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


def canonical_json(data: Any) -> str:
    """Воспроизводимый JSON: одинаковый вход -> одинаковые байты (ключи отсортированы, без NaN, UTF-8)."""
    return json.dumps(to_jsonable(data), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(data: Any) -> str:
    """sha256 канонического JSON — для сравнения прогонов (например, ответов жюри до/после изменения)."""
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


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
    coverage_problem: Optional[str] = None
    analysis_complete: bool = False


def _normalize_violation(v: Any, index: int, duration: Optional[float], meta: Mapping[str, Any],
                         warnings: list[str]) -> Optional[dict[str, Any]]:
    type_id = str(_get(v, "type_id", "") or "").strip()
    vtype = str(_get(v, "type", "") or "").strip() or DEFAULT_TYPE_NAMES.get(type_id, "") or "Возможное нарушение"

    frames: list[dict[str, Any]] = []
    for j, fr in enumerate(list(_get(v, "evidence_frames") or [])):
        ts = _finite(_get(fr, "timestamp_sec"))
        b64 = _clean_b64(_get(fr, "image_base64"))
        problem = jpeg_b64_problem(b64)
        if ts is None or ts < 0:
            warnings.append(f"нарушение {index}, кадр {j}: нет конечного timestamp_sec — кадр отброшен")
            continue
        if duration is not None and ts > duration + PTS_TOLERANCE_S:
            warnings.append(f"нарушение {index}, кадр {j}: timestamp_sec={ts:.2f} за концом видео "
                            f"({duration:.2f} с) — неверный PTS, кадр отброшен")
            continue
        if problem:
            warnings.append(f"нарушение {index}, кадр {j}: {problem} — кадр отброшен")
            continue
        frames.append({"timestamp_sec": ts, "image_base64": b64})

    tc = _finite(_get(v, "timecode_sec"))
    if tc is None or tc < 0:
        fallback = _finite(_get(v, "t_start"))  # начало действия — та же политика, что у таймкода
        src = "t_start"
        if fallback is None or fallback < 0:
            fallback = next((f["timestamp_sec"] for f in frames), None)
            src = "первый кадр"
        if fallback is None or fallback < 0:
            warnings.append(f"нарушение {index} ({type_id or vtype}): нет конечного таймкода — не отдаётся")
            return None
        warnings.append(f"нарушение {index}: timecode_sec не задан, взят {fallback:.1f} с ({src})")
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

    for f in frames:
        if not t_start - EVIDENCE_WINDOW_S <= f["timestamp_sec"] <= t_end + EVIDENCE_WINDOW_S:
            warnings.append(f"нарушение {index}: кадр {f['timestamp_sec']:.1f} с вне окна события "
                            f"[{t_start:.1f}; {t_end:.1f}] ± {EVIDENCE_WINDOW_S:g} с — проверьте PTS доказательства")
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


def resolve_outcome(result: AnalysisResult | Mapping[str, Any], *, allow_mock: Optional[bool] = None) -> Outcome:
    """Единая политика статуса/вердикта (см. docstring модуля). ``allow_mock=None`` — по ``SENTINEL_MOCK``."""
    cfg = load_config()
    allow = mock_allowed() if allow_mock is None else bool(allow_mock)
    job_id = str(_get(result, "job_id", "") or "")
    status = str(_get(result, "status", "failed") or "failed")
    meta_raw = _get(result, "meta") or {}
    meta: Mapping[str, Any] = meta_raw if isinstance(meta_raw, Mapping) else {}
    raw_w = meta.get(META_WARNINGS)
    warnings: list[str] = [str(w) for w in raw_w if w is not None] if isinstance(raw_w, list) else []
    cov_problem = coverage_problem(meta, float(cfg["coverage"]["min_ratio_for_no_violation"]), allow)
    complete = cov_problem is None

    def failed(reason: str, candidates: Optional[list] = None) -> Outcome:
        warnings.append(reason)
        return Outcome(job_id, "failed", None, "unknown", candidates=candidates or [], error=reason,
                       warnings=warnings, coverage_problem=cov_problem, analysis_complete=complete)

    if status not in _STATUSES:
        return failed(f"недопустимый статус {status!r}")
    if status == "processing":
        return Outcome(job_id, "processing", None, "pending", warnings=warnings, coverage_problem=cov_problem,
                       analysis_complete=complete)
    if status == "failed":
        err = meta.get("error")
        out = failed(str(err) if err else "Анализ завершился с ошибкой")
        return out
    if is_mock_result(meta) and not allow:
        return failed("результат получен на mock-бэкенде, а mock-режим явно не включён (SENTINEL_MOCK=1) — "
                      "production не переключается на mock")

    duration = video_duration(meta)
    valid = []
    for i, v in enumerate(_get(result, "violations") or []):
        nv = _normalize_violation(v, i, duration, meta, warnings)
        if nv is not None:
            valid.append(nv)
    verdict = _get(result, "verdict")

    if verdict == "violation":
        if not valid:
            return failed("verdict=violation, но нет ни одного нарушения с конечным таймкодом")
        return Outcome(job_id, "done", "violation", "violation", violations=valid, warnings=warnings,
                       coverage_problem=cov_problem, analysis_complete=complete)
    if verdict == "no_violation":
        if cov_problem:
            return failed(f"«нет нарушения» не подтверждено: {cov_problem}", candidates=valid)
        if valid:
            warnings.append("verdict=no_violation: найденные кандидаты не отдаются жюри (см. candidates)")
        return Outcome(job_id, "done", "no_violation", "no_violation", candidates=valid, warnings=warnings,
                       coverage_problem=None, analysis_complete=True)
    if verdict is not None:
        warnings.append(f"недопустимый verdict={verdict!r}")
    if valid:
        warnings.append("verdict не задан — выведен 'violation' по списку нарушений")
        return Outcome(job_id, "done", "violation", "violation", violations=valid, warnings=warnings,
                       coverage_problem=cov_problem, analysis_complete=complete)
    return failed("анализ не выдал вердикт — «нет нарушения» не выдумываем")


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


def result_to_jury_api(result: AnalysisResult | Mapping[str, Any], *,
                       embed_rule_in_description: Optional[bool] = None, allow_mock: Optional[bool] = None) -> dict:
    """Ответ ``GET /api/v1/jobs/{job_id}`` для жюри: ровно обязательные поля ТЗ, ничего лишнего.

    ``embed_rule_in_description=None`` — по флагу ``jury.embed_rule_in_description`` (по умолчанию выкл.).
    """
    cfg = load_config()
    embed = bool(cfg["jury"]["embed_rule_in_description"]) if embed_rule_in_description is None \
        else bool(embed_rule_in_description)
    max_frames = int(cfg["jury"]["max_frames"])
    oc = resolve_outcome(result, allow_mock=allow_mock)
    return {
        "job_id": oc.job_id,
        "status": oc.status,
        "verdict": oc.verdict,
        "violations": [
            {
                "type": v["type"],
                "timecode_sec": v["timecode_sec"],
                "description": _jury_description(v) if embed else v["description"],
                "evidence_frames": [{"timestamp_sec": f["timestamp_sec"], "image_base64": f["image_base64"]}
                                    for f in v["evidence_frames"][:max_frames]],
            }
            for v in oc.violations
        ],
    }


def result_to_ui_report(result: AnalysisResult | Mapping[str, Any], *, include_timeline: bool = True,
                        include_meta: bool = True, allow_mock: Optional[bool] = None) -> dict:
    """Расширенный отчёт для UI: обязательные поля ТЗ (те же значения, что у жюри) + дополнительные."""
    oc = resolve_outcome(result, allow_mock=allow_mock)
    meta_raw = _get(result, "meta") or {}
    meta: Mapping[str, Any] = meta_raw if isinstance(meta_raw, Mapping) else {}
    out: dict[str, Any] = {
        "job_id": oc.job_id,
        "status": oc.status,
        "verdict": oc.verdict,
        "violations": oc.violations,
        "verdict_state": oc.verdict_state,
        "analysis_complete": oc.analysis_complete,
        "coverage_problem": oc.coverage_problem,
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
    if isinstance(meta.get(META_COVERAGE), (Mapping, int, float)):
        out["coverage"] = to_jsonable(meta[META_COVERAGE])
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


def result_to_api(result: AnalysisResult | Mapping[str, Any], *, profile: str = "ui", **kwargs: Any) -> dict:
    """Имя из контракта v1 («схема ТЗ + доп. поля») = UI-report; ``profile="jury"`` — строгий ответ жюри."""
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

    Верхнеуровневые ``progress/stage/error/coverage`` возвращаются в ``meta``, если их там нет.
    """
    status = data.get("status", "failed")
    verdict = data.get("verdict")
    meta = dict(data.get("meta") or {})
    for key in ("progress", "stage", "error", META_COVERAGE):
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
