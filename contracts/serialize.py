"""Сериализация ``AnalysisResult`` <-> JSON-ответ ``GET /api/v1/jobs/{job_id}`` (владелец A01).

Обязательные поля — строго как в ТЗ::

    {"job_id", "status", "verdict",
     "violations": [{"type", "timecode_sec", "description",
                     "evidence_frames": [{"timestamp_sec", "image_base64"}]}]}

Дополнительно (разрешено ТЗ): ``type_id, confidence, t_start, t_end, rules[], rule{}`` у нарушения;
``timeline, meta, progress, stage, error`` на верхнем уровне.

Гарантии ``result_to_api``:
* ключ ``verdict`` есть всегда; пока ``status == "processing"`` он ``null``;
  при ``done`` без явного вердикта он выводится из наличия нарушений;
* таймкоды (``timecode_sec, timestamp_sec, t_start, t_end``, t в ``timeline``) округлены до 0.1 с;
* в ответе нет NaN/Inf (заменяются на 0.0 для обязательных чисел и на ``null`` в остальных местах);
* ``image_base64`` — чистый base64 JPEG без префикса ``data:image/...;base64,``;
* ``rules`` отдаётся и списком ``rules: [...]``, и первой нормой ``rule: {...}`` (или ``null``) — удобно UI;
* результат проходит ``json.dumps(..., allow_nan=False)``.

Модуль без тяжёлых зависимостей и без побочных эффектов при импорте.
"""
from __future__ import annotations

import base64
import dataclasses
import math
from collections.abc import Mapping
from typing import Any

from .sentinel_contracts import AnalysisResult, EvidenceFrame, RuleRef, Violation

__all__ = [
    "API_REQUIRED_TOP",
    "API_REQUIRED_VIOLATION",
    "API_REQUIRED_FRAME",
    "DEFAULT_TYPE_NAMES",
    "TIME_DECIMALS",
    "result_to_api",
    "api_to_result",
    "violation_to_api",
    "rule_to_api",
    "to_jsonable",
    "minimal_api",
]

API_REQUIRED_TOP = ("job_id", "status", "verdict", "violations")
API_REQUIRED_VIOLATION = ("type", "timecode_sec", "description", "evidence_frames")
API_REQUIRED_FRAME = ("timestamp_sec", "image_base64")

TIME_DECIMALS = 1      # таймкоды — до 0.1 с
SCORE_DECIMALS = 3     # confidence / risk

_STATUSES = ("processing", "done", "failed")
_VERDICTS = ("violation", "no_violation")

# Запасные русские названия — ТОЛЬКО если fusion не заполнил ``Violation.type``.
# Источник истины — norms/catalog.yaml (A13), здесь нет никаких ссылок на нормы.
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
    """float(x), если это конечное число, иначе None (NaN/Inf/мусор)."""
    if isinstance(x, bool):
        return float(x)
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _num(x: Any, default: float = 0.0, ndigits: int | None = None) -> float:
    """Обязательное число: конечный float (NaN/Inf/None -> default), опционально округлённый."""
    f = _finite(x)
    if f is None:
        f = float(default)
    if ndigits is not None:
        f = round(f, ndigits)
    return f + 0.0  # -0.0 -> 0.0


def _time(x: Any, default: float = 0.0) -> float:
    return max(0.0, _num(x, default, TIME_DECIMALS))


def _score(x: Any, default: float = 0.0) -> float:
    return min(1.0, max(0.0, _num(x, default, SCORE_DECIMALS)))


def _clean_b64(data: Any) -> str:
    """Чистый base64 JPEG: bytes кодируем, у data URI отрезаем префикс, пробелы убираем."""
    if data is None:
        return ""
    if isinstance(data, (bytes, bytearray, memoryview)):
        raw = bytes(data)
        # Похоже на уже закодированный base64 (ASCII) — оставляем, иначе кодируем сырые байты.
        if raw[:3] == b"\xff\xd8\xff" or not raw.isascii():
            return base64.b64encode(raw).decode("ascii")
        data = raw.decode("ascii")
    s = str(data).strip()
    if s.startswith("data:") and _DATA_URI_MARK in s[:64]:
        s = s.split(_DATA_URI_MARK, 1)[1]
    if any(c.isspace() for c in s):
        s = "".join(s.split())
    return s


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Атрибут dataclass-а или ключ словаря — сериализатор принимает и то и другое."""
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def to_jsonable(obj: Any, _depth: int = 0) -> Any:
    """Рекурсивно приводит объект к JSON-совместимому виду (без numpy/torch импортов).

    NaN/Inf -> None; numpy-скаляры/массивы -> python; dataclass -> dict; set/tuple -> list;
    Path и прочее -> str. Глубина ограничена, чтобы не зациклиться на ссылках.
    """
    if _depth > 32:
        return str(obj)
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, int):
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
    # numpy / torch скаляры и массивы — утиная типизация, без импорта библиотек
    if hasattr(obj, "tolist") and callable(obj.tolist):
        try:
            return to_jsonable(obj.tolist(), _depth + 1)
        except Exception:  # noqa: BLE001
            pass
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return to_jsonable(obj.item(), _depth + 1)
        except Exception:  # noqa: BLE001
            pass
    return str(obj)


# ----------------------------------------------------------------------------- to API

def rule_to_api(rule: RuleRef | Mapping[str, Any]) -> dict[str, Any]:
    clause = _get(rule, "clause")
    return {
        "document": str(_get(rule, "document", "") or ""),
        "clause": None if clause is None else str(clause),
        "url": str(_get(rule, "url", "") or ""),
        "quote": str(_get(rule, "quote", "") or ""),
        "verified": bool(_get(rule, "verified", False)),
    }


def _frame_to_api(fr: EvidenceFrame | Mapping[str, Any]) -> dict[str, Any]:
    return {
        "timestamp_sec": _time(_get(fr, "timestamp_sec")),
        "image_base64": _clean_b64(_get(fr, "image_base64")),
    }


def violation_to_api(v: Violation | Mapping[str, Any]) -> dict[str, Any]:
    """Одно нарушение: сначала обязательные поля ТЗ, затем дополнительные."""
    type_id = str(_get(v, "type_id", "") or "")
    vtype = str(_get(v, "type", "") or "").strip() or DEFAULT_TYPE_NAMES.get(type_id, type_id or "Возможное нарушение")
    tc = _time(_get(v, "timecode_sec"))
    description = str(_get(v, "description", "") or "").strip()
    if not description:
        description = f"Признаки возможного нарушения ({vtype}) на {tc:.1f} с. Требуется проверка человеком."
    frames = [_frame_to_api(fr) for fr in (_get(v, "evidence_frames") or [])]
    t_start_raw, t_end_raw = _get(v, "t_start", 0.0), _get(v, "t_end", 0.0)
    t_start, t_end = _time(t_start_raw), _time(t_end_raw)
    if t_end < t_start:
        t_start, t_end = t_end, t_start
    rules = [rule_to_api(r) for r in (_get(v, "rules") or [])]
    return {
        # --- обязательные (ТЗ) ---
        "type": vtype,
        "timecode_sec": tc,
        "description": description,
        "evidence_frames": frames,
        # --- дополнительные ---
        "type_id": type_id,
        "confidence": _score(_get(v, "confidence", 0.0)),
        "t_start": t_start,
        "t_end": t_end,
        "rules": rules,
        "rule": rules[0] if rules else None,
    }


def _resolve_verdict(status: str, verdict: Any, n_violations: int) -> str | None:
    if status == "processing":
        return None
    if verdict in _VERDICTS:
        return str(verdict)
    if status == "done":  # вердикт не заполнен — выводим из списка нарушений
        return "violation" if n_violations > 0 else "no_violation"
    return None  # failed без вердикта


def result_to_api(result: AnalysisResult, *, include_timeline: bool = True, include_meta: bool = True) -> dict:
    """``AnalysisResult`` -> dict ровно по схеме ТЗ + дополнительные поля.

    Пока ``status == "processing"``: ``verdict = null`` и ``violations = []``.
    ``meta["progress"]``, ``meta["stage"]``, ``meta["error"]`` дублируются на верхний уровень.
    """
    status = str(_get(result, "status", "failed") or "failed")
    if status not in _STATUSES:
        status = "failed"
    raw_violations = list(_get(result, "violations") or [])
    violations = [] if status == "processing" else [violation_to_api(v) for v in raw_violations]
    out: dict[str, Any] = {
        "job_id": str(_get(result, "job_id", "") or ""),
        "status": status,
        "verdict": _resolve_verdict(status, _get(result, "verdict"), len(violations)),
        "violations": violations,
    }
    meta = _get(result, "meta") or {}
    if isinstance(meta, Mapping):
        if "progress" in meta:
            p = _finite(meta.get("progress"))
            out["progress"] = None if p is None else round(min(1.0, max(0.0, p)), SCORE_DECIMALS)
        for key in ("stage", "error"):
            if meta.get(key) is not None:
                out[key] = str(meta[key])
    if include_timeline:
        timeline = []
        for point in _get(result, "timeline") or []:
            try:
                t, risk = point[0], point[1]
            except (TypeError, IndexError, KeyError):
                continue
            if _finite(t) is None:
                continue
            timeline.append([_time(t), _score(risk)])
        out["timeline"] = timeline
    if include_meta:
        out["meta"] = to_jsonable(dict(meta)) if isinstance(meta, Mapping) else {}
    return out


def minimal_api(result: AnalysisResult | Mapping[str, Any]) -> dict:
    """Только обязательные поля ТЗ (для строгих клиентов/сверки). Принимает результат или API-dict."""
    full = dict(result) if isinstance(result, Mapping) else result_to_api(result, include_timeline=False,
                                                                           include_meta=False)
    return {
        "job_id": full.get("job_id", ""),
        "status": full.get("status", "failed"),
        "verdict": full.get("verdict"),
        "violations": [
            {
                "type": v.get("type", ""),
                "timecode_sec": v.get("timecode_sec", 0.0),
                "description": v.get("description", ""),
                "evidence_frames": [
                    {"timestamp_sec": f.get("timestamp_sec", 0.0), "image_base64": f.get("image_base64", "")}
                    for f in v.get("evidence_frames", []) or []
                ],
            }
            for v in full.get("violations", []) or []
        ],
    }


# ----------------------------------------------------------------------------- from API

def _rule_from_api(d: Mapping[str, Any]) -> RuleRef:
    clause = d.get("clause")
    return RuleRef(
        document=str(d.get("document", "") or ""),
        clause=None if clause is None else str(clause),
        url=str(d.get("url", "") or ""),
        quote=str(d.get("quote", "") or ""),
        verified=bool(d.get("verified", False)),
    )


def _violation_from_api(d: Mapping[str, Any]) -> Violation:
    rules_raw = d.get("rules")
    if not rules_raw and isinstance(d.get("rule"), Mapping):
        rules_raw = [d["rule"]]
    tc = _num(d.get("timecode_sec"), 0.0)
    return Violation(
        type=str(d.get("type", "") or ""),
        timecode_sec=tc,
        description=str(d.get("description", "") or ""),
        evidence_frames=[
            EvidenceFrame(timestamp_sec=_num(f.get("timestamp_sec"), 0.0), image_base64=str(f.get("image_base64", "")))
            for f in (d.get("evidence_frames") or [])
            if isinstance(f, Mapping)
        ],
        type_id=str(d.get("type_id", "") or ""),
        confidence=_num(d.get("confidence"), 0.0),
        t_start=_num(d.get("t_start"), tc if "t_start" not in d else 0.0),
        t_end=_num(d.get("t_end"), tc if "t_end" not in d else 0.0),
        rules=[_rule_from_api(r) for r in (rules_raw or []) if isinstance(r, Mapping)],
    )


def api_to_result(data: Mapping[str, Any]) -> AnalysisResult:
    """Обратное преобразование API-dict -> ``AnalysisResult`` (для клиентов, eval и тестов).

    Верхнеуровневые ``progress/stage/error`` возвращаются в ``meta`` (если их там ещё нет).
    """
    status = data.get("status", "failed")
    if status not in _STATUSES:
        status = "failed"
    verdict = data.get("verdict")
    meta = dict(data.get("meta") or {})
    for key in ("progress", "stage", "error"):
        if key in data and key not in meta:
            meta[key] = data[key]
    timeline: list[tuple[float, float]] = []
    for point in data.get("timeline") or []:
        try:
            timeline.append((_num(point[0]), _num(point[1])))
        except (TypeError, IndexError, KeyError):
            continue
    return AnalysisResult(
        job_id=str(data.get("job_id", "") or ""),
        status=status,
        verdict=verdict if verdict in _VERDICTS else None,
        violations=[_violation_from_api(v) for v in (data.get("violations") or []) if isinstance(v, Mapping)],
        timeline=timeline,
        meta=meta,
    )
