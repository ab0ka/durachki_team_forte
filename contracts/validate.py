"""Валидация ответов API по ``contracts/api_schema.json`` (владелец A01).

* ``job_response_errors(data)`` — схема ТЗ, дополнительные поля разрешены (UI-report тоже проходит);
* ``job_response_errors(data, strict=True)`` — строгая схема жюри: только поля ТЗ (additionalProperties=false);
* ``jury_response_errors(data, duration=None)`` — строгая схема + смысловые ОШИБКИ: согласованность
  verdict/violations, конечные таймкоды (в т.ч. не больше длительности), валидный JPEG, не больше 3 кадров;
* ``check_consistency(data)`` — те же смысловые проверки как предупреждения (для UI-report).

CLI::

    python -m contracts.validate resp.json [--jury] [--analyze] [--strict] [--duration 48.3]

Без пакета ``jsonschema`` работает встроенная упрощённая проверка обязательных полей ТЗ.
"""
from __future__ import annotations

import functools
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

from .serialize import API_REQUIRED_TOP, API_REQUIRED_VIOLATION, MAX_JURY_FRAMES, jpeg_b64_problem

__all__ = [
    "SCHEMA_PATH",
    "load_schema",
    "job_response_errors",
    "jury_response_errors",
    "analyze_response_errors",
    "validate_job_response",
    "validate_jury_response",
    "validate_analyze_response",
    "is_valid_job_response",
    "check_consistency",
    "main",
]

SCHEMA_PATH = Path(__file__).with_name("api_schema.json")


@functools.lru_cache(maxsize=2)
def _load(strict: bool) -> str:
    with SCHEMA_PATH.open(encoding="utf-8") as f:
        schema = json.load(f)
    if strict:
        schema["additionalProperties"] = False
        schema["properties"] = {k: schema["properties"][k] for k in schema["required"]}
        for name in ("Violation", "EvidenceFrame"):
            d = schema["$defs"][name]
            d["additionalProperties"] = False
            d["properties"] = {k: d["properties"][k] for k in d["required"]}
        schema["title"] += " (строгая схема жюри)"
    return json.dumps(schema)


def load_schema(strict: bool = False) -> dict:
    """Схема ответа GET /jobs/{id}; ``strict=True`` — только поля ТЗ (для ответа жюри)."""
    return json.loads(_load(strict))


def _subschema(name: str) -> dict:
    root = load_schema()
    return {"$schema": root["$schema"], "$defs": root["$defs"], "$ref": f"#/$defs/{name}"}


def _jsonschema_errors(data: Any, schema: dict) -> list[str]:
    import jsonschema  # лениво

    validator = jsonschema.validators.validator_for(schema)(schema)
    out = []
    for e in sorted(validator.iter_errors(data), key=lambda e: [str(p) for p in e.absolute_path]):
        path = "/".join(str(p) for p in e.absolute_path) or "<root>"
        out.append(f"{path}: {e.message}")
    return out


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _basic_job_errors(data: Any, strict: bool) -> list[str]:
    """Запасная проверка без jsonschema — обязательные поля ТЗ (и лишние поля в strict)."""
    if not isinstance(data, Mapping):
        return ["<root>: ответ должен быть объектом"]
    errs = [f"<root>: нет обязательного поля '{k}'" for k in API_REQUIRED_TOP if k not in data]
    if errs:
        return errs
    if strict:
        errs += [f"<root>: лишнее поле '{k}'" for k in data if k not in API_REQUIRED_TOP]
    if not isinstance(data["job_id"], str) or not data["job_id"]:
        errs.append("job_id: должна быть непустая строка")
    if data["status"] not in ("processing", "done", "failed"):
        errs.append(f"status: недопустимое значение {data['status']!r}")
    if data["verdict"] not in ("violation", "no_violation", None):
        errs.append(f"verdict: недопустимое значение {data['verdict']!r}")
    if data["status"] == "done" and data["verdict"] is None:
        errs.append("verdict: при status=done вердикт обязателен")
    if data["status"] == "processing" and data["verdict"] is not None:
        errs.append("verdict: при status=processing должен быть null")
    if not isinstance(data["violations"], list):
        return errs + ["violations: должен быть массивом"]
    for i, v in enumerate(data["violations"]):
        if not isinstance(v, Mapping):
            errs.append(f"violations/{i}: должен быть объектом")
            continue
        errs += [f"violations/{i}: нет обязательного поля '{k}'" for k in API_REQUIRED_VIOLATION if k not in v]
        if strict:
            errs += [f"violations/{i}: лишнее поле '{k}'" for k in v if k not in API_REQUIRED_VIOLATION]
        if "timecode_sec" in v and (not _is_num(v["timecode_sec"]) or v["timecode_sec"] < 0):
            errs.append(f"violations/{i}/timecode_sec: должно быть числом >= 0")
        for j, fr in enumerate(v.get("evidence_frames") or []):
            if not isinstance(fr, Mapping) or "timestamp_sec" not in fr or "image_base64" not in fr:
                errs.append(f"violations/{i}/evidence_frames/{j}: нужны timestamp_sec и image_base64")
            elif strict:
                errs += [f"violations/{i}/evidence_frames/{j}: лишнее поле '{k}'" for k in fr
                         if k not in ("timestamp_sec", "image_base64")]
    return errs


def job_response_errors(data: Any, strict: bool = False) -> list[str]:
    """Ошибки схемы ответа GET /jobs/{id}; [] — валидно. ``strict`` — только поля ТЗ."""
    try:
        import jsonschema  # noqa: F401
    except ImportError:
        return _basic_job_errors(data, strict)
    return _jsonschema_errors(data, load_schema(strict))


def _semantic_problems(data: Mapping[str, Any], duration: Optional[float], jury: bool) -> list[str]:
    out: list[str] = []
    status, verdict = data.get("status"), data.get("verdict")
    violations = data.get("violations") or []
    if status in ("processing", "failed") and verdict is not None:
        out.append(f"status={status}: вердикт должен быть null (без фиктивного бинарного вердикта)")
    if status in ("processing", "failed") and violations:
        out.append(f"status={status}: violations должен быть пуст")
    if status == "done":
        if verdict == "violation" and not violations:
            out.append("verdict=violation, но список violations пуст")
        if verdict == "no_violation" and violations:
            out.append("verdict=no_violation, но violations не пуст (жюри сочтёт это ложным срабатыванием)")
    for i, v in enumerate(violations):
        if not isinstance(v, Mapping):
            continue
        tc = v.get("timecode_sec")
        if not _is_num(tc) or not math.isfinite(tc) or tc < 0:
            out.append(f"violations/{i}: timecode_sec не конечное неотрицательное число ({tc!r})")
        elif duration is not None and tc > duration + 0.05:
            out.append(f"violations/{i}: timecode_sec={tc} больше длительности видео {duration}")
        frames = v.get("evidence_frames") or []
        if jury and len(frames) > MAX_JURY_FRAMES:
            out.append(f"violations/{i}: больше {MAX_JURY_FRAMES} кадров-доказательств ({len(frames)})")
        for j, fr in enumerate(frames):
            if not isinstance(fr, Mapping):
                continue
            ts = fr.get("timestamp_sec")
            if not _is_num(ts) or not math.isfinite(ts) or ts < 0:
                out.append(f"violations/{i}/evidence_frames/{j}: timestamp_sec не конечное число ({ts!r})")
            problem = jpeg_b64_problem(fr.get("image_base64"))
            if problem:
                out.append(f"violations/{i}/evidence_frames/{j}: {problem}")
    return out


def jury_response_errors(data: Any, duration: Optional[float] = None) -> list[str]:
    """Строгая проверка ответа для жюри: схема ТЗ без лишних полей + смысловые ошибки."""
    errs = job_response_errors(data, strict=True)
    if isinstance(data, Mapping):
        errs += _semantic_problems(data, duration, jury=True)
    return errs


def analyze_response_errors(data: Any) -> list[str]:
    """Ошибки схемы ответа POST /analyze ({"job_id": "..."})."""
    try:
        import jsonschema  # noqa: F401
    except ImportError:
        if isinstance(data, Mapping) and isinstance(data.get("job_id"), str) and data["job_id"]:
            return []
        return ["job_id: обязательна непустая строка"]
    return _jsonschema_errors(data, _subschema("AnalyzeResponse"))


def validate_job_response(data: Any, strict: bool = False) -> None:
    errs = job_response_errors(data, strict=strict)
    if errs:
        raise ValueError("Ответ не соответствует схеме ТЗ:\n  " + "\n  ".join(errs))


def validate_jury_response(data: Any, duration: Optional[float] = None) -> None:
    errs = jury_response_errors(data, duration)
    if errs:
        raise ValueError("Ответ для жюри некорректен:\n  " + "\n  ".join(errs))


def validate_analyze_response(data: Any) -> None:
    errs = analyze_response_errors(data)
    if errs:
        raise ValueError("Ответ /analyze не соответствует схеме:\n  " + "\n  ".join(errs))


def is_valid_job_response(data: Any, strict: bool = False) -> bool:
    return not job_response_errors(data, strict=strict)


def check_consistency(data: Mapping[str, Any], duration: Optional[float] = None) -> list[str]:
    """Смысловые предупреждения (UI-report): то, на чём можно потерять баллы жюри."""
    warns = _semantic_problems(data, duration, jury=False)
    for i, v in enumerate(data.get("violations") or []):
        if not isinstance(v, Mapping):
            continue
        n = len(v.get("evidence_frames") or [])
        if not 1 <= n <= MAX_JURY_FRAMES:
            warns.append(f"violations/{i}: ожидается 1–{MAX_JURY_FRAMES} кадра-доказательства, получено {n}")
        if "type_id" in v and not v.get("type_id"):
            warns.append(f"violations/{i}: не заполнен type_id")
        if "rules" in v and not v.get("rules"):
            warns.append(f"violations/{i}: нет ссылок на нормы (rules)")
    if data.get("status") == "failed" and not data.get("error") and not (data.get("meta") or {}).get("error"):
        warns.append("status=failed без поля error/meta.error (только UI-report)")
    return warns


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Проверка JSON-ответа Kassa Sentinel по схеме ТЗ")
    ap.add_argument("files", nargs="+", help="JSON-файлы с ответами GET /api/v1/jobs/{id}")
    ap.add_argument("--jury", action="store_true", help="строгая проверка ответа для жюри (только поля ТЗ)")
    ap.add_argument("--analyze", action="store_true", help="проверять как ответ POST /analyze")
    ap.add_argument("--strict", action="store_true", help="предупреждения тоже считать ошибкой")
    ap.add_argument("--duration", type=float, default=None, help="длительность видео, с")
    args = ap.parse_args(argv)
    rc = 0
    for name in args.files:
        with open(name, encoding="utf-8") as f:
            data = json.load(f)
        if args.analyze:
            errs, warns = analyze_response_errors(data), []
        elif args.jury:
            errs, warns = jury_response_errors(data, args.duration), []
        else:
            errs, warns = job_response_errors(data), check_consistency(data, args.duration)
        state = "OK" if not errs and not (args.strict and warns) else "FAIL"
        print(f"[{state}] {name}")
        for e in errs:
            print(f"   ERROR {e}")
        for w in warns:
            print(f"   WARN  {w}")
        if state == "FAIL":
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
