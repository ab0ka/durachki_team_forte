"""Валидация ответов API по ``contracts/api_schema.json`` (владелец A01).

Использование::

    from contracts.validate import validate_job_response, job_response_errors, check_consistency
    validate_job_response(resp)            # бросает ValueError со списком ошибок
    errs = job_response_errors(resp)       # [] если валидно
    warns = check_consistency(resp)        # смысловые предупреждения (вердикт vs нарушения, кадры, base64)

CLI::

    python -m contracts.validate response.json [--analyze] [--strict]

Если пакет ``jsonschema`` не установлен — работает встроенная упрощённая проверка обязательных полей ТЗ.
"""
from __future__ import annotations

import base64
import binascii
import functools
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = [
    "SCHEMA_PATH",
    "load_schema",
    "job_response_errors",
    "analyze_response_errors",
    "validate_job_response",
    "validate_analyze_response",
    "is_valid_job_response",
    "check_consistency",
    "main",
]

SCHEMA_PATH = Path(__file__).with_name("api_schema.json")
_JPEG_B64_PREFIX = "/9j/"  # base64 от байтов FF D8 FF


@functools.lru_cache(maxsize=1)
def load_schema() -> dict:
    with SCHEMA_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def _subschema(name: str) -> dict:
    root = load_schema()
    return {"$schema": root["$schema"], "$defs": root["$defs"], "$ref": f"#/$defs/{name}"}


def _jsonschema_errors(data: Any, schema: dict) -> list[str]:
    import jsonschema  # лениво

    validator_cls = jsonschema.validators.validator_for(schema)
    validator = validator_cls(schema)
    errors = sorted(validator.iter_errors(data), key=lambda e: list(e.absolute_path))
    out = []
    for e in errors:
        path = "/".join(str(p) for p in e.absolute_path) or "<root>"
        out.append(f"{path}: {e.message}")
    return out


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _basic_job_errors(data: Any) -> list[str]:
    """Запасная проверка без jsonschema — только обязательные поля ТЗ."""
    if not isinstance(data, Mapping):
        return ["<root>: ответ должен быть объектом"]
    errs = [f"<root>: нет обязательного поля '{k}'" for k in ("job_id", "status", "verdict", "violations")
            if k not in data]
    if errs:
        return errs
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
        for k in ("type", "timecode_sec", "description", "evidence_frames"):
            if k not in v:
                errs.append(f"violations/{i}: нет обязательного поля '{k}'")
        if "timecode_sec" in v and (not _is_num(v["timecode_sec"]) or v["timecode_sec"] < 0):
            errs.append(f"violations/{i}/timecode_sec: должно быть числом >= 0")
        for j, fr in enumerate(v.get("evidence_frames") or []):
            if not isinstance(fr, Mapping) or "timestamp_sec" not in fr or "image_base64" not in fr:
                errs.append(f"violations/{i}/evidence_frames/{j}: нужны timestamp_sec и image_base64")
    return errs


def job_response_errors(data: Any) -> list[str]:
    """Список ошибок схемы ответа GET /jobs/{id}; пустой список — валидно."""
    try:
        import jsonschema  # noqa: F401
    except ImportError:
        return _basic_job_errors(data)
    return _jsonschema_errors(data, load_schema())


def analyze_response_errors(data: Any) -> list[str]:
    """Список ошибок схемы ответа POST /analyze ({"job_id": "..."})."""
    try:
        import jsonschema  # noqa: F401
    except ImportError:
        if isinstance(data, Mapping) and isinstance(data.get("job_id"), str) and data["job_id"]:
            return []
        return ["job_id: обязательна непустая строка"]
    return _jsonschema_errors(data, _subschema("AnalyzeResponse"))


def validate_job_response(data: Any) -> None:
    errs = job_response_errors(data)
    if errs:
        raise ValueError("Ответ не соответствует схеме ТЗ:\n  " + "\n  ".join(errs))


def validate_analyze_response(data: Any) -> None:
    errs = analyze_response_errors(data)
    if errs:
        raise ValueError("Ответ /analyze не соответствует схеме:\n  " + "\n  ".join(errs))


def is_valid_job_response(data: Any) -> bool:
    return not job_response_errors(data)


def _b64_problem(s: Any) -> str | None:
    if not isinstance(s, str) or not s:
        return "пустой image_base64"
    if s.startswith("data:"):
        return "image_base64 содержит префикс data URI (нужен чистый base64)"
    try:
        head = base64.b64decode(s[:64] + "=" * (-len(s[:64]) % 4), validate=False)
        base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError):
        return "image_base64 не декодируется как base64"
    if not head.startswith(b"\xff\xd8\xff"):
        return "image_base64 — не JPEG (нет сигнатуры FF D8 FF)"
    return None


def check_consistency(data: Mapping[str, Any], duration: float | None = None) -> list[str]:
    """Смысловые предупреждения (не ошибки схемы): то, на чём можно потерять баллы жюри."""
    warns: list[str] = []
    status, verdict = data.get("status"), data.get("verdict")
    violations = data.get("violations") or []
    if status == "done":
        if verdict == "violation" and not violations:
            warns.append("verdict=violation, но список violations пуст")
        if verdict == "no_violation" and violations:
            warns.append("verdict=no_violation, но violations не пуст (жюри может счесть это ложным срабатыванием)")
    if status == "failed" and not data.get("error") and not (data.get("meta") or {}).get("error"):
        warns.append("status=failed без поля error/meta.error")
    for i, v in enumerate(violations):
        if not isinstance(v, Mapping):
            continue
        frames = v.get("evidence_frames") or []
        if not 1 <= len(frames) <= 3:
            warns.append(f"violations/{i}: ожидается 1–3 кадра-доказательства, получено {len(frames)}")
        tc = v.get("timecode_sec")
        if duration is not None and _is_num(tc) and tc > duration + 0.5:
            warns.append(f"violations/{i}: timecode_sec={tc} больше длительности видео {duration}")
        if not v.get("type_id"):
            warns.append(f"violations/{i}: не заполнен type_id")
        if not v.get("rules"):
            warns.append(f"violations/{i}: нет ссылок на нормы (rules)")
        for j, fr in enumerate(frames):
            if not isinstance(fr, Mapping):
                continue
            problem = _b64_problem(fr.get("image_base64"))
            if problem:
                warns.append(f"violations/{i}/evidence_frames/{j}: {problem}")
    return warns


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Проверка JSON-ответа Kassa Sentinel по схеме ТЗ")
    ap.add_argument("files", nargs="+", help="JSON-файлы с ответами GET /api/v1/jobs/{id}")
    ap.add_argument("--analyze", action="store_true", help="проверять как ответ POST /analyze")
    ap.add_argument("--strict", action="store_true", help="предупреждения тоже считать ошибкой")
    args = ap.parse_args(argv)
    rc = 0
    for name in args.files:
        with open(name, encoding="utf-8") as f:
            data = json.load(f)
        errs = analyze_response_errors(data) if args.analyze else job_response_errors(data)
        warns = [] if args.analyze else check_consistency(data)
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
