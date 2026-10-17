"""AnalysisResult -> JSON ответа ТЗ: сериализация, проверка схемы, починка мелочей."""
from __future__ import annotations

import base64
import binascii
import dataclasses
import math
from pathlib import Path
from typing import Any, Optional

VERDICTS = ("violation", "no_violation")


def jsonable(x: Any) -> Any:
    """Рекурсивно в JSON-совместимое: dataclass/numpy/Path/set; NaN/inf -> None."""
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {f.name: jsonable(getattr(x, f.name)) for f in dataclasses.fields(x)}
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in x]
    if isinstance(x, bool) or x is None or isinstance(x, (int, str)):
        return x
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, bytes):
        return base64.b64encode(x).decode("ascii")
    item = getattr(x, "item", None)  # numpy scalar
    if callable(item):
        try:
            return jsonable(item())
        except Exception:
            pass
    tolist = getattr(x, "tolist", None)  # numpy array
    if callable(tolist):
        try:
            return jsonable(tolist())
        except Exception:
            pass
    return str(x)


def fallback_result_to_api(result: Any) -> dict[str, Any]:
    """Своя сериализация строго по ТЗ + доп. поля (если contracts.serialize недоступен)."""
    d = jsonable(result)
    if not isinstance(d, dict):
        raise TypeError(f"pipeline returned {type(result).__name__}, expected AnalysisResult")
    out: dict[str, Any] = {"job_id": d.get("job_id"), "status": d.get("status"),
                           "verdict": d.get("verdict"), "violations": []}
    for v in d.get("violations") or []:
        if not isinstance(v, dict):
            continue
        out["violations"].append({
            "type": v.get("type"), "timecode_sec": v.get("timecode_sec"),
            "description": v.get("description"),
            "evidence_frames": [{"timestamp_sec": f.get("timestamp_sec"),
                                 "image_base64": f.get("image_base64")}
                                for f in (v.get("evidence_frames") or []) if isinstance(f, dict)],
            "type_id": v.get("type_id", ""), "confidence": v.get("confidence", 0.0),
            "t_start": v.get("t_start", 0.0), "t_end": v.get("t_end", 0.0),
            "rules": v.get("rules") or [],
        })
    out["timeline"] = d.get("timeline") or []
    out["meta"] = d.get("meta") or {}
    return out


def to_api_dict(result: Any) -> tuple[dict[str, Any], list[str]]:
    """Через contracts.serialize.result_to_api (A01), иначе своя сериализация."""
    warns: list[str] = []
    if isinstance(result, dict):
        return jsonable(result), warns
    try:
        from contracts.serialize import result_to_api  # type: ignore
    except ImportError:
        return fallback_result_to_api(result), warns
    try:
        return jsonable(result_to_api(result)), warns
    except Exception as e:
        warns.append(f"contracts.serialize failed ({type(e).__name__}); used API fallback serializer")
        return fallback_result_to_api(result), warns


def _num(x: Any) -> Optional[float]:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _clean_b64(s: Any) -> Optional[str]:
    if not isinstance(s, str) or not s:
        return None
    if s.startswith("data:"):
        s = s.split(",", 1)[-1]
    s = "".join(s.split())
    try:
        raw = base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError):
        return None
    return s if raw[:2] == b"\xff\xd8" else None  # только JPEG


def normalize(d: dict[str, Any], job_id: str) -> tuple[dict[str, Any], list[str]]:
    """Гарантирует обязательные поля ТЗ и типы. Ничего не придумывает: только чинит формат."""
    warns: list[str] = []
    out = dict(d)
    out["job_id"] = job_id
    out["status"] = "done"
    vios_in = out.get("violations")
    if not isinstance(vios_in, list):
        if vios_in is not None:
            warns.append("violations was not a list")
        vios_in = []
    vios: list[dict[str, Any]] = []
    for i, v in enumerate(vios_in):
        if not isinstance(v, dict):
            warns.append(f"violation[{i}] dropped: not an object")
            continue
        v = dict(v)
        tc = _num(v.get("timecode_sec"))
        if tc is None:
            tc = _num(v.get("t_peak")) or _num(v.get("t_start"))
            warns.append(f"violation[{i}].timecode_sec invalid; repaired")
        v["timecode_sec"] = round(max(0.0, tc or 0.0), 3)
        if not isinstance(v.get("type"), str) or not v["type"].strip():
            v["type"] = str(v.get("type_id") or "Возможное нарушение")
            warns.append(f"violation[{i}].type missing")
        if not isinstance(v.get("description"), str):
            v["description"] = "" if v.get("description") is None else str(v["description"])
        frames = []
        for j, f in enumerate(v.get("evidence_frames") or []):
            if not isinstance(f, dict):
                continue
            b64 = _clean_b64(f.get("image_base64"))
            if b64 is None:
                warns.append(f"violation[{i}].evidence_frames[{j}] dropped: not a base64 JPEG")
                continue
            ts = _num(f.get("timestamp_sec"))
            frames.append({**f, "timestamp_sec": round(ts if ts is not None else v["timecode_sec"], 3),
                           "image_base64": b64})
        v["evidence_frames"] = frames
        vios.append(v)
    out["violations"] = vios
    if out.get("verdict") not in VERDICTS:
        out["verdict"] = "violation" if vios else "no_violation"
        warns.append("verdict missing/invalid; inferred from violations")
    elif out["verdict"] == "no_violation" and vios:
        warns.append("verdict=no_violation but violations list is not empty (left as is)")
    if not isinstance(out.get("meta"), dict):
        out["meta"] = {}
    return out, warns


def missing_evidence(d: dict[str, Any]) -> list[int]:
    return [i for i, v in enumerate(d.get("violations") or []) if not v.get("evidence_frames")]


def extract_jpeg(video_path: Path, t: float, max_side: int = 960, quality: int = 85) -> Optional[str]:
    """Кадр из видео в момент t как base64 JPEG (cv2, лениво). None если не получилось."""
    try:
        import cv2  # lazy
    except ImportError:
        return None
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(t)) * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if not ok or frame is None:
            return None
        h, w = frame.shape[:2]
        k = max_side / max(h, w)
        if k < 1:
            frame = cv2.resize(frame, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return base64.b64encode(buf.tobytes()).decode("ascii") if ok else None
    except Exception:
        return None
    finally:
        cap.release()


def fill_missing_evidence(d: dict[str, Any], video_path: Path) -> list[str]:
    """ТЗ требует кадр к каждому нарушению: если fusion его не дал — берём кадр в timecode_sec."""
    warns = []
    for i in missing_evidence(d):
        v = d["violations"][i]
        b64 = extract_jpeg(video_path, v["timecode_sec"])
        if b64:
            v["evidence_frames"] = [{"timestamp_sec": v["timecode_sec"], "image_base64": b64}]
            warns.append(f"violation[{i}]: evidence frame extracted by API at timecode (no blur applied)")
    return warns


def iter_frames(d: dict[str, Any]) -> list[tuple[int, int, dict[str, Any]]]:
    """Плоская нумерация кадров-доказательств: (i, индекс нарушения, кадр)."""
    out = []
    for vi, v in enumerate(d.get("violations") or []):
        for f in v.get("evidence_frames") or []:
            out.append((len(out), vi, f))
    return out


SCHEMA_REQUIRED = {
    "job_id": str, "status": str, "verdict": str, "violations": list,
}
VIOLATION_REQUIRED = {"type": str, "timecode_sec": (int, float), "description": str,
                      "evidence_frames": list}
FRAME_REQUIRED = {"timestamp_sec": (int, float), "image_base64": str}


def schema_errors(d: dict[str, Any]) -> list[str]:
    """Проверка ответа done ровно по ТЗ (используется в тестах и при финализации)."""
    errs = []
    for k, t in SCHEMA_REQUIRED.items():
        if not isinstance(d.get(k), t):
            errs.append(f"{k}: expected {t}")
    if d.get("status") not in ("processing", "done", "failed"):
        errs.append("status: bad value")
    if d.get("verdict") not in VERDICTS:
        errs.append("verdict: bad value")
    for i, v in enumerate(d.get("violations") or []):
        for k, t in VIOLATION_REQUIRED.items():
            if not isinstance(v.get(k), t) or isinstance(v.get(k), bool):
                errs.append(f"violations[{i}].{k}: expected {t}")
        for j, f in enumerate(v.get("evidence_frames") or []):
            for k, t in FRAME_REQUIRED.items():
                if not isinstance(f.get(k), t) or isinstance(f.get(k), bool):
                    errs.append(f"violations[{i}].evidence_frames[{j}].{k}: expected {t}")
    return errs
