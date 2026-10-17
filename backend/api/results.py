"""AnalysisResult -> JSON ответа ТЗ: сериализация, проверка схемы, починка мелочей."""
from __future__ import annotations

import base64
import binascii
import dataclasses
import json
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


OK_STAGE_STATUSES = frozenset({"ok", "done", "cached"})  # как contracts.extensions (A01)


def coverage_ratio(meta: Any) -> Optional[float]:
    """meta.coverage — число 0..1 или {"ratio": ...} (contracts.extensions.Coverage, A01)."""
    if not isinstance(meta, dict):
        return None
    cov = meta.get("coverage")
    raw = cov.get("ratio") if isinstance(cov, dict) else cov
    if isinstance(raw, bool):
        return None
    r = _num(raw)
    return None if r is None else min(1.0, max(0.0, r))


def coverage_problem(meta: Any, min_ratio: float = 0.98, *, allow_mock: bool = False,
                     pipeline_flags: bool = True) -> Optional[str]:
    """None — полнота ПОДТВЕРЖДЕНА и «нарушений нет» допустимо; иначе причина для failed.
    Зеркало contracts.extensions.coverage_problem (A01): без meta.coverage полнота не подтверждена
    (unknown != «нет нарушений»). pipeline_flags=False — для partial.json, где A03 по определению
    пишет complete=false/partial=true."""
    meta = meta if isinstance(meta, dict) else {}
    if pipeline_flags and meta.get("complete") is False:
        return "оркестратор сообщил незавершённый анализ (meta.complete=false)"
    cov = meta.get("coverage")
    if cov is None:
        return "полнота анализа не подтверждена: нет meta.coverage"
    if not isinstance(cov, dict):
        r = _num(cov)
        if r is None or isinstance(cov, bool):
            return f"meta.coverage не число и не объект: {cov!r}"[:120]
        cov = {"ratio": r, "complete": True}
    ratio = _num(cov.get("ratio"))
    if ratio is None:
        return "meta.coverage.ratio не задан"
    if ratio + 1e-9 < min_ratio:
        return f"проанализировано {ratio:.0%} видео (< {min_ratio:.0%}): неизвестные участки ≠ «нарушений нет»"
    if cov.get("complete") is False:
        return "meta.coverage.complete=false"
    stages = cov.get("stages") if isinstance(cov.get("stages"), dict) else {}
    top = meta.get("stages") if isinstance(meta.get("stages"), dict) else {}
    for req in list(cov.get("required") or meta.get("required_stages") or []):
        status = stages.get(req)
        if status is None and req in top:
            status = top[req].get("status") if isinstance(top[req], dict) else top[req]
        if status == "mock" and allow_mock:
            continue
        if status not in OK_STAGE_STATUSES:
            return f"обязательная стадия {req}: {status or 'нет статуса'}"
    return None


def analysis_complete(meta: Any, min_coverage: float = 0.98, pipeline_flags: bool = True,
                      allow_mock: bool = False) -> Optional[bool]:
    """Для отчёта: True — полнота подтверждена, False — сообщена неполнота, None — сведений нет."""
    if not isinstance(meta, dict) or (meta.get("coverage") is None and
                                      not (pipeline_flags and meta.get("complete") is False)):
        return None
    return coverage_problem(meta, min_coverage, allow_mock=allow_mock,
                            pipeline_flags=pipeline_flags) is None


def normalize(d: dict[str, Any], job_id: str, *, min_coverage: float = 0.98,
              allow_mock: bool = False, pipeline_flags: bool = True
              ) -> tuple[dict[str, Any], list[str]]:
    """Формат ТЗ + честная политика исхода (как contracts.serialize.resolve_outcome у A01):
    * violation без валидного нарушения -> failed; нет вердикта и нарушений -> failed (no_violation
      не выдумываем); no_violation при неполном анализе -> failed;
    * при no_violation найденные кандидаты уходят в candidates (только UI), жюри их не видит;
    * нарушение без конечного таймкода отбрасывается (0.0 не подставляем).
    Итог: out["status"] in {"done", "failed"}, при failed — out["error"]."""
    warns: list[str] = []
    out = dict(d)
    out["job_id"] = job_id
    if not isinstance(out.get("meta"), dict):
        out["meta"] = {}
    if out.get("status") == "failed":
        out.update(status="failed", verdict=None, violations=[])
        out["error"] = str(out.get("error") or out["meta"].get("error") or "pipeline reported failure")
        return out, warns
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
        frames = []
        for j, f in enumerate(v.get("evidence_frames") or []):
            if not isinstance(f, dict):
                continue
            b64 = _clean_b64(f.get("image_base64"))
            if b64 is None:
                warns.append(f"violation[{i}].evidence_frames[{j}] dropped: not a base64 JPEG")
                continue
            frames.append({**f, "timestamp_sec": _num(f.get("timestamp_sec")), "image_base64": b64})
        tc = _num(v.get("timecode_sec"))
        if tc is None:
            tc = _num(v.get("t_peak"))
            if tc is None:
                tc = _num(v.get("t_start"))
            if tc is None and frames and frames[0]["timestamp_sec"] is not None:
                tc = frames[0]["timestamp_sec"]
            if tc is None:
                warns.append(f"violation[{i}] dropped: no finite timecode")
                continue
            warns.append(f"violation[{i}].timecode_sec repaired from t_peak/t_start/frame")
        v["timecode_sec"] = round(max(0.0, tc), 3)
        for f in frames:
            if f["timestamp_sec"] is None:
                f["timestamp_sec"] = v["timecode_sec"]
            f["timestamp_sec"] = round(max(0.0, f["timestamp_sec"]), 3)
        if not isinstance(v.get("type"), str) or not v["type"].strip():
            v["type"] = str(v.get("type_id") or "Возможное нарушение")
            warns.append(f"violation[{i}].type missing")
        if not isinstance(v.get("description"), str):
            v["description"] = "" if v.get("description") is None else str(v["description"])
        v["evidence_frames"] = frames
        vios.append(v)

    problem = coverage_problem(out["meta"], min_coverage, allow_mock=allow_mock,
                               pipeline_flags=pipeline_flags)
    complete = analysis_complete(out["meta"], min_coverage, pipeline_flags, allow_mock)
    verdict = out.get("verdict")
    out["violations"] = vios

    def failed(reason: str) -> tuple[dict[str, Any], list[str]]:
        warns.append(reason)
        out.update(status="failed", verdict=None, violations=[], error=reason)
        if vios:
            out["candidates"] = vios
        return out, warns

    if verdict == "violation" and not vios:
        return failed("verdict=violation, но нет ни одного нарушения с конечным таймкодом")
    if verdict == "no_violation":
        if problem is not None:
            return failed(f"«нет нарушения» не подтверждено: {problem}")
        if vios:
            warns.append("verdict=no_violation: candidates are not shown to the jury (see candidates)")
            out["candidates"] = vios
            out["violations"] = []
    elif verdict != "violation":
        if verdict is not None:
            warns.append(f"invalid verdict={verdict!r}")
        if not vios:
            return failed("анализ не выдал вердикт — no_violation не выдумываем")
        out["verdict"] = "violation"
        warns.append("verdict missing; inferred 'violation' from violations")
    out["status"] = "done"
    out["analysis_complete"] = complete
    out.pop("error", None)
    return out, warns


def missing_evidence(d: dict[str, Any]) -> list[int]:
    return [i for i, v in enumerate(d.get("violations") or []) if not v.get("evidence_frames")]


def extract_jpeg(video_path: Path, t: float, max_side: int = 960, quality: int = 85,
                 time_budget_s: float = 20.0) -> Optional[str]:
    """Кадр в момент t как base64 JPEG. Время — только по PTS (CAP_PROP_POS_MSEC после grab):
    fps у роликов переменный, поэтому «номер кадра = t * fps» не годится. None — не получилось."""
    try:
        import cv2  # lazy
    except ImportError:
        return None
    import time as _time

    target = max(0.0, float(t))

    def walk(cap: Any, deadline: float) -> Any:
        frame, prev = None, None
        while _time.monotonic() < deadline:
            if not cap.grab():
                break
            pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if pos + 1e-3 >= target:
                ok, fr = cap.retrieve()
                return fr if ok else prev
            ok, prev = cap.retrieve() if pos >= target - 1.0 else (False, prev)
        return frame if frame is not None else prev

    deadline = _time.monotonic() + time_budget_s
    frame = None
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return None
        if target > 3.0:  # грубый переход ближе к t, дальше — по PTS
            cap.set(cv2.CAP_PROP_POS_MSEC, (target - 2.0) * 1000.0)
            if cap.grab() and cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 > target + 0.05:
                cap.release()  # перескочили (VFR) — идём с начала
                cap = cv2.VideoCapture(str(video_path))
        frame = walk(cap, deadline)
        if frame is None:
            cap.release()
            cap = cv2.VideoCapture(str(video_path))
            ok, frame = cap.read()
            if not ok:
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


# ------------------------------------------------------------------ представления и размер ответа
UI_ONLY_KEYS = ("candidates", "verdict_state", "partial")


def jury_description(v: dict[str, Any]) -> str:
    """Как contracts.serialize._jury_description (A01): норма rules[0] дописывается в description,
    если её URL там ещё нет — в строгой схеме ТЗ другого места для нормы нет."""
    desc = str(v.get("description") or "")
    rules = v.get("rules") if isinstance(v.get("rules"), list) else []
    rule = rules[0] if rules and isinstance(rules[0], dict) else None
    if rule and rule.get("url") and rule["url"] not in desc:
        ref = str(rule.get("document") or "")
        if rule.get("clause"):
            ref = f"{ref}, {rule['clause']}" if ref else str(rule["clause"])
        desc = f"{desc.rstrip()} Норма: {ref + ' — ' if ref else ''}{rule['url']}"
    return desc


def jury_view(d: dict[str, Any], profile: str = "extended", embed_rule: bool = False) -> dict[str, Any]:
    """profile=extended — схема ТЗ + доп. поля (разрешены ТЗ; на них опирается eval A16);
    profile=strict — РОВНО поля ТЗ (как contracts.serialize.result_to_jury_api; норма в description —
    только при embed_rule, у A01 по умолчанию выключено)."""
    if profile == "strict":
        return {
            "job_id": d.get("job_id"), "status": d.get("status"), "verdict": d.get("verdict"),
            "violations": [{
                "type": v.get("type"), "timecode_sec": v.get("timecode_sec"),
                "description": jury_description(v) if embed_rule else str(v.get("description") or ""),
                "evidence_frames": [{"timestamp_sec": f.get("timestamp_sec"),
                                     "image_base64": f.get("image_base64")}
                                    for f in (v.get("evidence_frames") or [])[:MAX_FRAMES]],
            } for v in (d.get("violations") or [])],
        }
    return {k: v for k, v in d.items() if k not in UI_ONLY_KEYS}


def status_view(job_id: str, status: str, profile: str = "extended", **extra: Any) -> dict[str, Any]:
    """processing/failed: обязательные поля есть всегда (verdict=null, violations=[]) — схема A01."""
    out: dict[str, Any] = {"job_id": job_id, "status": status, "verdict": None, "violations": []}
    if profile != "strict":
        out.update({k: v for k, v in extra.items() if v is not None})
    return out


MAX_FRAMES = 3  # ТЗ: 1–3 кадра-доказательства


def json_size(d: Any) -> int:
    return len(json.dumps(d, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _reencode_jpeg(b64: str, max_side: int, quality: int) -> Optional[str]:
    raw = base64.b64decode(b64)
    try:
        import cv2  # lazy
        import numpy as np
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        k = max_side / max(h, w)
        if k < 1:
            img = cv2.resize(img, (max(1, int(w * k)), max(1, int(h * k))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return base64.b64encode(buf.tobytes()).decode("ascii") if ok else None
    except ImportError:
        pass
    try:
        import io
        from PIL import Image  # lazy
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        im.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


def enforce_size(d: dict[str, Any], max_bytes: int) -> list[str]:
    """Ответ <= max_bytes (по умолчанию 3 МБ): по шагам, от безобидного к заметному.
    Нарушения и их таймкоды не трогаем, пока можно ужать кадры."""
    warns: list[str] = []
    for v in d.get("violations") or []:
        if len(v.get("evidence_frames") or []) > MAX_FRAMES:
            v["evidence_frames"] = v["evidence_frames"][:MAX_FRAMES]
            warns.append(f"evidence frames capped at {MAX_FRAMES} per violation")
    if json_size(d) <= max_bytes:
        return sorted(set(warns))
    for c in d.get("candidates") or []:  # кандидаты — только UI, кадры отдаются ручкой frames
        if c.get("evidence_frames"):
            c["evidence_frames"] = [{"timestamp_sec": f.get("timestamp_sec")} for f in c["evidence_frames"]]
    tl = d.get("timeline")
    if isinstance(tl, list) and len(tl) > 400:
        step = len(tl) / 400
        d["timeline"] = [tl[int(i * step)] for i in range(400)]
        warns.append("timeline downsampled to 400 points")
    meta = d.get("meta") if isinstance(d.get("meta"), dict) else {}
    if json_size(d) > max_bytes and meta:
        big = [k for k, v in meta.items() if k not in ("warnings", "api", "completeness", "coverage",
                                                       "version", "pipeline_version")
               and json_size(v) > 32 * 1024]
        for k in big:
            meta.pop(k, None)
        if big:
            meta.setdefault("api", {})["trimmed_meta_keys"] = big
            warns.append(f"large meta keys trimmed from response: {', '.join(big)}")
    for max_side, q in ((640, 75), (480, 60), (360, 50)):
        if json_size(d) <= max_bytes:
            break
        n = 0
        for v in d.get("violations") or []:
            for f in v.get("evidence_frames") or []:
                nb = _reencode_jpeg(f["image_base64"], max_side, q)
                if nb and len(nb) < len(f["image_base64"]):
                    f["image_base64"] = nb
                    n += 1
        if n:
            warns.append(f"evidence frames re-encoded to <= {max_side}px q{q} to fit response limit")
    if json_size(d) > max_bytes:
        for v in d.get("violations") or []:
            v["evidence_frames"] = (v.get("evidence_frames") or [])[:1]
        warns.append("evidence frames reduced to 1 per violation to fit response limit")
    while json_size(d) > max_bytes and len(d.get("violations") or []) > 1:
        d["violations"].pop()
        warns.append("violations truncated to fit response limit")
    return sorted(set(warns))
