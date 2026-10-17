"""Сквозной mock-прогон: синтетическое видео -> пайплайн -> ответ жюри -> строгая проверка схемы ТЗ (A01).

Порядок выбора реализации:

1. если есть ``backend.pipeline.run`` (A03) — гоняем его (``SENTINEL_MOCK=1``);
2. иначе собираем пайплайн из стадий контракта: для каждой стадии берём настоящую функцию модуля, если модуль
   уже в ветке, иначе — фикстуру тест-кита (``testkit.fixtures``). Выход каждой стадии проверяется
   ``testkit.contract_checks``; невалидный или упавший модуль заменяется фикстурой (или роняет прогон
   с ``--strict-stages``).

Код возврата 1 — ответ не прошёл строгую схему жюри / пайплайн упал / (опционально) стадии или вердикты не те.

    python -m testkit.e2e_stub                         # все сценарии, 8 с, 10 fps
    python -m testkit.e2e_stub --scenarios clean,V01 --strict-stages
    python -m testkit.e2e_stub --fixtures-only --require-accuracy   # самопроверка тест-кита
"""
from __future__ import annotations

import argparse
import base64
import importlib
import json
import os
import sys
import tempfile
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from contracts.extensions import Provenance, compute_coverage
from contracts.sentinel_contracts import AnalysisResult, Event, EvidenceFrame, Violation
from contracts.serialize import result_to_jury_api
from contracts.validate import jury_response_errors

from . import contract_checks as cc
from .fake_clip import clip_from_video
from .fixtures import (
    TYPE_NAMES,
    make_catalog,
    make_detections,
    make_expected_events,
    make_pose_frames,
    make_scene_states,
    make_tracks,
    make_zones,
)
from .scenarios import SCENARIOS
from .synth_video import make_video

__all__ = ["StageReport", "run_stub_pipeline", "run_backend", "run_scenario", "main"]


@dataclass
class StageReport:
    name: str
    source: str                 # real | fixture | real->fixture
    seconds: float = 0.0
    problems: list[str] = field(default_factory=list)
    error: Optional[str] = None


class StageError(RuntimeError):
    pass


def _resolve(module: str, func: str) -> Optional[Callable]:
    try:
        mod = importlib.import_module(module)
    except ImportError:
        return None
    except Exception:  # noqa: BLE001 — сломанный импорт модуля тоже повод для фолбэка
        return None
    fn = getattr(mod, func, None)
    return fn if callable(fn) else None


def _jpeg_b64(img: Any) -> Optional[str]:
    try:
        import cv2
    except ImportError:
        return None
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else None


def fallback_fuse(clip: Any, events: list[Event], risk: list, catalog: dict, job_id: str,
                  threshold: float = 0.5) -> AnalysisResult:
    """Простейший fusion для mock-прогона: лучшее событие каждого типа с confidence >= threshold."""
    best: dict[str, Event] = {}
    for e in events:
        if e.confidence >= threshold and (e.type_id not in best or e.confidence > best[e.type_id].confidence):
            best[e.type_id] = e
    violations = []
    for e in sorted(best.values(), key=lambda x: x.t_peak):
        b64 = _jpeg_b64(clip.frame_at(e.t_peak))
        frames = [EvidenceFrame(timestamp_sec=e.t_peak, image_base64=b64)] if b64 else []
        name = catalog.get("types", {}).get(e.type_id, {}).get("name") or TYPE_NAMES.get(e.type_id, e.type_id)
        facts = "; ".join(e.facts) or "наблюдаемые признаки по данным модулей"
        violations.append(Violation(type=name, timecode_sec=e.t_peak, description=f"На {e.t_peak:.1f} с: {facts}. "
                                    "Признаки возможного нарушения — требуется проверка человеком.",
                                    evidence_frames=frames, type_id=e.type_id, confidence=e.confidence,
                                    t_start=e.t_start, t_end=e.t_end, rules=[]))
    # Явная пометка: результат mock-прогона тест-кита (production такой результат без SENTINEL_MOCK=1 не отдаст).
    meta = {"duration_sec": clip.duration, "mock": True,
            "coverage": compute_coverage(clip.duration, clip.duration, stages={"fusion": "mock"},
                                         required=["fusion"], source="testkit-e2e-stub"),
            "provenance": Provenance(pipeline_version="testkit-e2e-stub", mock=True, sources=["fusion"]).to_dict()}
    return AnalysisResult(job_id=job_id, status="done", verdict="violation" if violations else "no_violation",
                          violations=violations, timeline=[(float(t), float(r)) for t, r in risk], meta=meta)


def run_stub_pipeline(video_path: str, job_id: str, gt: Optional[dict] = None, *, prefer_real: bool = True,
                      strict: bool = False, sample_fps: float = 5.0) -> tuple[AnalysisResult, list[StageReport]]:
    """Пайплайн из стадий контракта: настоящие модули, где есть, иначе фикстуры по ``gt['scenario']``."""
    scenario = (gt or {}).get("scenario")
    view = (gt or {}).get("view", "default")
    reports: list[StageReport] = []

    def stage(name: str, module: str, func: str, args: tuple, fallback: Callable[[], Any],
              check: Callable[[Any], list[str]]) -> Any:
        fn = _resolve(module, func) if prefer_real else None
        t0 = time.perf_counter()
        if fn is not None:
            rep = StageReport(name, "real")
            try:
                out = fn(*args)
                rep.problems = check(out)
            except Exception as e:  # noqa: BLE001
                rep.error = f"{type(e).__name__}: {e}"
                out = None
            rep.seconds = round(time.perf_counter() - t0, 3)
            if rep.error is None and not rep.problems:
                reports.append(rep)
                return out
            if strict:
                reports.append(rep)
                raise StageError(f"стадия {name} ({module}.{func}): {rep.error or rep.problems[:3]}")
            rep.source = "real->fixture"
            out = fallback()
            reports.append(rep)
            return out
        out = fallback()
        reports.append(StageReport(name, "fixture", round(time.perf_counter() - t0, 3)))
        return out

    clip = stage("ingest", "ml.ingest", "load_video", (video_path, sample_fps),
                 lambda: clip_from_video(video_path, sample_fps), cc.check_clip)
    dur = float(clip.duration)
    d = (gt or {}).get("duration", dur)
    dets = stage("detect", "ml.detect", "detect", (clip,),
                 lambda: make_detections(scenario, d, sample_fps, view=view) if scenario else [], cc.check_detections)
    tracks, zones = stage("track", "ml.track", "track", (clip, dets),
                          lambda: (make_tracks(scenario, d, sample_fps, view=view), make_zones(view))
                          if scenario else ([], []), cc.check_tracks_zones)
    poses = stage("pose", "ml.pose", "estimate", (clip, tracks),
                  lambda: make_pose_frames(scenario, d, sample_fps, view=view) if scenario else [],
                  cc.check_pose_frames)
    pose_states = stage("pose_signals", "ml.pose", "pose_signals", (poses, tracks, zones), lambda: [],
                        cc.check_scene_states)
    states = stage("scene", "ml.scene", "scene_states", (clip, dets, tracks, zones, pose_states),
                   lambda: make_scene_states(scenario, d, sample_fps) if scenario else [], cc.check_scene_states)
    catalog = stage("norms", "norms", "load_catalog", (), make_catalog,
                    lambda c: [] if isinstance(c, dict) and isinstance(c.get("types"), dict)
                    else ["load_catalog: ожидается dict с ключом types"])
    rule_events = stage("rules", "ml.rules", "run_rules", (states, tracks, zones, catalog),
                        lambda: make_expected_events(scenario, d) if scenario else [],
                        lambda ev: cc.check_events(ev, dur))
    risk, t_events = stage("temporal", "ml.temporal", "score", (clip,), lambda: ([], []),
                           lambda r: (cc.check_risk(r[0], dur) + cc.check_events(r[1], dur))
                           if isinstance(r, tuple) and len(r) == 2 else ["temporal.score: ожидается (risk, events)"])
    events = list(rule_events) + list(t_events)
    verified = stage("vlm_verify", "ml.vlm", "verify", (clip, events, 20.0), lambda: events,
                     lambda ev: cc.check_events(ev, dur))
    discovered = stage("vlm_discover", "ml.vlm", "discover", (clip, 10.0), lambda: [],
                       lambda ev: cc.check_events(ev, dur))
    all_events = list(verified) + list(discovered)
    result = stage("fusion", "ml.fusion", "fuse", (clip, all_events, risk, catalog, job_id),
                   lambda: fallback_fuse(clip, all_events, risk, catalog, job_id),
                   lambda r: cc.check_result(r, dur))
    return result, reports


def run_backend(video_path: str, job_id: str, deadline_s: float = 240.0) -> Optional[AnalysisResult]:
    """``backend.pipeline.run`` (A03), если он уже в ветке; иначе None."""
    run = _resolve("backend.pipeline", "run")
    if run is None:
        return None
    stages: list[tuple[str, float]] = []
    return run(video_path, job_id, lambda stage, p: stages.append((stage, p)), deadline_s=deadline_s)


def run_scenario(scenario: str, out_dir: Path, *, duration: float = 8.0, fps: float = 10.0, view: str = "default",
                 use_backend: bool = True, prefer_real: bool = True, strict: bool = False,
                 deadline_s: float = 120.0) -> dict:
    suffix = "" if view == "default" else f"_{view}"
    video = out_dir / f"{scenario}{suffix}.mov"
    gt = make_video(video, scenario, duration=duration, fps=fps, view=view)
    job_id = f"e2e-{scenario}{suffix}"
    rec: dict[str, Any] = {"scenario": scenario, "view": view, "gt_verdict": gt["verdict"], "gt_type": gt["type_id"],
                           "gt_t": gt["t"], "stages": [], "errors": []}
    t0 = time.perf_counter()
    try:
        result = run_backend(str(video), job_id, deadline_s) if use_backend else None
        if result is not None:
            rec["runner"] = "backend.pipeline"
        else:
            result, reports = run_stub_pipeline(str(video), job_id, gt, prefer_real=prefer_real, strict=strict)
            rec["runner"] = "stub"
            rec["stages"] = [asdict(r) for r in reports]
    except Exception as e:  # noqa: BLE001
        rec["runner"] = rec.get("runner", "?")
        rec["errors"].append(f"пайплайн упал: {type(e).__name__}: {e}")
        rec["traceback"] = traceback.format_exc(limit=5)
        rec["seconds"] = round(time.perf_counter() - t0, 2)
        return rec
    rec["seconds"] = round(time.perf_counter() - t0, 2)
    body = result_to_jury_api(result, allow_mock=True)  # e2e-заглушка — явно mock-прогон
    rec["response"] = {k: body[k] for k in ("job_id", "status", "verdict")}
    rec["violations"] = [{"type": v["type"], "timecode_sec": v["timecode_sec"]} for v in body["violations"]]
    rec["errors"] += jury_response_errors(body, duration)
    pred_types = {getattr(v, "type_id", "") for v in result.violations}
    rec["verdict_ok"] = body["verdict"] == gt["verdict"]
    rec["type_ok"] = gt["type_id"] is None or gt["type_id"] in pred_types
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Сквозной mock-прогон Kassa Sentinel на синтетических роликах")
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--view", default="default")
    ap.add_argument("--out", default=None, help="папка для роликов и отчёта (по умолчанию временная)")
    ap.add_argument("--no-backend", action="store_true", help="не вызывать backend.pipeline.run")
    ap.add_argument("--fixtures-only", action="store_true", help="все стадии — фикстуры (самопроверка тест-кита)")
    ap.add_argument("--strict-stages", action="store_true", help="падение/невалидный выход модуля -> код 1")
    ap.add_argument("--require-accuracy", action="store_true", help="вердикт и тип должны совпасть с GT")
    ap.add_argument("--deadline", type=float, default=120.0, help="deadline_s для backend.pipeline.run, с")
    args = ap.parse_args(argv)
    os.environ.setdefault("SENTINEL_MOCK", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    tmp = None
    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        tmp = tempfile.TemporaryDirectory(prefix="sentinel-e2e-")
        out_dir = Path(tmp.name)
    rc = 0
    records = []
    try:
        for sc in [s.strip() for s in args.scenarios.split(",") if s.strip()]:
            rec = run_scenario(sc, out_dir, duration=args.duration, fps=args.fps, view=args.view,
                               use_backend=not args.no_backend and not args.fixtures_only,
                               prefer_real=not args.fixtures_only, strict=args.strict_stages,
                               deadline_s=args.deadline)
            records.append(rec)
            stages = " ".join(f"{s['name']}:{s['source']}" for s in rec["stages"]) or "-"
            resp = rec.get("response", {})
            mark = "OK " if not rec["errors"] else "ERR"
            print(f"[{mark}] {sc:<16} runner={rec.get('runner')} status={resp.get('status')} "
                  f"verdict={resp.get('verdict')} (gt {rec['gt_verdict']}/{rec['gt_type']}) {rec.get('seconds')}s")
            print(f"      stages: {stages}")
            for s in rec["stages"]:
                if s["error"] or s["problems"]:
                    msg = s["error"] or "; ".join(s["problems"][:3])
                    print(f"      ! {s['name']} ({s['source']}): {msg}")
                    if os.environ.get("GITHUB_ACTIONS"):
                        print(f"::warning::e2e {sc}: стадия {s['name']}: {msg[:300]}")
            for e in rec["errors"]:
                print(f"      ERROR {e}")
            if rec["errors"]:
                rc = 1
            if args.require_accuracy and not (rec.get("verdict_ok") and rec.get("type_ok")):
                print("      ERROR вердикт/тип не совпали с GT")
                rc = 1
        (out_dir / "e2e_report.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        n_ok = sum(1 for r in records if r.get("verdict_ok"))
        print(f"verdict accuracy (информативно в mock-режиме): {n_ok}/{len(records)}; отчёт: {out_dir / 'e2e_report.json'}")
    finally:
        if tmp is not None:
            tmp.cleanup()
    return rc


if __name__ == "__main__":
    sys.exit(main())
