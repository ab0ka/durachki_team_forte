"""Сквозной прогон: синтетическое видео -> пайплайн -> канонический сериализатор жюри -> строгая проверка ТЗ (A01).

Порядок выбора реализации:

1. если есть ``backend.pipeline.run`` (A03) — гоняем его;
2. иначе (или с ``--no-backend``) собираем пайплайн из стадий контракта: настоящая функция модуля, если модуль
   уже в ветке, иначе — фикстура тест-кита. Выход каждой стадии проверяется ``testkit.contract_checks``;
   невалидный/упавший модуль заменяется фикстурой или роняет прогон (``--strict-stages``).

``--require-full`` — проверка ПОЛНОЙ сборки: обязателен ``backend.pipeline.run``, фикстуры запрещены, ответ —
только через ``contracts.serialize.result_to_jury_api`` + ``jury_response_errors``; ``--expect-status done``
требует, чтобы оркестратор довёл каждый синтетический ролик до ``done`` (регрессия «нет meta.coverage ->
failed»), а время прогона укладывалось в ``--deadline`` (+10%). Нет ``backend.pipeline`` -> «NOT RUN», код 3.

Окружение процесса НЕ меняется. Mock-режим модулей включает вызывающий (``SENTINEL_MOCK=1`` в окружении
шага CI / оболочки). Сериализатор допускает mock только (а) для собственных фикстур тест-кита и (б) для
результата оркестратора, если mock включён явно; mock-результат без явного режима — ошибка
(production не переключается на mock сам).

Коды возврата: 0 — ок; 1 — ошибка (схема жюри, падение, дедлайн, статус, точность); 3 — NOT RUN.

    SENTINEL_MOCK=1 python -m testkit.e2e_stub --require-full --expect-status done --scenarios clean,V01
    python -m testkit.e2e_stub --no-backend --strict-stages        # контракт стадий (частичная сборка)
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

from contracts.extensions import Provenance, compute_coverage, is_mock_result, mock_allowed
from contracts.sentinel_contracts import AnalysisResult, Event, EvidenceFrame, Violation
from contracts.serialize import resolve_outcome, result_to_jury_api
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

__all__ = ["StageReport", "run_stub_pipeline", "run_backend", "run_scenario", "run_case", "main", "EXIT_NOT_RUN"]

EXIT_NOT_RUN = 3


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


class _Fixtures:
    """Фикстуры стадий по GT: кейс спецификации (``gt['spec']``, testkit.spec) или сценарий v1 (``gt['scenario']``)."""

    def __init__(self, gt: Optional[dict], sample_fps: float, duration: float):
        self.gt, self.fps, self.d = gt or {}, sample_fps, float((gt or {}).get("duration", duration))
        self.spec = self.gt.get("spec")
        self.scenario = self.gt.get("scenario") if self.gt.get("scenario") in SCENARIOS else None
        self.view = self.gt.get("view", "default")
        self._bundle = None

    def _b(self):
        if self._bundle is None:
            from .spec import CaseSpec, case_bundle

            self._bundle = case_bundle(CaseSpec(**self.spec), self.fps)
        return self._bundle

    def detections(self):
        if self.spec:
            return self._b().detections
        return make_detections(self.scenario, self.d, self.fps, view=self.view) if self.scenario else []

    def tracks_zones(self):
        if self.spec:
            return self._b().tracks, self._b().zones
        return (make_tracks(self.scenario, self.d, self.fps, view=self.view), make_zones(self.view)) \
            if self.scenario else ([], [])

    def poses(self):
        if self.spec:
            return self._b().poses
        return make_pose_frames(self.scenario, self.d, self.fps, view=self.view) if self.scenario else []

    def states(self):
        if self.spec:
            return self._b().states
        return make_scene_states(self.scenario, self.d, self.fps) if self.scenario else []

    def events(self):
        if self.spec:
            return self._b().events
        return make_expected_events(self.scenario, self.d) if self.scenario else []


def run_stub_pipeline(video_path: str, job_id: str, gt: Optional[dict] = None, *, prefer_real: bool = True,
                      strict: bool = False, sample_fps: float = 5.0) -> tuple[AnalysisResult, list[StageReport]]:
    """Пайплайн из стадий контракта: настоящие модули, где есть, иначе фикстуры (кейс spec или сценарий v1)."""
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
    fx = _Fixtures(gt, sample_fps, dur)
    dets = stage("detect", "ml.detect", "detect", (clip,), fx.detections, cc.check_detections)
    tracks, zones = stage("track", "ml.track", "track", (clip, dets), fx.tracks_zones, cc.check_tracks_zones)
    poses = stage("pose", "ml.pose", "estimate", (clip, tracks), fx.poses, cc.check_pose_frames)
    pose_states = stage("pose_signals", "ml.pose", "pose_signals", (poses, tracks, zones), lambda: [],
                        cc.check_scene_states)
    states = stage("scene", "ml.scene", "scene_states", (clip, dets, tracks, zones, pose_states), fx.states,
                   cc.check_scene_states)
    catalog = stage("norms", "norms", "load_catalog", (), make_catalog,
                    lambda c: [] if isinstance(c, dict) and isinstance(c.get("types"), dict)
                    else ["load_catalog: ожидается dict с ключом types"])
    rule_events = stage("rules", "ml.rules", "run_rules", (states, tracks, zones, catalog), fx.events,
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
                 deadline_s: float = 240.0, require_backend: bool = False,
                 expect_status: Optional[str] = None) -> dict:
    """Сценарий v1 (testkit.scenarios)."""
    suffix = "" if view == "default" else f"_{view}"
    video = out_dir / f"{scenario}{suffix}.mov"
    gt = make_video(video, scenario, duration=duration, fps=fps, view=view)
    return _run(video, gt, f"e2e-{scenario}{suffix}", scenario, use_backend, prefer_real, strict, deadline_s,
                require_backend, expect_status)


def run_case(spec: Any, out_dir: Path, *, fps: float = 10.0, use_backend: bool = True, prefer_real: bool = True,
             strict: bool = False, deadline_s: float = 240.0, require_backend: bool = False,
             expect_status: Optional[str] = None) -> dict:
    """Кейс спецификации (testkit.spec.CaseSpec)."""
    from .spec import make_case_video

    name = f"{spec.case}_{spec.layout}_{spec.orientation}"
    gt = make_case_video(out_dir / f"{name}.mov", spec, fps=fps)
    gt["view"] = spec.view
    return _run(out_dir / f"{name}.mov", gt, f"e2e-{name}", name, use_backend, prefer_real, strict, deadline_s,
                require_backend, expect_status)


def _run(video: Path, gt: dict, job_id: str, label: str, use_backend: bool, prefer_real: bool, strict: bool,
         deadline_s: float, require_backend: bool = False, expect_status: Optional[str] = None) -> dict:
    duration = float(gt["duration"])
    rec: dict[str, Any] = {"scenario": label, "view": gt.get("view", "default"), "gt_verdict": gt["verdict"],
                           "gt_type": gt["type_id"], "gt_t": gt["t"], "stages": [], "errors": [], "not_run": None}
    t0 = time.perf_counter()
    try:
        result = run_backend(str(video), job_id, deadline_s) if use_backend else None
        if result is not None:
            rec["runner"] = "backend.pipeline"
        elif require_backend:
            rec["runner"] = None
            rec["not_run"] = "нет backend.pipeline.run (A03) — полная сборка не проверена"
            return rec
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
    meta = result.meta if isinstance(result.meta, dict) else {}
    rec["mock_result"] = is_mock_result(meta)
    rec["fixture_stages"] = [s["name"] for s in rec["stages"] if s["source"] != "real"]
    if rec["runner"] == "backend.pipeline":
        # результат оркестратора судится по правилам production: mock допустим только при явном режиме
        allow = mock_allowed()
        if rec["mock_result"] and not allow:
            rec["errors"].append("оркестратор отдал mock-результат, хотя mock-режим явно не включён "
                                 "(SENTINEL_MOCK) — production не должен переключаться на mock сам")
        if rec["seconds"] > deadline_s * 1.1:
            rec["errors"].append(f"дедлайн превышен: {rec['seconds']} с > {deadline_s} с")
    else:
        allow = True  # собственные фикстуры тест-кита — явно mock
    body = result_to_jury_api(result, allow_mock=allow)
    oc = resolve_outcome(result, allow_mock=allow)
    rec["response"] = {k: body[k] for k in ("job_id", "status", "verdict")}
    rec["reason"] = oc.error
    rec["violations"] = [{"type": v["type"], "timecode_sec": v["timecode_sec"]} for v in body["violations"]]
    rec["errors"] += jury_response_errors(body, duration)
    if expect_status and body["status"] != expect_status:
        rec["errors"].append(f"ожидался status={expect_status}, получен {body['status']}: {oc.error}")
    pred_types = {getattr(v, "type_id", "") for v in result.violations}
    ok_types = {gt["type_id"]} | {a for e in gt.get("events", []) for a in [e["type_id"], *e.get("alt_type_ids", [])]}
    rec["verdict_ok"] = body["verdict"] == gt["verdict"]
    rec["type_ok"] = gt["type_id"] is None or bool(pred_types & ok_types)
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Сквозной mock-прогон Kassa Sentinel на синтетических роликах")
    ap.add_argument("--suite", choices=("v1", "spec"), default="v1", help="v1 — сценарии scenarios.py; spec — кейсы "
                    "спецификации (testkit.spec)")
    ap.add_argument("--scenarios", default=",".join(SCENARIOS), help="v1: список сценариев")
    ap.add_argument("--cases", default=None, help="spec: список кейсов (по умолчанию все)")
    ap.add_argument("--layouts", default="A,B")
    ap.add_argument("--orientations", default="portrait")
    ap.add_argument("--duration-range", default="14,20", help="spec: диапазон длительности, с")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--view", default="default")
    ap.add_argument("--out", default=None, help="папка для роликов и отчёта (по умолчанию временная)")
    ap.add_argument("--no-backend", action="store_true", help="не вызывать backend.pipeline.run")
    ap.add_argument("--fixtures-only", action="store_true", help="все стадии — фикстуры (самопроверка тест-кита)")
    ap.add_argument("--strict-stages", action="store_true", help="падение/невалидный выход модуля -> код 1")
    ap.add_argument("--require-accuracy", action="store_true", help="вердикт и тип должны совпасть с GT")
    # 240 с — штатный дедлайн контракта: меньший меняет адаптивный sample_fps оркестратора и маскирует ошибки
    ap.add_argument("--deadline", type=float, default=240.0, help="deadline_s для backend.pipeline.run, с")
    ap.add_argument("--require-full", action="store_true", help="полная сборка: обязателен backend.pipeline.run, "
                    "без фикстур; нет оркестратора -> NOT RUN (код 3)")
    ap.add_argument("--expect-status", choices=("done", "failed", "processing"), default=None,
                    help="каждый ответ жюри должен иметь этот статус")
    args = ap.parse_args(argv)
    if args.require_full and (args.no_backend or args.fixtures_only):
        ap.error("--require-full несовместим с --no-backend/--fixtures-only")
    # окружение процесса не меняем: mock-режим модулей включает вызывающий (SENTINEL_MOCK=1)
    print(f"mock-режим модулей: {'включён явно (SENTINEL_MOCK)' if mock_allowed() else 'выключен'}")

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
        common = dict(use_backend=not args.no_backend and not args.fixtures_only, prefer_real=not args.fixtures_only,
                      strict=args.strict_stages or args.require_full, deadline_s=args.deadline,
                      require_backend=args.require_full, expect_status=args.expect_status)
        if args.suite == "spec":
            from .spec import make_suite

            lo, hi = (float(x) for x in args.duration_range.split(","))
            cases = [c.strip() for c in args.cases.split(",")] if args.cases else None
            jobs = [(f"{s.case}_{s.layout}", lambda s=s: run_case(s, out_dir, fps=args.fps, **common))
                    for s in make_suite(args.seed, cases, tuple(args.layouts.split(",")),
                                        tuple(args.orientations.split(",")), duration_range=(lo, hi))]
        else:
            jobs = [(sc, lambda sc=sc: run_scenario(sc, out_dir, duration=args.duration, fps=args.fps, view=args.view,
                                                    **common))
                    for sc in [s.strip() for s in args.scenarios.split(",") if s.strip()]]
        for sc, job in jobs:
            rec = job()
            records.append(rec)
            if rec.get("not_run"):
                print(f"[NOT RUN] {sc}: {rec['not_run']}")
                continue
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
        not_run = [r for r in records if r.get("not_run")]
        partial = sorted({st for r in records for st in r.get("fixture_stages", [])})
        full = bool(records) and not not_run and all(r.get("runner") == "backend.pipeline" for r in records)
        print(f"verdict accuracy (информативно в mock-режиме): {n_ok}/{len(records)}; отчёт: {out_dir / 'e2e_report.json'}")
        n_err = sum(1 for r in records if r.get("errors"))
        if full:
            print(f"ПОЛНАЯ СБОРКА ПРОВЕРЕНА (backend.pipeline.run + канонический сериализатор жюри): "
                  f"{'ошибок нет' if n_err == 0 else f'прогонов с ошибками: {n_err} из {len(records)}'}")
        else:
            print("ПОЛНАЯ СБОРКА НЕ ПРОВЕРЕНА: " + (f"{len(not_run)} прогонов NOT RUN; " if not_run else "")
                  + (f"стадии из фикстур: {', '.join(partial)}" if partial else "оркестратор не вызывался"))
        if not_run and rc == 0:
            rc = EXIT_NOT_RUN
    finally:
        if any(r.get("runner") == "backend.pipeline" for r in records):
            _release_backend()
        if tmp is not None:
            tmp.cleanup()
    return rc


def _release_backend() -> None:
    """Остановить процесс-воркер оркестратора (A03: ``release_gpu``) — иначе интерпретатор не завершается."""
    release = _resolve("backend.pipeline", "release_gpu")
    if release is not None:
        try:
            release()
        except Exception as e:  # noqa: BLE001
            print(f"backend.pipeline.release_gpu: {type(e).__name__}: {e}")


if __name__ == "__main__":
    sys.exit(main())
