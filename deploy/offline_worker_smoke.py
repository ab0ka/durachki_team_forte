#!/usr/bin/env python3
"""Доказательство локальности: РЕАЛЬНЫЙ worker (backend.pipeline.run) на локальных весах без сети.

Это проверка worker-а, а не публичного API: входящий HTTP проверяют отдельно (healthcheck.py,
tunnel_probe.py). Порядок:
  1) жёсткий офлайн HF/transformers + netguard=block в этом же процессе (до импорта моделей);
  2) weights.py verify: каждый файл из deploy/weights.lock.json на месте и совпадает по sha256;
  3) backend.pipeline.warmup() + run() на синтетическом видео (SENTINEL_MOCK=1 → отказ);
  4) FAIL, если была хоть одна попытка выхода в сеть (даже если пайплайн её «проглотил»).
Отчёт — только числа/типы/таймкоды; ни кадров, ни описаний (в т.ч. чтобы не утекли в чат-ассистента).

  python deploy/offline_worker_smoke.py                        # Windows/Linux нативно
  docker run --rm --network none -v ./weights:/app/weights:ro IMAGE python deploy/offline_worker_smoke.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
               "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1"}


def _install_guard(log_path: Path) -> object:
    try:
        from deploy.netguard import sentinel_netguard as ng
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent / "netguard"))
        import sentinel_netguard as ng  # type: ignore
    ng.install("block", str(log_path))
    return ng


def _get(obj: object, key: str, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def check_result(result: object) -> list[str]:
    problems = []
    if _get(result, "status") != "done":
        problems.append(f"status={_get(result, 'status')!r}")
    if _get(result, "verdict") not in ("violation", "no_violation"):
        problems.append(f"verdict={_get(result, 'verdict')!r}")
    for i, v in enumerate(_get(result, "violations", []) or []):
        for k in ("type", "timecode_sec", "description", "evidence_frames"):
            if _get(v, k) in (None, ""):
                problems.append(f"violations[{i}].{k} пусто")
    return problems


def summarize(result: object) -> dict:
    vs = _get(result, "violations", []) or []
    meta = _get(result, "meta", {}) or {}
    return {"status": _get(result, "status"), "verdict": _get(result, "verdict"), "n_violations": len(vs),
            "violations": [{"type_id": _get(v, "type_id", ""), "timecode_sec": _get(v, "timecode_sec"),
                            "n_frames": len(_get(v, "evidence_frames", []) or [])} for v in vs],
            "meta_keys": sorted(meta) if isinstance(meta, dict) else [],
            "meta_warnings": (meta.get("warnings") if isinstance(meta, dict) else None)}


def gpu_evidence() -> dict:
    torch = sys.modules.get("torch")
    if torch is None:
        return {"torch_imported": False}
    out = {"torch_imported": True, "cuda_available": bool(torch.cuda.is_available())}
    if out["cuda_available"]:
        out["device"] = torch.cuda.get_device_name(0)
        out["max_memory_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, default=None, help="по умолчанию — синтетическое")
    ap.add_argument("--seconds", type=float, default=20.0, help="длина синтетического видео")
    ap.add_argument("--deadline", type=float, default=240.0)
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--fast-verify", action="store_true", help="размеры без sha256")
    ap.add_argument("--allow-empty-lock", action="store_true", help="lock пуст — всё равно гонять")
    ap.add_argument("--report", type=Path, default=None)
    a = ap.parse_args(argv)

    os.environ.update(OFFLINE_ENV)
    log_dir = Path(os.environ.get("SENTINEL_LOG_DIR") or REPO_ROOT / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    guard_log = log_dir / "netguard_offline_smoke.jsonl"
    guard_log.unlink(missing_ok=True)
    ng = _install_guard(guard_log)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    report: dict = {"check": "offline_worker_smoke", "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "netguard": "block (process)", "env": {k: os.environ.get(k) for k in OFFLINE_ENV}}
    problems: list[str] = []

    if (os.environ.get("SENTINEL_MOCK") or "0") not in ("0", "", "false", "False"):
        print("[smoke] SENTINEL_MOCK включён: mock не доказывает локальность моделей — отказ", file=sys.stderr)
        return 2

    from deploy import weights as W  # noqa: E402 — после установки guard
    weights_dir = a.weights or Path(os.environ.get("SENTINEL_WEIGHTS") or REPO_ROOT / "weights")
    lock = W.load_lock(REPO_ROOT / "deploy" / "weights.lock.json")
    code, results = W.verify(weights_dir, lock, fast=a.fast_verify)
    report["weights"] = {"dir": str(weights_dir), "models": [r["name"] for r in results],
                         "ok": code == 0, "problems": [p for r in results for p in r["problems"]]}
    if code:
        problems.append("weights verify не прошёл")
    if not results and not a.allow_empty_lock:
        problems.append("weights.lock.json пуст — нечего доказывать (weights.py prepare)")

    video = a.video
    if video is None:
        from deploy.synth_video import make_video
        video = make_video(log_dir / "offline_smoke_synth.mov", seconds=a.seconds)
    report["video"] = {"synthetic": a.video is None, "seconds": a.seconds if a.video is None else None}

    if not problems:
        try:
            import backend.pipeline as pipeline  # noqa: E402
            t0 = time.perf_counter()
            pipeline.warmup()
            t1 = time.perf_counter()
            stages: list[tuple[str, float]] = []
            result = pipeline.run(str(video), "offline-smoke", lambda s, p: stages.append((s, round(p, 3))),
                                  deadline_s=a.deadline)
            t2 = time.perf_counter()
            report["timings_s"] = {"warmup": round(t1 - t0, 2), "run": round(t2 - t1, 2)}
            if a.video is None and a.seconds:
                report["timings_s"]["run_per_video_second"] = round((t2 - t1) / a.seconds, 3)
            report["stages"] = [s for s, _ in stages]
            report["result"] = summarize(result)
            problems += check_result(result)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"pipeline: {type(exc).__name__}: {exc}")
            report["traceback"] = traceback.format_exc(limit=8)
    report["gpu"] = gpu_evidence()
    attempts = ng.status()["attempts"]
    report["network_attempts"] = attempts
    if attempts:
        problems.append(f"{attempts} попыток выхода в сеть (см. {guard_log})")
    report["ok"] = not problems
    report["problems"] = problems
    out = a.report or log_dir / "offline_worker_smoke.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report.get(k) for k in ("ok", "problems", "timings_s", "network_attempts", "gpu")},
                     ensure_ascii=False, indent=2))
    print(f"[smoke] отчёт: {out}", file=sys.stderr)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
