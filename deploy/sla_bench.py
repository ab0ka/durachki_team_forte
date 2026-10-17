#!/usr/bin/env python3
"""ЗАМЕР SLA: K видео одновременно (как жюри), время от отправки до результата у КАЖДОГО.

Ничего не объявляем — меряем. Цель ТЗ: всё ≤ 300 с при 5+ видео; ориентир ≤ 60 с на 145-секундное.
Реальные ролики остаются на машине команды (--videos); без них — синтетические (--synth N --seconds S).
Во время замера ноутбук не должен засыпать: запускайте при работающем start.ps1 (он держит систему
бодрствующей; гаснущий дисплей не мешает).

  python deploy/sla_bench.py --url http://127.0.0.1:8000 --synth 5 --seconds 145
  python deploy/sla_bench.py --url https://xxx.trycloudflare.com --videos D:\\data\\a.mov D:\\data\\b.mov
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

try:
    from deploy import httpkit
except ImportError:
    import httpkit  # type: ignore


def run_bench(url: str, api_key: str, videos: list[Path], limit_s: float) -> dict:
    jobs: list[dict] = [{"video": v.name, "size_mb": round(v.stat().st_size / 2**20, 1)} for v in videos]
    t0 = time.perf_counter()

    def one(i: int, v: Path) -> None:
        j = jobs[i]
        sub = httpkit.submit(url, v, api_key, timeout=limit_s)
        j["upload_s"] = round(sub.elapsed_s, 2)
        job_id = (sub.json() or {}).get("job_id") if sub.status == 200 else None
        if not job_id:
            j.update(status=f"submit {sub.status or sub.error}", total_s=None)
            return
        doc, _ = httpkit.poll(url, job_id, api_key, limit_s=max(1.0, limit_s - (time.perf_counter() - t0)))
        j["status"] = (doc or {}).get("status")
        j["verdict"] = (doc or {}).get("verdict")
        j["n_violations"] = len((doc or {}).get("violations") or [])
        j["total_s"] = round(time.perf_counter() - t0, 2)  # от общего старта: так считает жюри

    threads = [threading.Thread(target=one, args=(i, v), daemon=True) for i, v in enumerate(videos)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(limit_s + 120)
    done = [j["total_s"] for j in jobs if j.get("status") == "done" and j.get("total_s") is not None]
    rep = {"url": url, "n": len(videos), "limit_s": limit_s, "jobs": jobs,
           "all_done": len(done) == len(videos),
           "max_s": max(done) if done else None,
           "median_s": round(statistics.median(done), 2) if done else None,
           "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    rep["ok"] = rep["all_done"] and rep["max_s"] is not None and rep["max_s"] <= limit_s
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=f"http://127.0.0.1:{os.environ.get('SENTINEL_PORT', '8000')}")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--videos", nargs="*", type=Path, default=[])
    ap.add_argument("--synth", type=int, default=0, help="сколько синтетических видео")
    ap.add_argument("--seconds", type=float, default=145.0)
    ap.add_argument("--limit", type=float, default=300.0)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    videos = list(a.videos)
    if a.synth:
        try:
            from deploy.synth_video import make_video
        except ImportError:
            from synth_video import make_video  # type: ignore
        cache = Path(os.environ.get("SENTINEL_CACHE_DIR") or Path(__file__).resolve().parents[1] / "cache")
        base = make_video(cache / f"sla_synth_{int(a.seconds)}s.mov", seconds=a.seconds)
        videos += [base] * a.synth  # одинаковый файл: учтите кэш backend по md5 (может завысить скорость)
    if not videos:
        ap.error("нужны --videos или --synth")
    rep = run_bench(a.url, a.api_key or httpkit.api_key_from_env(), videos, a.limit)
    if a.synth:
        rep["note"] = "синтетика одним файлом; если backend кэширует по md5 — меряйте на разных реальных роликах"
    text = json.dumps(rep, ensure_ascii=False, indent=2)
    out = a.out or Path(os.environ.get("SENTINEL_LOG_DIR") or Path(__file__).resolve().parents[1] / "logs") / "sla_bench.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
