#!/usr/bin/env python3
"""Проверки сервиса по HTTP (stdlib, без зависимостей).

  live  — процесс жив: GET /healthz отвечает кодом < 500 (для Docker HEALTHCHECK / супервизора).
  ready — сервис ГОТОВ, только если прошёл РЕАЛЬНЫЙ пробный inference: без ключа → 401,
          POST синтетического видео → job_id → done, ответ по схеме ТЗ, это не заглушка.
          start.ps1 открывает туннель только после успешного ready.

  python deploy/healthcheck.py live --url http://127.0.0.1:8000
  python deploy/healthcheck.py ready --url http://127.0.0.1:8000 --api-key KEY [--video synth.mov] [--json]
Ключ: --api-key или SENTINEL_API_KEY / первый из SENTINEL_API_KEYS (не храните ключи в репо).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    from deploy import httpkit
except ImportError:  # запуск как `python deploy/healthcheck.py`
    import httpkit  # type: ignore

DEFAULT_URL = os.environ.get("SENTINEL_HEALTH_URL") or f"http://127.0.0.1:{os.environ.get('SENTINEL_PORT', '8000')}"


def live(url: str) -> dict:
    r = httpkit.request("GET", url.rstrip("/") + "/healthz", timeout=5)
    return {"check": "live", "ok": 0 < r.status < 500, "status": r.status, "elapsed_s": round(r.elapsed_s, 3),
            "error": r.error}


def ready(url: str, api_key: str, video: Path, limit_s: float = 300.0, allow_stub: bool = False) -> dict:
    rep: dict = {"check": "ready", "url": url, "problems": []}
    p = rep["problems"]
    if not api_key:
        p.append("нет API-ключа (--api-key / SENTINEL_API_KEYS)")
        rep["ok"] = False
        return rep
    lv = live(url)
    rep["live"] = lv
    if not lv["ok"]:
        p.append(f"сервер не отвечает: {lv['error'] or lv['status']}")
        rep["ok"] = False
        return rep
    rz = httpkit.request("GET", url.rstrip("/") + "/readyz", timeout=10)
    rzb = rz.json() if rz.status else None
    rep["readyz"] = {"status": rz.status, "state": (rzb or {}).get("state") if isinstance(rzb, dict) else None,
                     "pipeline": (rzb or {}).get("pipeline") if isinstance(rzb, dict) else None}
    if rz.status == 503:
        p.append(f"/readyz 503 (state={rep['readyz']['state']}): warmup A02 не завершён или нет весов")
        rep["ok"] = False
        return rep
    if rep["readyz"]["pipeline"] == "fake" and not allow_stub:
        p.append("/readyz: pipeline=fake — фиктивный пайплайн в проде запрещён")
    body, ctype = httpkit.multipart("file", video.name, video.read_bytes())
    noauth = httpkit.request("POST", url.rstrip("/") + "/api/v1/analyze", {"Content-Type": ctype}, body, timeout=120)
    rep["no_key_status"] = noauth.status
    if noauth.status != 401:
        p.append(f"без X-API-Key ожидали 401, получили {noauth.status or noauth.error}")
    t0 = time.perf_counter()
    sub = httpkit.submit(url, video, api_key, timeout=300)
    rep["upload_s"] = round(sub.elapsed_s, 2)
    if sub.headers.get("x-sentinel-stub") and not allow_stub:
        p.append("отвечает ЗАГЛУШКА deploy/stub_api.py, а не реальный backend")
    job_id = (sub.json() or {}).get("job_id") if sub.status == 200 else None
    if not job_id:
        p.append(f"POST /api/v1/analyze: {sub.status or sub.error} {sub.body[:200]!r}")
        rep["ok"] = False
        return rep
    doc, waited = httpkit.poll(url, job_id, api_key, limit_s=limit_s)
    rep["time_to_result_s"] = round(time.perf_counter() - t0, 2)
    rep["status"] = (doc or {}).get("status")
    rep["verdict"] = (doc or {}).get("verdict")
    rep["n_violations"] = len((doc or {}).get("violations") or [])
    if doc is None or doc.get("status") != "done":
        p.append(f"задача не завершилась успешно за {limit_s:.0f} с: status={rep['status']!r}")
    else:
        p += httpkit.validate_job(doc)
        meta = doc.get("meta") if isinstance(doc.get("meta"), dict) else {}
        if (meta.get("fake_pipeline") or meta.get("mock")) and not allow_stub:
            p.append("meta.fake_pipeline/mock=true — результат не от реальных моделей")
        if doc.get("verdict") == "no_violation" and meta.get("complete") is False:
            p.append("no_violation при meta.complete=false — неполный анализ ≠ «нарушений нет»")
        rep["complete"] = meta.get("complete")
    rep["ok"] = not p
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("live", "ready"))
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--video", type=Path, default=None, help="по умолчанию — синтетическое 10 с")
    ap.add_argument("--limit", type=float, default=300.0, help="сек на пробный inference")
    ap.add_argument("--allow-stub", action="store_true", help="только для тестов deploy/")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    if a.mode == "live":
        rep = live(a.url)
    else:
        video = a.video
        if video is None:
            try:
                from deploy.synth_video import make_video
            except ImportError:
                from synth_video import make_video  # type: ignore
            cache = Path(os.environ.get("SENTINEL_CACHE_DIR") or Path(__file__).resolve().parents[1] / "cache")
            video = cache / "healthcheck_synth.mov"
            if not video.is_file():
                make_video(video, seconds=10)
        rep = ready(a.url, a.api_key or httpkit.api_key_from_env(), video, a.limit, a.allow_stub)
    if not a.quiet:
        print(json.dumps(rep, ensure_ascii=False, indent=2) if a.json else
              f"[{rep['check']}] {'OK' if rep['ok'] else 'FAIL'} " + json.dumps(
                  {k: v for k, v in rep.items() if k not in ("check", "ok")}, ensure_ascii=False))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
