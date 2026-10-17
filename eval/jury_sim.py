"""Имитация проверки жюри по публичному URL (Cloudflare Tunnel) — сначала последовательно, потом bounded stress.

Фазы:
1. auth: без ключа / с неверным ключом → 401 (POST analyze и GET jobs);
2. неверный файл (.txt) → 415; неизвестный job → 404 (мягко);
3. sequential: N видео по одному; время каждого job — от приёма файла (job_id получен) до done ≤ лимита;
4. bounded stress: N видео с ограниченным параллелизмом (по умолчанию 5); лимит 300 с считается
   ОТ ПРИЁМА каждого job ВКЛЮЧАЯ ОЧЕРЕДЬ (одновременная загрузка жюри не обязательна, но возможна);
5. большой файл → 413/4xx или job=failed, но не 5xx и не зависание.
Ответы валидируются по схеме ТЗ. В отчёт пишутся только статусы и тайминги (без кадров и описаний).

CLI::

    python -m eval.jury_sim --url https://<tunnel> --key $SENTINEL_API_KEY --videos data/videos --n 6
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from .common import list_videos, load_config, video_id_for_path, write_json
from .schema import check_api_response


@dataclass
class Check:
    name: str
    status: str            # PASS | WARN | FAIL | SKIP
    detail: str = ""
    seconds: Optional[float] = None
    data: dict[str, Any] = field(default_factory=dict)


class JurySim:
    def __init__(self, url: str, key: str, cfg: Optional[dict] = None, session: Any = None) -> None:
        import requests  # лениво

        self.url, self.key = url.rstrip("/"), key
        self.cfg = (cfg or load_config()).get("jury_sim") or {}
        self.s = session or requests.Session()
        self.req_timeout = float(self.cfg.get("request_timeout_s", 120))
        self.poll_s = float(self.cfg.get("poll_s", 2.0))
        self.limit_s = float(self.cfg.get("total_limit_s", 300.0))
        self.checks: list[Check] = []

    # ------------------------------------------------------------------ http
    def _post(self, path: Path, key: Optional[str], ctype: str = "video/quicktime") -> Any:
        headers = {"X-API-Key": key} if key is not None else {}
        with open(path, "rb") as f:
            return self.s.post(f"{self.url}/api/v1/analyze", headers=headers, files={"file": (path.name, f, ctype)},
                               timeout=self.req_timeout)

    def _get(self, job_id: str, key: Optional[str]) -> Any:
        headers = {"X-API-Key": key} if key is not None else {}
        return self.s.get(f"{self.url}/api/v1/jobs/{job_id}", headers=headers, timeout=self.req_timeout)

    def add(self, *a: Any, **kw: Any) -> Check:
        c = Check(*a, **kw)
        self.checks.append(c)
        return c

    # ------------------------------------------------------------------ фазы
    def check_auth(self, sample: Path) -> None:
        for name, key in (("auth: POST без ключа → 401", None), ("auth: POST с неверным ключом → 401", "wrong-key-xyz")):
            try:
                r = self._post(sample, key)
                self.add(name, "PASS" if r.status_code == 401 else "FAIL", f"HTTP {r.status_code}")
            except Exception as e:  # noqa: BLE001
                self.add(name, "FAIL", repr(e)[:200])
        try:
            r = self._get("nonexistent-job", None)
            self.add("auth: GET jobs без ключа → 401", "PASS" if r.status_code == 401 else "FAIL", f"HTTP {r.status_code}")
            r = self._get("nonexistent-job-000", self.key)
            self.add("GET неизвестного job → 404", "PASS" if r.status_code == 404 else "WARN", f"HTTP {r.status_code}")
        except Exception as e:  # noqa: BLE001
            self.add("auth: GET jobs", "FAIL", repr(e)[:200])

    def check_wrong_file(self, tmp: Path) -> None:
        p = tmp / "not_a_video.txt"
        p.write_text("это не видео", encoding="utf-8")
        try:
            r = self._post(p, self.key, "text/plain")
            st = "PASS" if r.status_code == 415 else ("WARN" if r.status_code in (400, 422) else "FAIL")
            self.add("неверный файл (.txt) → 415", st, f"HTTP {r.status_code}")
        except Exception as e:  # noqa: BLE001
            self.add("неверный файл (.txt) → 415", "FAIL", repr(e)[:200])

    def check_big_file(self, tmp: Path, mb: int) -> None:
        if mb <= 0:
            self.add("большой файл", "SKIP", "big_file_mb=0")
            return
        p = tmp / "big_random.mov"
        with open(p, "wb") as f:
            for _ in range(mb):
                f.write(os.urandom(1 << 20))
        t0 = time.perf_counter()
        name = f"большой файл ({mb} МБ мусора) → 413/4xx или failed, не 5xx"
        try:
            r = self._post(p, self.key)
            if r.status_code in (413, 400, 415, 422):
                self.add(name, "PASS", f"HTTP {r.status_code}", time.perf_counter() - t0)
            elif r.status_code < 300:
                job = r.json().get("job_id")
                st = self._wait(job, time.perf_counter())
                ok = st.get("status") == "failed"
                self.add(name, "PASS" if ok else "WARN", f"принят, job → {st.get('status')}", time.perf_counter() - t0)
            else:
                self.add(name, "FAIL", f"HTTP {r.status_code}", time.perf_counter() - t0)
        except Exception as e:  # noqa: BLE001
            self.add(name, "FAIL", repr(e)[:200], time.perf_counter() - t0)
        finally:
            p.unlink(missing_ok=True)

    def _wait(self, job_id: str, t_accept: float) -> dict:
        last: dict = {}
        while time.perf_counter() - t_accept < self.limit_s + 60:
            try:
                r = self._get(job_id, self.key)
                if r.ok:
                    last = r.json()
            except Exception:  # noqa: BLE001
                pass
            if last.get("status") in ("done", "failed"):
                break
            time.sleep(self.poll_s)
        last["_elapsed_from_accept_s"] = round(time.perf_counter() - t_accept, 2)
        return last

    def _job(self, p: Path) -> dict:
        rec: dict[str, Any] = {"video": video_id_for_path(p)}
        t0 = time.perf_counter()
        try:
            r = self._post(p, self.key)
            rec["upload_http"] = r.status_code
            if r.status_code >= 300:
                rec["error"] = f"HTTP {r.status_code}"
                return rec
            job_id = r.json()["job_id"]
        except Exception as e:  # noqa: BLE001
            rec["error"] = repr(e)[:200]
            return rec
        t_acc = time.perf_counter()
        rec["upload_s"] = round(t_acc - t0, 2)
        res = self._wait(job_id, t_acc)
        rec["status"] = res.get("status")
        rec["verdict"] = res.get("verdict")
        rec["n_violations"] = len(res.get("violations") or [])
        rec["accept_to_done_s"] = res.get("_elapsed_from_accept_s")
        res.pop("_elapsed_from_accept_s", None)
        rep = check_api_response(res)
        rec["schema_errors"] = rep.errors[:5]
        return rec

    def _judge(self, name: str, recs: list[dict], wall: float) -> None:
        bad_schema = [r["video"] for r in recs if r.get("schema_errors") or r.get("error")]
        slow = [r["video"] for r in recs if (r.get("accept_to_done_s") or 1e9) > self.limit_s]
        not_done = [r["video"] for r in recs if r.get("status") != "done"]
        mx = max((r.get("accept_to_done_s") or 0 for r in recs), default=0)
        st = "PASS" if not (bad_schema or slow or not_done) else "FAIL"
        self.add(name, st, f"{len(recs)} видео; max от приёма до done {mx:.1f} с (лимит {self.limit_s:g}); "
                           f"не done: {not_done or '—'}; > лимита: {slow or '—'}; ошибки схемы: {bad_schema or '—'}",
                 wall, {"jobs": recs})

    def sequential(self, videos: list[Path]) -> None:
        t0 = time.perf_counter()
        recs = [self._job(p) for p in videos]
        self._judge("sequential: по одному видео, схема ТЗ + ≤ лимита", recs, time.perf_counter() - t0)

    def stress(self, videos: list[Path], concurrency: int) -> None:
        t0 = time.perf_counter()
        with cf.ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
            recs = list(ex.map(self._job, videos))
        self._judge(f"bounded stress: {len(videos)} видео, параллельно ≤ {concurrency}; лимит от приёма вкл. очередь",
                    recs, time.perf_counter() - t0)

    def summary(self) -> dict:
        counted = [c for c in self.checks if c.status != "SKIP"]
        return {"n_checks": len(counted), "n_pass": sum(c.status == "PASS" for c in counted),
                "n_warn": sum(c.status == "WARN" for c in counted), "n_fail": sum(c.status == "FAIL" for c in counted),
                "pass_rate": round(sum(c.status == "PASS" for c in counted) / len(counted), 3) if counted else None,
                "limit_s": self.limit_s, "url_host": self.url.split("//")[-1].split("/")[0]}


def run_jury_sim(url: str, key: str, videos: list[Path], cfg: Optional[dict] = None, concurrency: int = 5,
                 skip_stress: bool = False, big_mb: Optional[int] = None, session: Any = None) -> dict:
    sim = JurySim(url, key, cfg, session)
    tmp = Path(tempfile.mkdtemp(prefix="jury_sim_"))
    if videos:
        sim.check_auth(videos[0])
    sim.check_wrong_file(tmp)
    if videos:
        sim.sequential(videos)
        if not skip_stress:
            sim.stress(videos, concurrency)
    sim.check_big_file(tmp, int(sim.cfg.get("big_file_mb", 150) if big_mb is None else big_mb))
    return {"summary": sim.summary(), "checks": [asdict(c) for c in sim.checks]}


def format_table(res: dict) -> str:
    lines = [f"{'статус':6}  {'сек':>7}  проверка — детали"]
    for c in res["checks"]:
        sec = f"{c['seconds']:.1f}" if c.get("seconds") is not None else "—"
        lines.append(f"{c['status']:6}  {sec:>7}  {c['name']} — {c['detail']}")
    s = res["summary"]
    lines.append(f"ИТОГО: PASS {s['n_pass']} / WARN {s['n_warn']} / FAIL {s['n_fail']} (pass_rate={s['pass_rate']})")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Имитация жюри по публичному URL")
    ap.add_argument("--url", required=True)
    ap.add_argument("--key", default=os.environ.get("SENTINEL_API_KEY", ""))
    ap.add_argument("--videos", default="data/videos")
    ap.add_argument("--n", type=int, default=None, help="сколько видео (по умолчанию config jury_sim.n_videos)")
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--skip-stress", action="store_true")
    ap.add_argument("--big-mb", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/jury_sim.json")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    n = args.n or int((cfg.get("jury_sim") or {}).get("n_videos", 6))
    vids = list_videos(args.videos) if Path(args.videos).exists() else []
    random.Random(args.seed).shuffle(vids)
    vids = sorted(vids[:n], key=lambda p: p.stat().st_size, reverse=True)  # самые длинные — первыми
    res = run_jury_sim(args.url, args.key, vids, cfg, args.concurrency, args.skip_stress, args.big_mb)
    write_json(args.out, res)
    print(format_table(res))
    print(f"→ {args.out}")
    return 1 if res["summary"]["n_fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
