"""Очередь задач: FIFO, воркеры-потоки, GPU-семафор, бюджет времени, watchdog, рестарт.

Честность результата: done — только когда pipeline.run вернул результат. Нет пайплайна,
исключение, таймаут -> failed (partial.json сохраняется как подробности для UI)."""
from __future__ import annotations

import base64
import importlib
import json
import logging
import math
import os
import shutil
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from .config import Settings
from .results import fill_missing_evidence, iter_frames, normalize, to_api_dict
from .store import JobStore

log = logging.getLogger("sentinel.api.worker")

ProgressFn = Callable[[str, float], None]
RunFn = Callable[[str, str, ProgressFn, float], Any]


class JobCancelled(BaseException):
    """Бросается из progress(), если watchdog уже закрыл задачу: run() должен свернуться.
    BaseException (как asyncio.CancelledError), чтобы не глоталось `except Exception`."""


@dataclass
class PipelineSpec:
    run: Optional[RunFn] = None
    warmup: Optional[Callable[[], None]] = None
    mode: str = "real"            # real | fake | unavailable
    error: str = ""
    manages_gpu: bool = False     # pipeline сам держит GPU-лок -> API не оборачивает run()


def resolve_pipeline(settings: Settings) -> PipelineSpec:
    """Ленивый импорт backend.pipeline. Фейк — только при явном SENTINEL_FAKE_PIPELINE=1."""
    if settings.fake_pipeline:
        from .fake_pipeline import make_fake_runner
        return PipelineSpec(run=make_fake_runner(settings.fake_pipeline_s), mode="fake")
    mod_name, _, fn = settings.pipeline.partition(":")
    try:
        mod = importlib.import_module(mod_name)
        run = getattr(mod, fn or "run")
        if not callable(run):
            raise TypeError(f"{settings.pipeline} is not callable")
    except Exception as e:
        return PipelineSpec(mode="unavailable", error=f"{type(e).__name__}: {e}"[:400])
    warm = getattr(mod, "warmup", None) if not fn else None
    manages = any(hasattr(mod, a) for a in ("GPU_LOCK", "GPU_SEMAPHORE", "gpu_lock"))
    return PipelineSpec(run=run, warmup=warm if callable(warm) else None, mode="real",
                        manages_gpu=manages)


class GpuGate:
    """Семафор с владельцами: watchdog может освободить слот зависшей задачи."""

    def __init__(self, slots: int):
        self.slots = max(1, int(slots))
        self._cv = threading.Condition()
        self._holders: set[str] = set()

    def acquire(self, owner: str, timeout: float) -> bool:
        end = time.monotonic() + max(0.0, timeout)
        with self._cv:
            while len(self._holders) >= self.slots:
                rem = end - time.monotonic()
                if rem <= 0:
                    return False
                self._cv.wait(rem)
            self._holders.add(owner)
            return True

    def release(self, owner: str) -> None:
        with self._cv:
            if owner in self._holders:
                self._holders.discard(owner)
                self._cv.notify_all()

    @property
    def busy(self) -> int:
        with self._cv:
            return len(self._holders)


def write_json(path: Path, data: Any) -> None:
    tmp = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    os.replace(tmp, path)


class JobManager:
    def __init__(self, settings: Settings, store: JobStore,
                 pipeline: Optional[PipelineSpec | Callable[[], PipelineSpec]] = None):
        self.s = settings
        self.store = store
        self._pipeline_src = pipeline
        self.pipeline: Optional[PipelineSpec] = None
        self.state = "loading"            # loading | warming | ready | unavailable | warmup_failed
        self.state_detail = ""
        self._resolved = threading.Event()
        self._warm = threading.Event()
        self._stop = threading.Event()
        self._cv = threading.Condition()
        self._q: deque[str] = deque()
        self._queued_at: dict[str, float] = {}
        self._lock = threading.RLock()
        self._live: dict[str, dict[str, Any]] = {}
        self._running: dict[str, dict[str, Any]] = {}
        self._workers: set[threading.Thread] = set()
        self._zombies: set[threading.Thread] = set()
        self._threads: list[threading.Thread] = []
        self._seq = 0
        self.gpu = GpuGate(settings.gpu_slots)
        self.completed = 0

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self.s.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._recover()
        self._spawn(self._init_pipeline, "sentinel-warmup")
        for _ in range(self.s.workers):
            self._spawn_worker()
        self._spawn(self._watchdog_loop, "sentinel-watchdog")
        if self.s.job_ttl_h > 0:
            self._spawn(self._cleanup_loop, "sentinel-cleanup")

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        end = time.monotonic() + timeout
        for t in list(self._threads):
            t.join(max(0.0, end - time.monotonic()))

    def _spawn(self, target: Callable[[], None], name: str) -> threading.Thread:
        t = threading.Thread(target=target, name=name, daemon=True)
        self._threads.append(t)
        t.start()
        return t

    def _spawn_worker(self) -> None:
        with self._lock:
            self._seq += 1
            name = f"sentinel-worker-{self._seq}"
        t = threading.Thread(target=self._worker_loop, name=name, daemon=True)
        with self._lock:
            self._workers.add(t)
        self._threads.append(t)
        t.start()

    # ------------------------------------------------------------ pipeline/warmup
    def _init_pipeline(self) -> None:
        src = self._pipeline_src
        try:
            spec = src() if callable(src) else (src or resolve_pipeline(self.s))
        except Exception as e:
            spec = PipelineSpec(mode="unavailable", error=f"{type(e).__name__}: {e}"[:400])
        self.pipeline = spec
        self._resolved.set()
        if spec.run is None:
            self.state, self.state_detail = "unavailable", spec.error
            log.error("analysis pipeline unavailable: jobs will fail", extra={"error": spec.error})
            self._warm.set()
            return
        if spec.mode == "fake":
            log.warning("FAKE PIPELINE ENABLED (SENTINEL_FAKE_PIPELINE=1): results are synthetic")
        if spec.warmup and self.s.warmup:
            self.state = "warming"
            t0 = time.monotonic()
            try:
                spec.warmup()
                self.state = "ready"
                log.info("pipeline warmed up", extra={"seconds": round(time.monotonic() - t0, 2)})
            except Exception as e:
                self.state, self.state_detail = "warmup_failed", f"{type(e).__name__}: {e}"[:400]
                log.error("pipeline warmup failed", extra={"error": self.state_detail})
        else:
            self.state = "ready"
        self._warm.set()

    @property
    def ready(self) -> bool:
        return self.state == "ready"

    # ------------------------------------------------------------ queue
    def submit(self, job_id: str, queued_at: Optional[float] = None) -> None:
        with self._cv:
            self._q.append(job_id)
            self._queued_at[job_id] = time.time() if queued_at is None else queued_at
            self._cv.notify()

    def _take(self, timeout: float) -> Optional[str]:
        with self._cv:
            if not self._q:
                self._cv.wait(timeout)
            if not self._q:
                return None
            return self._q.popleft()

    def queue_position(self, job_id: str) -> Optional[int]:
        with self._cv:
            try:
                return list(self._q).index(job_id) + 1
            except ValueError:
                return None

    def queue_len(self) -> int:
        with self._cv:
            return len(self._q)

    def live(self, job_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            v = self._live.get(job_id)
            return dict(v) if v else None

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {"queued": self.queue_len(), "running": len(self._running),
                    "workers": len(self._workers - self._zombies), "zombies": len(self._zombies),
                    "gpu_busy": self.gpu.busy, "gpu_slots": self.gpu.slots,
                    "completed": self.completed}

    # ------------------------------------------------------------ worker
    def _worker_loop(self) -> None:
        me = threading.current_thread()
        try:
            while not self._stop.is_set():
                job_id = self._take(0.5)
                if job_id is None:
                    continue
                try:
                    self._process(job_id)
                except Exception:
                    log.exception("worker crashed on job", extra={"job_id": job_id})
                with self._lock:
                    if me in self._zombies:
                        self._zombies.discard(me)
                        if len(self._workers - self._zombies) > self.s.workers:
                            return
        finally:
            with self._lock:
                self._workers.discard(me)
                self._zombies.discard(me)

    def job_dir(self, job_id: str) -> Path:
        return self.s.jobs_dir / job_id

    def _is_final(self, job_id: str) -> bool:
        row = self.store.get(job_id)
        return not row or row["status"] in ("done", "failed")

    def _set_live(self, job_id: str, stage: str, progress: float) -> None:
        with self._lock:
            self._live[job_id] = {"stage": stage, "progress": progress, "updated": time.time()}

    def _process(self, job_id: str) -> None:
        with self._cv:
            t0 = self._queued_at.pop(job_id, None)
        row = self.store.get(job_id)
        if not row or row["status"] != "queued":
            return
        t0 = row["queued_at"] if t0 is None else t0
        if not self.store.mark_running(job_id):
            return
        self._set_live(job_id, "starting", 0.0)

        if not self._warm.is_set():
            self._set_live(job_id, "waiting_warmup", 0.0)
            self.store.update(job_id, stage="waiting_warmup")
            self._warm.wait(max(0.0, min(self.s.warmup_wait_s, t0 + self.s.watchdog_s - time.time())))
        spec = self.pipeline
        if spec is None or spec.run is None:
            err = spec.error if spec else "pipeline not initialised (warmup timeout)"
            self._fail(job_id, f"analysis pipeline unavailable: {err}", "pipeline_unavailable")
            return

        gated = self.s.gpu_lock_scope == "job" and not spec.manages_gpu
        if gated:
            self._set_live(job_id, "waiting_gpu", 0.0)
            while not self.gpu.acquire(job_id, timeout=0.5):
                if self._stop.is_set() or self._is_final(job_id):
                    return
        try:
            if self._is_final(job_id):
                return
            deadline = self._deadline_for(t0)
            started = time.time()
            self.store.update(job_id, deadline_s=deadline, stage="running")
            self._set_live(job_id, "running", 0.0)
            with self._lock:
                self._running[job_id] = {"thread": threading.current_thread(),
                                         "started": started, "deadline": deadline}
            video = row["video_path"]
            try:
                result = spec.run(video, job_id, self._progress_cb(job_id), deadline)
            except JobCancelled:
                return
            except Exception as e:
                log.error("pipeline.run raised", extra={"job_id": job_id, "error": repr(e)[:300],
                                                         "trace": traceback.format_exc(limit=8)})
                self._fail(job_id, f"analysis error: {type(e).__name__}: {e}"[:500], "pipeline_error")
                return
            self._complete(job_id, result, spec, t0=t0, started=started, deadline=deadline)
        finally:
            if gated:
                self.gpu.release(job_id)
            with self._lock:
                self._running.pop(job_id, None)
                self._live.pop(job_id, None)

    def _deadline_for(self, t0: float) -> float:
        """deadline_s = budget − ожидание (очередь включена), не больше job_cap_s и честной доли
        для тех, кто ждёт за нами: иначе 6-е видео из пачки жюри не уложится в 5 минут."""
        s, now = self.s, time.time()
        d = min(s.budget_s - (now - t0), s.job_cap_s)
        if s.fair_share:
            par = min(s.workers, s.gpu_slots) if s.gpu_lock_scope == "job" else s.workers
            par = max(1, par)
            with self._cv:
                waiting = [self._queued_at.get(j, now) for j in self._q]
            for k, qt in enumerate(waiting, start=1):
                d = min(d, (s.budget_s - (now - qt)) / (math.ceil(k / par) + 1))
        return round(max(s.min_deadline_s, d), 1)

    def _progress_cb(self, job_id: str) -> ProgressFn:
        last = {"t": 0.0, "stage": None}

        def progress(stage: str, fraction: float = 0.0) -> None:
            with self._lock:
                cancelled = job_id not in self._running
            if cancelled and self._is_final(job_id):
                raise JobCancelled(job_id)
            try:
                f = float(fraction)
                f = 0.0 if not math.isfinite(f) else min(1.0, max(0.0, f))
            except (TypeError, ValueError):
                f = 0.0
            st = str(stage)[:64]
            self._set_live(job_id, st, f)
            now = time.monotonic()
            if st != last["stage"] or now - last["t"] >= 1.0:
                last.update(t=now, stage=st)
                try:
                    self.store.update(job_id, stage=st, progress=f)
                except Exception:
                    pass
        return progress

    # ------------------------------------------------------------ finalize
    def _complete(self, job_id: str, result: Any, spec: PipelineSpec, *, t0: float,
                  started: float, deadline: float) -> None:
        warns: list[str] = []
        try:
            api, w = to_api_dict(result)
            warns += w
        except Exception as e:
            self._fail(job_id, f"pipeline returned unusable result: {type(e).__name__}: {e}"[:500],
                       "bad_result")
            return
        if api.get("status") == "failed":
            meta = api.get("meta") if isinstance(api.get("meta"), dict) else {}
            self._fail(job_id, str(meta.get("error") or "pipeline reported status=failed")[:500],
                       "pipeline_failed")
            return
        api, w = normalize(api, job_id)
        warns += w
        row = self.store.get(job_id) or {}
        if self.s.evidence_fallback and row.get("video_path"):
            warns += fill_missing_evidence(api, Path(row["video_path"]))
        now = time.time()
        meta = api["meta"]
        mw = meta.get("warnings")
        meta["warnings"] = (list(mw) if isinstance(mw, list) else ([str(mw)] if mw else [])) + warns
        if spec.mode == "fake":
            meta["fake_pipeline"] = True
        frames = [{"i": i, "violation": vi, "timestamp_sec": f.get("timestamp_sec"),
                   "path": f"frames/{i}.jpg"} for i, vi, f in iter_frames(api)]
        meta["api"] = {
            "pipeline": spec.mode, "queue_wait_s": round(started - t0, 2),
            "processing_s": round(now - started, 2), "total_s": round(now - t0, 2),
            "deadline_s": deadline, "filename": row.get("filename"),
            "video": row.get("video_meta") or {}, "frames": frames, "api_version": "1",
        }
        jd = self.job_dir(job_id)
        path = jd / f"result-{int(now * 1000)}-{threading.get_ident()}.json"
        try:
            write_json(path, api)
        except (OSError, ValueError) as e:
            self._fail(job_id, f"could not store result: {type(e).__name__}", "store_error")
            return
        if self.store.finalize(job_id, "done", result_path=str(path), warnings=meta["warnings"]):
            with self._lock:
                self.completed += 1
            if spec.mode == "real" and self.state == "warmup_failed":
                self.state = "ready"  # рабочий прогон доказал готовность
            self._dump_frames(jd, api)
            log.info("job done", extra={"job_id": job_id, "verdict": api.get("verdict"),
                                         "violations": len(api.get("violations") or []),
                                         "total_s": meta["api"]["total_s"]})
        else:  # watchdog успел раньше: результат поздний, наружу не меняем
            try:
                os.replace(path, jd / "late_result.json")
            except OSError:
                pass

    def _dump_frames(self, jd: Path, api: dict[str, Any]) -> None:
        try:
            fd = jd / "frames"
            fd.mkdir(exist_ok=True)
            for i, _, f in iter_frames(api):
                (fd / f"{i}.jpg").write_bytes(base64.b64decode(f["image_base64"]))
        except Exception:
            log.warning("could not dump evidence frames", exc_info=True)

    def _read_partial(self, job_id: str) -> Optional[dict[str, Any]]:
        p = self.job_dir(job_id) / "partial.json"
        if not p.is_file():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def _fail(self, job_id: str, error: str, code: str) -> bool:
        """failed — наружу без вердикта. partial.json (если pipeline успел) — подробности для UI."""
        partial = self._read_partial(job_id)
        result_path = None
        if partial is not None:
            details = {"job_id": job_id, "complete": False, "code": code, "partial": partial}
            p = self.job_dir(job_id) / "failed_details.json"
            try:
                write_json(p, details)
                result_path = str(p)
            except (OSError, ValueError):
                result_path = None
        ok = self.store.finalize(job_id, "failed", error=f"{code}: {error}"[:600],
                                 result_path=result_path,
                                 warnings=[f"{code}: {error}"[:300]])
        if ok:
            log.warning("job failed", extra={"job_id": job_id, "code": code, "error": error[:300],
                                              "has_partial": partial is not None})
        return ok

    # ------------------------------------------------------------ watchdog
    def _watchdog_loop(self) -> None:
        while not self._stop.wait(0.25):
            try:
                self._watchdog_tick()
            except Exception:
                log.exception("watchdog tick failed")

    def _watchdog_tick(self, now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        for row in self.store.active():
            jid = row["id"]
            with self._lock:
                info = self._running.get(jid)
            limit = row["queued_at"] + self.s.watchdog_s
            if info:
                limit = min(limit, info["started"] + info["deadline"] + self.s.deadline_grace_s)
            if now < limit:
                continue
            waited = now - row["queued_at"]
            if self._fail(jid, f"analysis did not finish in time ({waited:.0f}s since upload)",
                          "timeout"):
                self._abandon(jid, info)

    def _abandon(self, job_id: str, info: Optional[dict[str, Any]]) -> None:
        with self._cv:
            if job_id in self._q:
                self._q.remove(job_id)
            self._queued_at.pop(job_id, None)
        if not info:
            return
        with self._lock:
            self._running.pop(job_id, None)
            th = info.get("thread")
            if th is not None and th.is_alive():
                self._zombies.add(th)
            spawn = (len(self._workers - self._zombies) < self.s.workers
                     and len(self._zombies) <= self.s.max_zombie_workers)
        self.gpu.release(job_id)
        if spawn:
            log.warning("worker stuck in pipeline.run; spawning replacement",
                        extra={"job_id": job_id})
            self._spawn_worker()

    # ------------------------------------------------------------ restart / cleanup
    def _recover(self) -> None:
        now = time.time()
        for row in self.store.active():
            jid = row["id"]
            if row["attempts"] >= self.s.max_attempts:
                self.store.finalize(jid, "failed", error="crashed: server restarted during "
                                    f"processing {row['attempts']} times")
            elif not row.get("video_path") or not Path(row["video_path"]).is_file():
                self.store.finalize(jid, "failed", error="video file missing after restart")
            else:
                self.store.requeue(jid, now)
                self.submit(jid, now)
                log.info("requeued after restart", extra={"job_id": jid})

    def _cleanup_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.cleanup()
            except Exception:
                log.exception("cleanup failed")
            if self._stop.wait(self.s.cleanup_interval_s):
                return

    def cleanup(self, now: Optional[float] = None) -> int:
        now = time.time() if now is None else now
        ttl = self.s.job_ttl_h * 3600
        if ttl <= 0:
            return 0
        n = 0
        for row in self.store.finished_before(now - ttl):
            shutil.rmtree(self.job_dir(row["id"]), ignore_errors=True)
            self.store.delete(row["id"])
            n += 1
        known = self.store.ids()
        if self.s.jobs_dir.is_dir():
            for d in self.s.jobs_dir.iterdir():
                try:
                    if d.is_dir() and d.name not in known and now - d.stat().st_mtime > max(ttl, 3600):
                        shutil.rmtree(d, ignore_errors=True)
                        n += 1
                except OSError:
                    continue
        if n:
            log.info("cleanup removed jobs", extra={"count": n})
        return n
