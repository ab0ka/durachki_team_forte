"""estimate(clip, tracks): top-down поза COCO-17 по bbox-ам person-треков (батчи кропов, бюджет, сглаживание)."""
from __future__ import annotations

import bisect
import logging
import math
import time
from typing import Any, Optional, Sequence

import numpy as np

from ._contract import PoseFrameX, Track
from .backends import get_backend
from .config import is_mock_mode, load_config
from .filters import smooth_keypoint_track
from .status import set_status

log = logging.getLogger("ml.pose")


class BoxLookup:
    """bbox трека в момент t: ближайшее наблюдение в ±tol или линейная интерполяция (разрыв ≤ max_gap)."""

    def __init__(self, track: Track, tol: float, max_gap: float) -> None:
        items = sorted(((float(t), np.asarray(b, dtype=np.float64)) for t, b in track.boxes), key=lambda x: x[0])
        self.ts = [t for t, _ in items]
        self.bb = [b for _, b in items]
        self.tol = tol
        self.max_gap = max_gap

    def at(self, t: float) -> Optional[np.ndarray]:
        if not self.ts:
            return None
        i = bisect.bisect_left(self.ts, t)
        best, best_dt = None, self.tol + 1e-9
        for k in (i - 1, i):
            if 0 <= k < len(self.ts) and abs(self.ts[k] - t) <= best_dt:
                best, best_dt = self.bb[k], abs(self.ts[k] - t)
        if best is not None:
            return best
        if 0 < i < len(self.ts) and self.ts[i] - self.ts[i - 1] <= self.max_gap:
            a = (t - self.ts[i - 1]) / (self.ts[i] - self.ts[i - 1])
            return (1 - a) * self.bb[i - 1] + a * self.bb[i]
        return None


def _select(lookups: dict[int, BoxLookup], tracks: dict[int, Track], t: float, sel: dict[str, Any]) -> list[tuple[int, np.ndarray]]:
    pri_map = sel.get("role_priority", {}) or {}
    cands = []
    for tid, lk in lookups.items():
        b = lk.at(t)
        if b is None:
            continue
        b = np.clip(b, 0.0, 1.0)
        area = float((b[2] - b[0]) * (b[3] - b[1]))
        if area < float(sel.get("min_box_area", 0.002)):
            continue
        tr = tracks[tid]
        pri = float(pri_map.get(tr.role, 1.0)) + float(tr.role_conf or 0.0) + 0.5 * math.sqrt(area)
        cands.append((pri, tid, b))
    cands.sort(key=lambda x: (-x[0], x[1]))
    return [(tid, b) for _, tid, b in cands[: int(sel.get("max_persons_per_frame", 3))]]


def estimate_impl(clip: Any, tracks: Sequence[Track], cfg: Optional[dict[str, Any]] = None) -> list[PoseFrameX]:
    cfg = cfg or load_config()
    sel = cfg.get("selection", {}) or {}
    t_start = time.perf_counter()
    persons = {int(t.track_id): t for t in tracks or [] if t.label == "person" and t.boxes}
    times = [float(t) for t in (getattr(clip, "times", None) or [])]
    if not persons or not times:
        set_status(state="no_persons", n_crops=0, elapsed_s=0.0)
        return []
    backend, errors = get_backend(cfg, mock=is_mock_mode())
    if backend is None:
        msg = "pose backend unavailable: " + "; ".join(errors)
        log.warning("%s — позы пропущены, сигналы позы будут unknown", msg)
        set_status(state="failed", error=msg, warnings=[msg], n_crops=0, elapsed_s=time.perf_counter() - t_start)
        return []
    tol = float(sel.get("box_time_tolerance_s", 0.25))
    lookups = {tid: BoxLookup(tr, tol, float(sel.get("max_interp_gap_s", 1.0))) for tid, tr in persons.items()}
    budget = float(sel.get("budget_s", 10.0))
    probe = max(1, int(sel.get("budget_probe_frames", 20)))
    max_stride = max(1, int(sel.get("max_stride", 5)))
    max_total = int(sel.get("max_crops_total", 2500))
    flush_n = max(8, int(getattr(backend, "batch_size", 16)) * 2)
    warnings: list[str] = []
    results: dict[int, list[tuple[float, np.ndarray, np.ndarray, tuple[int, int]]]] = {}
    pending: list[tuple[np.ndarray, np.ndarray, float, int, np.ndarray, tuple[int, int]]] = []
    state = "mock" if backend.name == "mock" else "ok"
    error: Optional[str] = None
    n_crops = 0
    n_frames = 0
    stride = 1

    def flush() -> bool:
        nonlocal error, state
        if not pending:
            return True
        items = [(p[0], p[1]) for p in pending]
        meta = [(p[2], p[3]) for p in pending]
        try:
            outs = backend.infer(items, meta)
        except Exception as exc:  # noqa: BLE001 — не роняем пайплайн, но и не скрываем
            error = f"{backend.name}.infer: {type(exc).__name__}: {exc}"
            log.warning("pose inference failed — %s", error)
            state = "failed" if not results else "partial"
            pending.clear()
            return False
        for (frame, box_px, t, tid, bbox_n, wh), (kp, sc) in zip(pending, outs):
            w, h = wh
            kn = np.zeros((17, 3), dtype=np.float32)
            kn[:, 0] = kp[:, 0] / w
            kn[:, 1] = kp[:, 1] / h
            kn[:, 2] = sc
            outside = (kn[:, 0] < -0.02) | (kn[:, 0] > 1.02) | (kn[:, 1] < -0.02) | (kn[:, 1] > 1.02)
            kn[outside, 2] *= 0.5
            kn[:, :2] = np.clip(kn[:, :2], 0.0, 1.0)
            results.setdefault(tid, []).append((t, kn, bbox_n, wh))
        pending.clear()
        return True

    i = 0
    while i < len(times):
        t = times[i]
        chosen = _select(lookups, persons, t, sel)
        if chosen:
            if n_crops + len(chosen) > max_total:
                warnings.append(f"max_crops_total={max_total} достигнут на t={t:.2f}s")
                break
            try:
                frame = clip.frame(i)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"clip.frame({i}) failed: {type(exc).__name__}: {exc}")
                i += stride
                continue
            h, w = frame.shape[:2]
            for tid, b in chosen:
                box_px = np.array([b[0] * w, b[1] * h, b[2] * w, b[3] * h], dtype=np.float32)
                pending.append((frame, box_px, t, tid, b, (w, h)))
            n_crops += len(chosen)
            n_frames += 1
            if len(pending) >= flush_n and not flush():
                break
            if n_frames % probe == 0 and i + 1 < len(times):
                elapsed = time.perf_counter() - t_start
                per_frame = elapsed / n_frames
                remaining = (len(times) - i - 1) / stride
                if elapsed + remaining * per_frame > budget and stride < max_stride:
                    left = max(budget - elapsed, 1e-3)
                    stride = min(max_stride, max(stride + 1, math.ceil(remaining * per_frame * stride / left)))
                    warnings.append(f"бюджет {budget:.0f}s: шаг по сэмплам → {stride}")
        i += stride
    if error is None:
        flush()
    smooth_cfg = cfg.get("smoothing", {}) or {}
    poses: list[PoseFrameX] = []
    for tid, items in results.items():
        items.sort(key=lambda x: x[0])
        ts = np.array([x[0] for x in items])
        kps = np.stack([x[1] for x in items])
        raw = kps.copy()
        smoothed = bool(smooth_cfg.get("enabled", True)) and len(items) > 1
        if smoothed:
            kps = smooth_keypoint_track(ts, kps, smooth_cfg)
        role = persons[tid].role
        for j, (t, _, bbox_n, wh) in enumerate(items):
            extra: dict[str, Any] = {"bbox": tuple(float(v) for v in bbox_n), "source": backend.name,
                                     "frame_wh": (int(wh[0]), int(wh[1])), "role": role, "smoothed": smoothed,
                                     "schema": "coco17"}
            if smooth_cfg.get("keep_raw"):
                extra["raw_keypoints"] = [tuple(float(v) for v in row) for row in raw[j]]
            poses.append(PoseFrameX(t=float(t), track_id=tid,
                                    keypoints=[(float(x), float(y), float(s)) for x, y, s in kps[j]], extra=extra))
    poses.sort(key=lambda p: (p.t, p.track_id if p.track_id is not None else -1))
    if stride > 1 and state == "ok":
        state = "partial"
    set_status(state=state, backend=backend.name, error=error, warnings=warnings + errors, n_crops=n_crops,
               n_frames=n_frames, stride=stride, elapsed_s=round(time.perf_counter() - t_start, 3),
               providers=list(getattr(backend, "providers", []) or []))
    return poses


def warmup_impl(cfg: Optional[dict[str, Any]] = None) -> bool:
    """Загрузить бэкенд и прогнать пустой кроп (держим модель тёплой). False — бэкенд недоступен."""
    cfg = cfg or load_config()
    backend, errors = get_backend(cfg, mock=is_mock_mode())
    if backend is None:
        set_status(state="failed", error="; ".join(errors), warnings=errors)
        return False
    try:
        backend.infer([(np.zeros((256, 192, 3), dtype=np.uint8), np.array([0, 0, 192, 256], dtype=np.float32))],
                      [(0.0, None)])
    except Exception as exc:  # noqa: BLE001
        set_status(state="failed", backend=backend.name, error=f"warmup: {type(exc).__name__}: {exc}")
        return False
    return True
