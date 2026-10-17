"""Пост-обработка: вероятности по шагам -> риск-кривая 0.5 с -> события из пиков (общая для CV и инференса)."""
from __future__ import annotations

from typing import Optional

import numpy as np


def risk_curve(probs: np.ndarray, grid: np.ndarray, cfg: dict, duration: Optional[float] = None
               ) -> tuple[np.ndarray, np.ndarray]:
    """Сгладить (скользящее среднее smooth_s) и пересэмплировать на шаг risk_step_s (max в корзине)."""
    inf = cfg["infer"]
    probs = np.asarray(probs, np.float32)
    grid = np.asarray(grid, np.float32)
    if len(probs) == 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    dt = float(grid[1] - grid[0]) if len(grid) > 1 else 0.25
    win = max(1, int(round(float(inf.get("smooth_s", 1.0)) / max(dt, 1e-6))))
    if win > 1:
        k = np.ones(win, np.float32) / win
        pad = win // 2
        sm = np.convolve(np.pad(probs, (pad, win - 1 - pad), mode="edge"), k, mode="valid")
    else:
        sm = probs
    step = float(inf.get("risk_step_s", 0.5))
    end = float(duration) if duration else float(grid[-1])
    t_out = np.arange(0.0, end + 1e-6, step, dtype=np.float32)
    if len(t_out) == 0:
        t_out = np.zeros(1, np.float32)
    bins = np.clip(np.floor((grid + step / 2) / step).astype(int), 0, len(t_out) - 1)
    r = np.full(len(t_out), np.nan, np.float32)
    np.fmax.at(r, bins, sm)
    if np.isnan(r).any():  # пустые корзины — интерполяция
        ok = ~np.isnan(r)
        r = np.interp(t_out, t_out[ok], r[ok]).astype(np.float32) if ok.any() else np.zeros_like(t_out)
    return t_out, np.clip(r, 0.0, 1.0)


def find_peaks(t: np.ndarray, r: np.ndarray, threshold: float, min_dist_s: float, max_n: int,
               extent_ratio: float = 0.5) -> list[dict]:
    """Жадные пики выше порога с минимальным расстоянием; границы — где риск >= ratio * пик."""
    out: list[dict] = []
    if len(r) == 0:
        return out
    used = np.zeros(len(r), bool)
    order = np.argsort(-r, kind="stable")
    for i in order:
        if r[i] < threshold or len(out) >= max_n:
            break
        if used[i] or any(abs(float(t[i]) - p["t_peak"]) < min_dist_s for p in out):
            continue
        lvl = extent_ratio * r[i]
        a = i
        while a > 0 and r[a - 1] >= lvl:
            a -= 1
        b = i
        while b < len(r) - 1 and r[b + 1] >= lvl:
            b += 1
        used[a:b + 1] = True
        out.append({"t_peak": float(t[i]), "t_start": float(t[a]), "t_end": float(t[b]), "peak": float(r[i]),
                    "i": int(i)})
    return out


def video_score(r: np.ndarray) -> float:
    return float(np.max(r)) if len(r) else 0.0
