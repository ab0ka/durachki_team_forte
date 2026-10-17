"""Минимальный загрузчик видео для CLI (python -m ml.vlm.try), если ml.ingest (A04) недоступен.

В пайплайне используется ml.ingest.load_video; здесь — только запасной вариант на cv2 (лениво).
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any


class SimpleClip:
    def __init__(self, path: str, sample_fps: float = 5.0, cache: int = 64) -> None:
        import cv2  # лениво

        self.path = path
        self._cv2 = cv2
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise IOError(f"cannot open video: {path}")
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 30.0)
        n = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.n_frames = n
        self.duration = n / self.fps if self.fps > 0 else 0.0
        ok, f = self._read_index(0)
        if not ok:
            raise IOError(f"cannot decode video: {path}")
        self.height, self.width = int(f.shape[0]), int(f.shape[1])  # уже с учётом rotation (cv2 auto-orient)
        step = 1.0 / float(sample_fps)
        k = int(self.duration / step) + 1 if self.duration > 0 else 1
        self.times = [round(i * step, 4) for i in range(k) if i * step < max(self.duration, 1e-6)]
        self._cache: "OrderedDict[int, Any]" = OrderedDict()
        self._cache_n = int(cache)

    def _read_index(self, idx: int) -> tuple[bool, Any]:
        cv2 = self._cv2
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(idx)))
        return self._cap.read()

    def _get(self, idx: int) -> Any:
        idx = max(0, min(int(idx), max(0, self.n_frames - 1)))
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]
        ok, f = self._read_index(idx)
        if not ok:
            return None
        self._cache[idx] = f
        if len(self._cache) > self._cache_n:
            self._cache.popitem(last=False)
        return f

    def frame(self, i: int) -> Any:
        return self.frame_at(self.times[i])

    def frame_at(self, t: float) -> Any:
        return self._get(int(round(float(t) * self.fps)))

    def close(self) -> None:
        self._cap.release()


def open_clip(path: str, sample_fps: float = 5.0) -> Any:
    """ml.ingest.load_video, если ветка A04 уже есть; иначе SimpleClip."""
    try:
        from ml.ingest import load_video  # type: ignore

        return load_video(path, sample_fps=sample_fps)
    except ImportError:
        return SimpleClip(path, sample_fps=sample_fps)
