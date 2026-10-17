"""``VideoClip`` для тестов: поверх numpy-кадров (``FakeClip``) и поверх файла через OpenCV (``clip_from_video``).

Это эталонная реализация протокола ``contracts.VideoClip`` для тест-кита. Боевой декодер с учётом rotation,
качества и т.п. — ``ml.ingest.load_video`` (A04).

    clip = FakeClip.from_scenario("V01", duration=10, fps=10)   # без файлов на диске
    clip = clip_from_video("x.mov", sample_fps=5.0)              # OpenCV
"""
from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any, Optional

__all__ = ["FakeClip", "VideoFileClip", "clip_from_video", "sample_times"]


def sample_times(duration: float, sample_fps: float) -> list[float]:
    """Таймстемпы сэмплирования: 0, 1/sample_fps, ... < duration (округлены до мс)."""
    if duration <= 0 or sample_fps <= 0:
        return []
    n = int(math.floor(duration * sample_fps - 1e-9)) + 1
    return [round(i / sample_fps, 3) for i in range(n) if i / sample_fps < duration]


class FakeClip:
    """``VideoClip`` поверх списка/массива кадров BGR uint8 с исходной частотой ``fps``."""

    def __init__(self, frames: Sequence[Any], fps: float = 30.0, sample_fps: float = 5.0, path: str = "<memory>",
                 meta: Optional[dict] = None):
        if len(frames) == 0:
            raise ValueError("FakeClip: нужен хотя бы один кадр")
        self._frames = frames
        self.path = path
        self.fps = float(fps)
        self.height, self.width = int(frames[0].shape[0]), int(frames[0].shape[1])
        self.duration = len(frames) / self.fps
        self.sample_fps = float(sample_fps)
        self.times = sample_times(self.duration, self.sample_fps)
        self.meta: dict = dict(meta or {})

    def __len__(self) -> int:
        return len(self.times)

    def _index_at(self, t: float) -> int:
        return min(len(self._frames) - 1, max(0, int(round(t * self.fps))))

    def frame(self, i: int) -> Any:
        return self._frames[self._index_at(self.times[i])]

    def frame_at(self, t: float) -> Any:
        return self._frames[self._index_at(t)]

    @classmethod
    def from_scenario(cls, scenario: str, duration: float = 10.0, fps: float = 10.0, sample_fps: float = 5.0,
                      size: tuple[int, int] = (464, 848), view: str = "default", seed: int = 0) -> "FakeClip":
        """Синтетическая касса в памяти (без файла). ``meta['gt']`` — ground truth сценария."""
        from .synth_video import render_frames

        frames, gt = render_frames(scenario, duration=duration, fps=fps, size=size, view=view, seed=seed)
        return cls(frames, fps=fps, sample_fps=sample_fps, path=f"<synth:{scenario}:{view}>", meta={"gt": gt})


class VideoFileClip:
    """``VideoClip`` поверх файла через OpenCV: сэмплированные кадры — в памяти, ``frame_at`` — с seek и LRU.

    Для тестов и коротких роликов. ``width/height`` — по реально декодированному кадру (OpenCV сам применяет
    rotation из метаданных, если сборка это поддерживает).
    """

    def __init__(self, path: str, sample_fps: float = 5.0, max_cache: int = 64):
        import cv2  # лениво

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise OSError(f"не удалось открыть видео: {path}")
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            self.fps = float(fps) if fps and math.isfinite(fps) and fps > 0 else 30.0
            sampled: dict[int, Any] = {}
            n_read = 0
            next_t = 0.0
            first = None
            step = 1.0 / sample_fps
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if first is None:
                    first = frame
                t = n_read / self.fps
                if t + 1e-6 >= next_t:
                    sampled[len(sampled)] = frame
                    next_t += step
                n_read += 1
        finally:
            cap.release()
        if first is None:
            raise OSError(f"в видео нет кадров: {path}")
        self.path = str(path)
        self.height, self.width = int(first.shape[0]), int(first.shape[1])
        self.n_frames = n_read
        self.duration = n_read / self.fps
        self.sample_fps = float(sample_fps)
        self._sampled = [sampled[i] for i in range(len(sampled))]
        self.times = [round(i * step, 3) for i in range(len(self._sampled))]
        self._cache: OrderedDict[int, Any] = OrderedDict()
        self._max_cache = max_cache

    def __len__(self) -> int:
        return len(self.times)

    def frame(self, i: int) -> Any:
        return self._sampled[i]

    def frame_at(self, t: float) -> Any:
        import cv2  # лениво

        idx = min(self.n_frames - 1, max(0, int(round(t * self.fps))))
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]
        cap = cv2.VideoCapture(self.path)
        frame = None
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                frame = None
        finally:
            cap.release()
        if frame is None:  # seek не поддержан — ближайший сэмплированный кадр
            j = min(range(len(self.times)), key=lambda k: abs(self.times[k] - t))
            frame = self._sampled[j]
        self._cache[idx] = frame
        if len(self._cache) > self._max_cache:
            self._cache.popitem(last=False)
        return frame


def clip_from_video(path: str, sample_fps: float = 5.0) -> VideoFileClip:
    """Открыть видеофайл как ``VideoClip`` (OpenCV)."""
    return VideoFileClip(path, sample_fps=sample_fps)
