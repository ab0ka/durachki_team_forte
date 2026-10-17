"""ml.detect — детекция объектов (A05). Публичный API (сигнатура заморожена контрактом):

    detect(clip, every_n: int = 1) -> list[Detection]

Доп. функции: warmup(), last_stats(). Тяжёлые зависимости (torch/transformers/onnxruntime/cv2)
импортируются лениво — импорт пакета без побочных эффектов.
"""
from __future__ import annotations

from typing import Any


def detect(clip: Any, every_n: int = 1, **kw: Any) -> list:
    from .pipeline import detect as _detect

    return _detect(clip, every_n, **kw)


def warmup(**kw: Any) -> dict:
    from .pipeline import warmup as _warmup

    return _warmup(**kw)


def last_stats() -> dict:
    from .pipeline import last_stats as _last_stats

    return _last_stats()


__all__ = ["detect", "warmup", "last_stats"]
