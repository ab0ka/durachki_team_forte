"""ml.pose — поза COCO-17 и поза-сигналы Kassa Sentinel (агент A07).

Публичные функции (сигнатуры контракта v1):
    estimate(clip, tracks) -> list[PoseFrame]
    pose_signals(poses, tracks, zones) -> list[SceneState]
Дополнительно: warmup() -> bool, status() -> dict (последний запуск estimate: backend, ошибки, тайминги).

Тяжёлые зависимости (numpy/cv2/onnxruntime) импортируются лениво — импорт пакета без побочных эффектов.
"""
from __future__ import annotations

from typing import Any, Sequence

from ._contract import PoseFrame, PoseFrameX, PoseSceneState, SceneState, Track, Zone
from .status import status

__all__ = ["estimate", "pose_signals", "warmup", "status", "PoseFrameX", "PoseSceneState"]


def estimate(clip: Any, tracks: Sequence[Track]) -> list[PoseFrame]:
    """Top-down поза по bbox-ам person-треков. При сбое бэкенда — [] + запись в status() (без исключений)."""
    from .estimator import estimate_impl

    return list(estimate_impl(clip, tracks))


def pose_signals(poses: Sequence[PoseFrame], tracks: Sequence[Track], zones: Sequence[Zone]) -> list[SceneState]:
    """Поза → SceneState (signals только из SIGNALS; нет ключа = unknown; ≥ 0.5 = активен)."""
    from .signals import compute_pose_signals

    return list(compute_pose_signals(poses, tracks, zones))


def warmup() -> bool:
    """Загрузить и прогреть бэкенд позы. False — недоступен (пайплайн продолжает без позы)."""
    from .estimator import warmup_impl

    return warmup_impl()
