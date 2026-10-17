"""Kassa Sentinel — оркестратор пайплайна (A03).

Публичное API (сигнатуры заморожены контрактом):
    run(video_path, job_id, progress, deadline_s=240) -> AnalysisResult
    warmup() -> None
Импорт пакета не тянет тяжёлых зависимостей и не имеет побочных эффектов.
"""
from __future__ import annotations

from typing import Any

__all__ = ["run", "warmup", "run_with_context", "ModelRegistry", "PIPELINE_VERSION"]
__version__ = "1.0.0"
PIPELINE_VERSION = __version__


def run(video_path: str, job_id: str, progress: Any = None, deadline_s: float = 240, **kwargs: Any):
    from .orchestrator import run as _run

    return _run(video_path, job_id, progress, deadline_s, **kwargs)


def run_with_context(video_path: str, job_id: str, progress: Any = None, deadline_s: float = 240, **kwargs: Any):
    from .orchestrator import run_with_context as _rwc

    return _rwc(video_path, job_id, progress, deadline_s, **kwargs)


def warmup() -> None:
    from .orchestrator import warmup as _warmup

    _warmup()


def __getattr__(name: str) -> Any:  # ленивый доступ к ModelRegistry
    if name == "ModelRegistry":
        from .registry import ModelRegistry

        return ModelRegistry
    raise AttributeError(name)
