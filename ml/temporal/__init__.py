"""Kassa Sentinel — ml/temporal (A10): обучаемая временная модель на замороженных признаках SigLIP2.

Публично (контракт): ``score(clip) -> (risk_curve, events)``. Фаза 2: ``propose(clip) -> list[Event]``
— 3–8 кандидатов для VLM/правил. Доп.: ``warmup()``, ``status()``.
Импорт пакета лёгкий: torch/transformers/cv2 подгружаются лениво внутри функций.
"""
from __future__ import annotations

from typing import Any, Optional


def score(clip: Any, zones: Any = None, budget_s: Optional[float] = None
          ) -> tuple[list[tuple[float, float]], list[Any]]:
    from .infer import score as _score

    return _score(clip, zones=zones, budget_s=budget_s)


def propose(clip: Any, zones: Any = None, budget_s: Optional[float] = None, n_max: Optional[int] = None,
            mode: Optional[str] = None) -> list[Any]:
    """3–8 моментов-кандидатов (Event, extra.role="candidate") для VLM и правил."""
    from .infer import propose as _propose

    return _propose(clip, zones=zones, budget_s=budget_s, n_max=n_max, mode=mode)


def warmup() -> None:
    from .infer import warmup as _warmup

    _warmup()


def status() -> dict:
    from .infer import status as _status

    return _status()


__all__ = ["score", "propose", "warmup", "status"]
