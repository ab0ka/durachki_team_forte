"""ml.vlm — локальная мультимодальная модель (A11): верификация событий и поиск новых типов нарушений.

Публичные функции (сигнатуры контракта v1 + необязательные keyword-only расширения):
    verify(clip, events, budget_s, *, limits=None) -> list[Event]
    discover(clip, budget_s, *, limits=None) -> list[Event]
Служебные: warmup(), unload(), status(), set_gpu_lock(lock), set_catalog(catalog), set_backend(backend).

Импорт пакета лёгкий: torch/transformers/cv2 подгружаются только при первом реальном вызове.
Ни verify, ни discover не бросают исключений наружу — при сбое стадия деградирует (события как были / []).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

log = logging.getLogger(__name__)

__all__ = ["verify", "discover", "warmup", "unload", "status", "set_gpu_lock", "set_catalog", "set_backend"]


def verify(clip: Any, events: list[Any], budget_s: float, *, limits: Optional[dict[str, Any]] = None) -> list[Any]:
    """Проверка событий-кандидатов. Возвращает те же события (1:1, тот же порядок) с обновлёнными
    confidence/facts и extra.vlm (agreement: supports|contradicts|uncertain|type_mismatch|not_checked)."""
    from .verifier import _copy_event, verify_events

    try:
        return verify_events(clip, list(events or []), float(budget_s), limits=limits)
    except Exception as e:  # noqa: BLE001 — стадия не должна ронять пайплайн
        log.warning("ml.vlm.verify failed: %s", type(e).__name__)
        out = [_copy_event(ev) for ev in (events or [])]
        for ev in out:
            ev.extra["vlm"] = {"status": "error", "agreement": "not_checked", "error": type(e).__name__}
        return out


def discover(clip: Any, budget_s: float, *, limits: Optional[dict[str, Any]] = None) -> list[Any]:
    """Поиск нарушений вне правил (включая V99). По умолчанию выключен (config: discover.enabled=false)."""
    from .discoverer import discover_events

    try:
        return discover_events(clip, float(budget_s), limits=limits)
    except Exception as e:  # noqa: BLE001
        log.warning("ml.vlm.discover failed: %s", type(e).__name__)
        return []


def warmup() -> None:
    """Загрузить модель заранее (держим тёплой). Без бэкенда — тихо ничего не делает."""
    from .config import load_config
    from .runtime import ensure_loaded, get_backend

    be, _ = get_backend(load_config())
    if be is not None:
        ensure_loaded(be)


def unload() -> None:
    """Освободить VRAM (A03 может вызывать между видео/перед тяжёлыми стадиями)."""
    from .runtime import unload_backend

    unload_backend()


def status() -> dict[str, Any]:
    """Технический статус (без содержимого кадров/ответов): бэкенд, модель, загружена ли, причина отказа."""
    from .runtime import backend_status

    return backend_status()


def set_gpu_lock(lock: Any) -> None:
    from .runtime import set_gpu_lock as _set

    _set(lock)


def set_catalog(catalog: Optional[dict[str, Any]]) -> None:
    from .catalog import set_catalog as _set

    _set(catalog)


def set_backend(backend: Any) -> None:
    from .runtime import set_backend as _set

    _set(backend)
