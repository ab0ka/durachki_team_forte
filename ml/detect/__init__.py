"""ml.detect — детекция объектов (A05). Публичный API (сигнатура заморожена контрактом):

    detect(clip, every_n: int = 1) -> list[Detection]

Доп. функции: warmup(), last_stats(), available(). Тяжёлые зависимости (torch/transformers/onnxruntime/cv2)
импортируются лениво — импорт пакета без побочных эффектов и без сети.
Нет локальных весов → detect() бросает DetectorUnavailable (mock — только явный SENTINEL_MOCK=1).
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


def available() -> dict:
    """Быстрая проверка без загрузки моделей и без сети: {mock, closed_set: {id: bool}, open_vocab: {...}}."""
    from .backends import is_mock
    from .config import load_config
    from .weights import is_cached, resolve

    cfg = load_config()
    cs, ov = cfg["closed_set"], cfg["open_vocab"]
    closed = {mid: is_cached(mid) for mid in [cs["hf_model_id"], *cs.get("hf_fallback_ids", [])]}
    if "onnx_finetuned" in cs.get("backends", []):
        closed["onnx_finetuned"] = resolve(cs["onnx_path"]).exists() and resolve(cs["onnx_labels"]).exists()
    out = {"mock": is_mock(), "closed_set": closed,
           "open_vocab": {ov["hf_model_id"]: is_cached(ov["hf_model_id"])} if ov.get("enabled") else {}}
    out["ready"] = out["mock"] or any(closed.values())
    return out


def __getattr__(name: str) -> Any:
    if name == "DetectorUnavailable":
        from .backends import DetectorUnavailable

        return DetectorUnavailable
    raise AttributeError(name)


__all__ = ["detect", "warmup", "last_stats", "available", "DetectorUnavailable"]
