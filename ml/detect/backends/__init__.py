"""Фабрика бэкендов с кэшем (модели держим тёплыми между видео)."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

from ..config import env_flag
from ..labels import COCO_TO_LABEL
from ..weights import resolve
from .base import FrameDetector, resolve_device
from .mock import MockColorDetector

log = logging.getLogger("ml.detect.backends")

CLOSED_SET_LABELS = frozenset(COCO_TO_LABEL.values())


@dataclass
class Backends:
    closed: list[FrameDetector] = field(default_factory=list)
    open: Optional[FrameDetector] = None
    mock: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def names(self) -> list[str]:
        out = [getattr(b, "name", type(b).__name__) for b in self.closed]
        if self.open is not None:
            out.append(getattr(self.open, "name", type(self.open).__name__))
        return out


_LOCK = threading.Lock()
_CACHE: dict[tuple, Backends] = {}


def is_mock() -> bool:
    return env_flag("SENTINEL_MOCK")


def _mock_backends(cfg: dict[str, Any], warnings: Optional[list[str]] = None, motion_only: bool = False) -> Backends:
    """motion_only — аварийный режим на реальном видео без весов: палитру не ищем (серый сейф/белая бумага
    дали бы ложные метки), только person по движущимся пятнам."""
    m = cfg["mock"]
    palette = {} if motion_only else m["palette"]
    common = dict(palette=palette, tolerance=m["tolerance"], min_area_px=m["min_area_px"], score=m["score"])
    closed = MockColorDetector(**common, motion_person=m.get("motion_person", True),
                               motion_person_min_area=m.get("motion_person_min_area", 0.02), name="mock")
    open_labels = {q["label"] for q in cfg["open_vocab"]["queries"]}
    opn = (MockColorDetector(**common, labels=open_labels, name="mock:open")
           if cfg["open_vocab"].get("enabled") and not motion_only else None)
    return Backends(closed=[closed], open=opn, mock=True, warnings=list(warnings or []))


def _load_closed(cfg: dict[str, Any], warnings: list[str]) -> list[FrameDetector]:
    cs = cfg["closed_set"]
    rt = cfg["runtime"]
    out: list[FrameDetector] = []
    for kind in cs.get("backends", ["hf"]):
        if kind == "onnx_finetuned":
            p, lp = resolve(cs["onnx_path"]), resolve(cs["onnx_labels"])
            if not (p.exists() and lp.exists()):
                continue
            try:
                from .onnx_det import OnnxDetrBackend

                out.append(OnnxDetrBackend(p, lp, device=rt["device"], input_size=cs["input_size"],
                                           min_score=cs["min_score"], max_dets=cs["max_dets_per_frame"]))
                if not cs.get("onnx_merge_with_hf", True):
                    break
            except Exception as e:  # noqa: BLE001
                warnings.append(f"detect: finetuned ONNX not loaded: {e}")
        elif kind == "hf":
            from .hf_detr import HFDetrBackend

            for mid in [cs["hf_model_id"], *cs.get("hf_fallback_ids", [])]:
                try:
                    out.append(HFDetrBackend(mid, device=rt["device"], fp16=rt["fp16"], input_size=cs["input_size"],
                                             min_score=cs["min_score"], max_dets=cs["max_dets_per_frame"]))
                    break
                except Exception as e:  # noqa: BLE001
                    warnings.append(f"detect: closed-set {mid} not loaded: {type(e).__name__}: {e}")
    return out


def _load_open(cfg: dict[str, Any], warnings: list[str]) -> Optional[FrameDetector]:
    ov = cfg["open_vocab"]
    if not ov.get("enabled", True):
        return None
    try:
        from .owlv2 import OwlV2Backend

        rt = cfg["runtime"]
        return OwlV2Backend(ov["hf_model_id"], ov["queries"], device=rt["device"], fp16=rt["fp16"],
                            input_size=ov["input_size"], nms_iou=ov["nms_iou"], max_dets=ov["max_dets_per_frame"],
                            min_area=ov.get("min_area", 0.0), max_area=ov.get("max_area"))
    except Exception as e:  # noqa: BLE001
        warnings.append(f"detect: open-vocab {ov['hf_model_id']} not loaded: {type(e).__name__}: {e}")
        return None


def _key(cfg: dict[str, Any]) -> tuple:
    import json

    sub = {k: cfg.get(k) for k in ("runtime", "closed_set", "open_vocab", "mock")}
    sub["runtime"] = {k: v for k, v in (sub["runtime"] or {}).items() if k in ("device", "fp16")}
    return (is_mock(), json.dumps(sub, sort_keys=True, default=str))


def get_backends(cfg: dict[str, Any]) -> Backends:
    key = _key(cfg)
    with _LOCK:
        if key in _CACHE:
            be = _CACHE[key]
            for b in [*be.closed, be.open]:
                if hasattr(b, "reset"):
                    b.reset()
            return be
        if is_mock():
            be = _mock_backends(cfg)
        else:
            warnings: list[str] = []
            closed = _load_closed(cfg, warnings)
            opn = _load_open(cfg, warnings)
            if not closed and cfg["runtime"].get("fallback_to_mock", True):
                warnings.append("detect: closed-set unavailable → mock fallback (только person по движению)")
                be = _mock_backends(cfg, warnings, motion_only=True)
                be.open = opn
                be.mock = False
            else:
                be = Backends(closed=closed, open=opn, mock=False, warnings=warnings)
            for w in warnings:
                log.warning(w)
        _CACHE[key] = be
        return be


def clear_cache() -> None:
    with _LOCK:
        _CACHE.clear()


__all__ = ["Backends", "CLOSED_SET_LABELS", "get_backends", "clear_cache", "is_mock", "resolve_device"]
