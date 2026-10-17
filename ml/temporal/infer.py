"""Инференс: score(clip) -> (risk curve 0.5 с, события). Без обученного чекпоинта — ([], []):
случайный «production score» не отдаём. Тип события — только из головы, обученной на проверенных type_id;
иначе V99 + zero-shot подсказка в extra (явно помечена как НЕ обученная классификация)."""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .config import TYPE_IDS, feature_signature, is_mock, load_config, stable_hash, weights_dir
from .features import FeatureSet, cache_path, extract_clip, get_backend, type_scores

try:
    from contracts.sentinel_contracts import Event  # noqa
except ImportError:  # пока ветка A01 не смерджена
    from contracts.sentinel_contracts import Event  # noqa

log = logging.getLogger("sentinel.temporal")
_CK_LOCK = threading.Lock()
_CK: dict[str, Any] = {}
_LAST: dict[str, Any] = {}


def _checkpoint(cfg: dict) -> Optional[tuple[Any, Any, dict]]:
    from .predictors import load_checkpoint

    root = weights_dir(cfg)
    sel = root / "selected.json"
    stamp = f"{root}:{sel.stat().st_mtime if sel.exists() else 0}"
    with _CK_LOCK:
        if stamp not in _CK:
            try:
                _CK.clear()
                _CK[stamp] = load_checkpoint(root, cfg)
            except Exception as e:  # битый/несовместимый чекпоинт — не роняем пайплайн
                log.warning("temporal: не удалось загрузить чекпоинт из %s: %s", root, e)
                _CK[stamp] = None
        return _CK[stamp]


def status() -> dict:
    """Диагностика последнего вызова score() (для meta/warnings пайплайна)."""
    return dict(_LAST)


def warmup() -> None:
    """Загрузить чекпоинт и бэкенд признаков заранее (держим тёплыми)."""
    cfg = load_config()
    ck = _checkpoint(cfg)
    if ck is None:
        return
    try:
        from .data_index import bank_from_meta

        backend = get_backend(cfg, force=ck[1].backend)
        bank_from_meta(cfg, ck[2], backend)
        backend.encode_images(np.zeros((1, 224, 224, 3), np.uint8))
    except Exception as e:
        log.warning("temporal: warmup не удался: %s", e)


def _clip_key(clip: Any, cfg: dict, backend: str, fps: float, views: list, zones: Any) -> Optional[dict]:
    p = Path(str(getattr(clip, "path", "") or ""))
    if not p.is_file():
        return None
    from .labels import file_md5

    zkey = stable_hash([(getattr(z, "name", ""), getattr(z, "polygon", [])) for z in (zones or [])])
    return {"md5": file_md5(p), "sig": feature_signature(cfg, backend), "fps": fps, "views": views, "z": zkey,
            "aug": 0}


def score(clip: Any, zones: Any = None, budget_s: Optional[float] = None
          ) -> tuple[list[tuple[float, float]], list[Any]]:
    """Публичная функция (сигнатура контракта: score(clip)); zones/budget_s — необязательные расширения."""
    from .postproc import find_peaks, risk_curve

    t0 = time.monotonic()
    cfg = load_config()
    inf = cfg["infer"]
    _LAST.clear()
    ck = _checkpoint(cfg)
    if ck is None:
        _LAST.update(status="no_checkpoint", warning=f"нет обученной модели в {weights_dir(cfg)} — temporal пропущен")
        log.warning("temporal: %s", _LAST["warning"])
        return [], []
    model, spec, meta = ck
    if (spec.backend == "mock") != is_mock():
        _LAST.update(status="backend_mismatch",
                     warning=f"чекпоинт обучен на признаках '{spec.backend}', а режим mock={is_mock()}")
        log.warning("temporal: %s", _LAST["warning"])
        return [], []
    try:
        from .data_index import bank_from_meta

        backend = get_backend(cfg, force=spec.backend)
        bank = bank_from_meta(cfg, meta, backend)
        on_cuda = str(getattr(backend, "device", "cpu")).startswith("cuda") or backend.name == "mock"
        fps = float(spec.fps if on_cuda else cfg["features"].get("cpu_fps", 1.0))
        views = list(spec.views if on_cuda else cfg["features"].get("cpu_views", ["full"]))
        budget = float(budget_s if budget_s is not None else inf.get("budget_s", 30.0))
        key = _clip_key(clip, cfg, backend.name, fps, views, zones)
        fs: Optional[FeatureSet] = None
        if key is not None and cache_path(cfg, key).exists():
            fs = FeatureSet.load(cache_path(cfg, key))
        if fs is None:
            fs = extract_clip(clip, backend, cfg, zones=zones, fps=fps, views=views,
                              deadline=t0 + 0.85 * budget)
            if key is not None and "truncated_at" not in fs.meta:
                try:
                    fs.save(cache_path(cfg, key))
                except OSError:
                    pass
        if len(fs) < int(inf.get("min_frames", 4)):
            _LAST.update(status="too_short", n=len(fs))
            return [], []
        from .dataset import build_inputs

        X, grid = build_inputs(fs, bank, spec)
        probs, tprobs = model.predict(X, spec.fps)
    except Exception as e:  # graceful: временная модель не должна ронять пайплайн
        _LAST.update(status="error", warning=f"temporal: {type(e).__name__}: {e}")
        log.exception("temporal: ошибка инференса")
        return [], []

    duration = float(getattr(clip, "duration", 0.0) or grid[-1])
    t_curve, r_curve = risk_curve(probs, grid, cfg, duration)
    thr = float(meta.get("threshold", inf["event_threshold"]))
    peaks = find_peaks(t_curve, r_curve, thr, float(inf["min_peak_distance_s"]), int(inf["max_events"]),
                       float(inf["extent_ratio"]))
    zs, zs_keys = type_scores(fs, bank)
    events = []
    for pk in peaks:
        events.append(_make_event(pk, model, meta, grid, tprobs, fs, zs, zs_keys, bank))
    _LAST.update(status="ok", kind=model.kind, threshold=thr, n_events=len(events),
                 video_score=float(r_curve.max()) if len(r_curve) else 0.0,
                 elapsed_s=round(time.monotonic() - t0, 3), n_frames=len(fs), fps=fs.meta.get("fps"),
                 views=fs.views, truncated_at=fs.meta.get("truncated_at"))
    return [(round(float(t), 2), round(float(r), 4)) for t, r in zip(t_curve, r_curve)], events


def _window(times: np.ndarray, t: float, half: float) -> np.ndarray:
    m = np.abs(times - t) <= half
    if not m.any():
        m = np.zeros(len(times), bool)
        m[int(np.argmin(np.abs(times - t)))] = True
    return m


def _make_event(pk: dict, model: Any, meta: dict, grid: np.ndarray, tprobs: Optional[np.ndarray],
                fs: FeatureSet, zs: np.ndarray, zs_keys: list[str], bank: Any) -> Any:
    tp, a, b = pk["t_peak"], pk["t_start"], pk["t_end"]
    extra: dict[str, Any] = {"kind": model.kind, "peak_risk": round(pk["peak"], 4),
                             "threshold": meta.get("threshold"), "model_created": meta.get("created")}
    type_id = "V99"
    if tprobs is not None and getattr(model, "type_trained", False):
        tp_mean = tprobs[_window(grid, tp, 1.5)].mean(axis=0)
        k = int(np.argmax(tp_mean))
        type_id = TYPE_IDS[k] if k < len(TYPE_IDS) else "V99"
        extra["type_source"] = "trained_type_head"
        extra["type_probs"] = {TYPE_IDS[i]: round(float(v), 4) for i, v in enumerate(tp_mean) if i < len(TYPE_IDS)}
    else:
        extra["type_source"] = "none"  # тип должен прийти от rules/vlm; ниже — только подсказка
    if len(zs_keys):
        m = _window(fs.times, tp, 1.5)
        zmean = zs[m].mean(axis=0)
        order = np.argsort(-zmean)[:3]
        extra["zero_shot_hint"] = {
            "trained": False, "source": f"zero-shot {fs.meta.get('backend', '')}",
            "note": "сходство кадра с текстовыми подсказками, не обученная классификация",
            "top": [{"type_id": zs_keys[i], "margin_vs_normal": round(float(zmean[i]), 4)} for i in order]}
    facts = [f"Временная модель ({model.kind}) отметила повышенный риск в интервале {a:.1f}–{b:.1f} с "
             f"(максимум {pk['peak']:.2f} на {tp:.1f} с)."]
    bboxes = {}
    if "focus" in fs.views and len(fs.boxes):
        i = int(np.argmin(np.abs(fs.times - tp)))
        box = tuple(float(round(v, 4)) for v in fs.boxes[i])
        if box != (0.0, 0.0, 1.0, 1.0):
            bboxes["temporal_focus"] = box
            extra["temporal_focus_mode"] = fs.meta.get("focus_mode")
    return Event(type_id=type_id, t_start=round(a, 2), t_end=round(b, 2), t_peak=round(tp, 2),
                 confidence=round(float(pk["peak"]), 4), source="temporal", facts=facts, bboxes=bboxes,
                 extra=extra)
