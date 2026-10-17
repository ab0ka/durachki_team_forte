"""JSON-(де)сериализация Track/Zone/Detection и кэш стадии трекинга.

Файлы в каталоге кэша (cache/<sha>/):
  track.json       — {"tracks": [...], "zones": [...], "info": {...}} (пишет ml.track)
  detections.json  — список детекций (или {"detections": [...]}) — читаем, если есть.
Также понимаем dets.json / detect.json и pickle (*.pkl) с list[Detection] — только свои локальные кэши.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Optional, Sequence

from .contract import Detection, Track, Zone


def _get(o: Any, k: str, default: Any = None) -> Any:
    return o.get(k, default) if isinstance(o, dict) else getattr(o, k, default)


def track_to_dict(t: Any) -> dict:
    return {"track_id": int(_get(t, "track_id")), "label": _get(t, "label"), "role": _get(t, "role", "unknown"),
            "role_conf": float(_get(t, "role_conf", 0.0) or 0.0),
            "boxes": [[float(tt), *[float(x) for x in b]] for tt, b in (_get(t, "boxes", []) or [])]}


def track_from_dict(d: dict) -> Track:
    boxes = []
    for row in d.get("boxes", []):
        if len(row) == 2:  # [t, [x1,y1,x2,y2]]
            boxes.append((float(row[0]), tuple(float(x) for x in row[1])))
        else:
            boxes.append((float(row[0]), tuple(float(x) for x in row[1:5])))
    return Track(track_id=int(d["track_id"]), label=str(d.get("label", "person")), role=d.get("role", "unknown"),
                 role_conf=float(d.get("role_conf", 0.0)), boxes=boxes)


def zone_to_dict(z: Any) -> dict:
    return {"name": _get(z, "name"), "polygon": [[float(x), float(y)] for x, y in _get(z, "polygon", [])],
            "conf": float(_get(z, "conf", 1.0))}


def zone_from_dict(d: dict) -> Zone:
    return Zone(name=str(d["name"]), polygon=[(float(x), float(y)) for x, y in d.get("polygon", [])],
                conf=float(d.get("conf", 1.0)))


def detection_from_dict(d: dict) -> Detection:
    bb = d.get("bbox") or [d.get("x1"), d.get("y1"), d.get("x2"), d.get("y2")]
    return Detection(t=float(d["t"]), label=str(d["label"]), bbox=tuple(float(x) for x in bb),
                     score=float(d.get("score", 1.0)), track_id=d.get("track_id"), source=str(d.get("source", "")))


def detection_to_dict(d: Any) -> dict:
    return {"t": float(_get(d, "t")), "label": _get(d, "label"), "bbox": [float(x) for x in _get(d, "bbox")],
            "score": float(_get(d, "score", 1.0)), "track_id": _get(d, "track_id"), "source": _get(d, "source", "")}


def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if hasattr(o, "item") and callable(o.item):
        try:
            return o.item()
        except Exception:
            pass
    if isinstance(o, float) and o != o:
        return None
    return o


def save_track_cache(cache_dir: str, tracks: Sequence[Any], zones: Sequence[Any], info: Optional[dict] = None) -> Path:
    p = Path(cache_dir)
    p.mkdir(parents=True, exist_ok=True)
    out = p / "track.json"
    data = {"tracks": [track_to_dict(t) for t in tracks], "zones": [zone_to_dict(z) for z in zones],
            "info": _jsonable(info or {})}
    out.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return out


def load_track_cache(cache_dir: str) -> Optional[tuple[list[Track], list[Zone], dict]]:
    p = Path(cache_dir) / "track.json"
    if not p.is_file():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    return ([track_from_dict(d) for d in data.get("tracks", [])], [zone_from_dict(d) for d in data.get("zones", [])],
            data.get("info", {}))


def load_detections(cache_dir: str) -> Optional[list[Detection]]:
    p = Path(cache_dir)
    for name in ("detections.json", "dets.json", "detect.json"):
        f = p / name
        if f.is_file():
            data = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data = data.get("detections") or data.get("dets") or []
            return [detection_from_dict(d) for d in data]
    for name in ("detections.pkl", "dets.pkl", "detect.pkl"):
        f = p / name
        if f.is_file():
            with f.open("rb") as fh:
                data = pickle.load(fh)  # локальный кэш собственного пайплайна
            return [d if hasattr(d, "bbox") else detection_from_dict(d) for d in data]
    return None
