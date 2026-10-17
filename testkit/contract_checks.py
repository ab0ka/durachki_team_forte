"""Проверки выходов стадий на соответствие контракту v1 — для интеграционных тестов модулей (A01).

Каждая функция возвращает список проблем (пустой — всё хорошо)::

    from testkit.contract_checks import check_detections
    assert check_detections(ml.detect.detect(clip)) == []
"""
from __future__ import annotations

import math
import re
from typing import Any, Optional

from contracts.extensions import EXTRA_LABELS, EXTRA_SIGNALS
from contracts.sentinel_contracts import (
    LABELS,
    SIGNALS,
    AnalysisResult,
    Detection,
    Event,
    PoseFrame,
    SceneState,
    Track,
    Zone,
)

__all__ = ["check_clip", "check_detections", "check_tracks_zones", "check_pose_frames", "check_scene_states",
           "check_events", "check_risk", "check_result", "check_bbox"]

ROLES = ("cashier", "visitor", "collector", "unknown")
ZONE_NAMES = ("cashier_area", "desk", "safe", "door", "client_area", "cash_tray")
TYPE_ID_RE = re.compile(r"^V(0[1-9]|[1-9]\d)$")
EVENT_SOURCES = ("rules", "temporal", "vlm", "fusion")


def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def check_bbox(b: Any, where: str) -> list[str]:
    if not isinstance(b, (tuple, list)) or len(b) != 4 or not all(_num(v) for v in b):
        return [f"{where}: bbox должен быть 4 конечных числа, получено {b!r}"]
    x1, y1, x2, y2 = b
    out = []
    if not all(-1e-6 <= v <= 1 + 1e-6 for v in b):
        out.append(f"{where}: bbox не нормирован в 0..1: {b!r}")
    if x2 < x1 or y2 < y1:
        out.append(f"{where}: bbox перевёрнут (x2<x1 или y2<y1): {b!r}")
    return out


def check_clip(clip: Any) -> list[str]:
    out = []
    for attr in ("path", "duration", "fps", "width", "height", "times"):
        if not hasattr(clip, attr):
            out.append(f"VideoClip: нет атрибута {attr}")
    if out:
        return out
    if not (_num(clip.duration) and clip.duration > 0):
        out.append(f"VideoClip.duration={clip.duration!r}")
    if not (_num(clip.fps) and clip.fps > 0):
        out.append(f"VideoClip.fps={clip.fps!r}")
    times = list(clip.times)
    if not times:
        out.append("VideoClip.times пуст")
    elif any(b <= a for a, b in zip(times, times[1:])) or times[0] < 0 or times[-1] > clip.duration + 1e-3:
        out.append("VideoClip.times должен строго возрастать в пределах [0, duration]")
    if times:
        f0 = clip.frame(0)
        shape = getattr(f0, "shape", None)
        if not shape or len(shape) != 3 or shape[2] != 3:
            out.append(f"VideoClip.frame(0): ожидается HxWx3 BGR, получено {shape}")
        elif (shape[1], shape[0]) != (clip.width, clip.height):
            out.append(f"VideoClip: width/height {clip.width}x{clip.height} != кадру {shape[1]}x{shape[0]}")
        if str(getattr(f0, "dtype", "uint8")) != "uint8":
            out.append(f"VideoClip.frame(0).dtype={f0.dtype} (нужен uint8)")
        fa = clip.frame_at(min(clip.duration, times[-1]))
        if getattr(fa, "shape", None) != shape:
            out.append("VideoClip.frame_at: форма кадра отличается от frame()")
    return out


def check_detections(dets: Any) -> list[str]:
    if not isinstance(dets, list):
        return [f"detect: ожидается list[Detection], получено {type(dets).__name__}"]
    out = []
    for i, d in enumerate(dets):
        if not isinstance(d, Detection):
            out.append(f"detections[{i}]: не Detection ({type(d).__name__})")
            continue
        if d.label not in LABELS and d.label not in EXTRA_LABELS:
            out.append(f"detections[{i}]: label {d.label!r} не из LABELS")
        if not (_num(d.t) and d.t >= 0):
            out.append(f"detections[{i}]: t={d.t!r}")
        if not (_num(d.score) and 0 <= d.score <= 1):
            out.append(f"detections[{i}]: score={d.score!r} вне 0..1")
        out += check_bbox(d.bbox, f"detections[{i}]")
    return out[:50]


def check_tracks_zones(res: Any) -> list[str]:
    if not (isinstance(res, tuple) and len(res) == 2):
        return ["track: ожидается tuple[list[Track], list[Zone]]"]
    tracks, zones = res
    out = []
    ids = set()
    for i, tr in enumerate(tracks):
        if not isinstance(tr, Track):
            out.append(f"tracks[{i}]: не Track")
            continue
        if tr.track_id in ids:
            out.append(f"tracks[{i}]: повтор track_id {tr.track_id}")
        ids.add(tr.track_id)
        if tr.role not in ROLES:
            out.append(f"tracks[{i}]: role {tr.role!r} не из {ROLES}")
        if tr.label not in LABELS:
            out.append(f"tracks[{i}]: label {tr.label!r} не из LABELS")
        ts = [t for t, _ in tr.boxes]
        if any(b < a for a, b in zip(ts, ts[1:])):
            out.append(f"tracks[{i}]: boxes не упорядочены по времени")
        for t, b in tr.boxes[:200]:
            out += check_bbox(b, f"tracks[{i}]@{t}")
    for i, z in enumerate(zones):
        if not isinstance(z, Zone):
            out.append(f"zones[{i}]: не Zone")
            continue
        if z.name not in ZONE_NAMES:
            out.append(f"zones[{i}]: имя {z.name!r} не из {ZONE_NAMES}")
        if len(z.polygon) < 3 or not all(len(p) == 2 and all(_num(c) and -1e-6 <= c <= 1 + 1e-6 for c in p)
                                         for p in z.polygon):
            out.append(f"zones[{i}]: polygon должен быть >=3 нормированных точек")
    return out[:50]


def check_pose_frames(poses: Any) -> list[str]:
    if not isinstance(poses, list):
        return ["pose: ожидается list[PoseFrame]"]
    out = []
    for i, p in enumerate(poses):
        if not isinstance(p, PoseFrame):
            out.append(f"poses[{i}]: не PoseFrame")
            continue
        if len(p.keypoints) != 17:
            out.append(f"poses[{i}]: ожидается 17 точек COCO, получено {len(p.keypoints)}")
        elif not all(len(k) == 3 and all(_num(v) for v in k) and 0 <= k[2] <= 1 for k in p.keypoints):
            out.append(f"poses[{i}]: точки должны быть (x, y, score) конечные, score 0..1")
    return out[:50]


def check_scene_states(states: Any, require_all: bool = False) -> list[str]:
    if not isinstance(states, list):
        return ["scene: ожидается list[SceneState]"]
    out = []
    prev = -1.0
    for i, s in enumerate(states):
        if not isinstance(s, SceneState):
            out.append(f"states[{i}]: не SceneState")
            continue
        if not _num(s.t) or s.t < prev:
            out.append(f"states[{i}]: t={s.t!r} не возрастает")
        prev = s.t if _num(s.t) else prev
        # расширения допустимы: EXTRA_SIGNALS (v1.1) и экспериментальные с префиксом x_
        unknown = {k for k in set(s.signals) - set(SIGNALS) - set(EXTRA_SIGNALS) if not k.startswith("x_")}
        if unknown:
            out.append(f"states[{i}]: неизвестные сигналы {sorted(unknown)}")
        if require_all and set(SIGNALS) - set(s.signals):
            out.append(f"states[{i}]: нет сигналов {sorted(set(SIGNALS) - set(s.signals))}")
        for k, v in s.signals.items():
            if not _num(v):
                out.append(f"states[{i}].{k}={v!r} не конечное число")
            elif k == "n_persons":
                if v < 0 or abs(v - round(v)) > 1e-6:
                    out.append(f"states[{i}].n_persons={v} должен быть целым >= 0")
            elif not 0 <= v <= 1:
                out.append(f"states[{i}].{k}={v} вне 0..1")
    return out[:50]


def check_events(events: Any, duration: Optional[float] = None, catalog: Optional[dict] = None) -> list[str]:
    if not isinstance(events, list):
        return ["events: ожидается list[Event]"]
    out = []
    known = set((catalog or {}).get("types", {}))
    for i, e in enumerate(events):
        if not isinstance(e, Event):
            out.append(f"events[{i}]: не Event")
            continue
        if not TYPE_ID_RE.match(e.type_id or ""):
            out.append(f"events[{i}]: type_id {e.type_id!r} не вида V01..V99")
        elif known and e.type_id not in known:
            out.append(f"events[{i}]: type_id {e.type_id} нет в каталоге")
        if not all(_num(v) for v in (e.t_start, e.t_end, e.t_peak, e.confidence)):
            out.append(f"events[{i}]: t_start/t_end/t_peak/confidence должны быть конечными")
            continue
        if not e.t_start - 1e-6 <= e.t_peak <= e.t_end + 1e-6:
            out.append(f"events[{i}]: нужно t_start <= t_peak <= t_end ({e.t_start}, {e.t_peak}, {e.t_end})")
        if not 0 <= e.confidence <= 1:
            out.append(f"events[{i}]: confidence {e.confidence} вне 0..1")
        if duration is not None and e.t_peak > duration + 0.05:
            out.append(f"events[{i}]: t_peak {e.t_peak} > длительности {duration}")
        if e.source not in EVENT_SOURCES and e.source != "testkit":
            out.append(f"events[{i}]: source {e.source!r} не из {EVENT_SOURCES}")
        for k, b in e.bboxes.items():
            out += check_bbox(b, f"events[{i}].bboxes[{k}]")
    return out[:50]


def check_risk(risk: Any, duration: Optional[float] = None) -> list[str]:
    if not isinstance(risk, list):
        return ["risk: ожидается list[tuple[t, risk]]"]
    out = []
    for i, pt in enumerate(risk):
        try:
            t, r = pt
        except (TypeError, ValueError):
            out.append(f"risk[{i}]: ожидается пара (t, risk)")
            continue
        if not (_num(t) and _num(r) and 0 <= r <= 1 and t >= 0):
            out.append(f"risk[{i}]: ({t!r}, {r!r}) — t>=0, risk в 0..1")
        elif duration is not None and t > duration + 0.05:
            out.append(f"risk[{i}]: t={t} > длительности")
    return out[:50]


def check_result(result: Any, duration: Optional[float] = None) -> list[str]:
    """AnalysisResult + его сериализация для жюри проходит строгую проверку ТЗ."""
    if not isinstance(result, AnalysisResult):
        return [f"fuse/run: ожидается AnalysisResult, получено {type(result).__name__}"]
    from contracts.serialize import result_to_jury_api
    from contracts.validate import jury_response_errors

    return jury_response_errors(result_to_jury_api(result), duration)
