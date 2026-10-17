"""Синтетические ряды контракта по сценариям — чтобы rules/fusion/pipeline тестировались без CV (A01).

Всё строится из того же ``SceneScript``, что и синтетическое видео, поэтому фикстуры согласованы с роликом::

    b = make_bundle("V02", duration=20, sample_fps=5)
    events = ml.rules.run_rules(b.states, b.tracks, b.zones, b.catalog)
    assert any(e.type_id == "V02" and abs(e.t_peak - b.gt["t"]) < 3 for e in events)

``view`` (default/mirror/rot180/landscape/perspective) преобразует геометрию (bbox, зоны, позы) как при другом
угле камеры; семантика сигналов ``SceneState`` от ракурса не зависит.
В каталоге-заглушке ``make_catalog()`` НЕТ ссылок на нормы — настоящие только в norms/catalog.yaml (A13).
"""
from __future__ import annotations

import base64
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from contracts.extensions import Completeness, Provenance
from contracts.sentinel_contracts import (
    LABELS,
    SIGNALS,
    AnalysisResult,
    Detection,
    Event,
    EvidenceFrame,
    PoseFrame,
    SceneState,
    Track,
    Violation,
    Zone,
)

from .fake_clip import sample_times
from .scenarios import (
    BODY_A,
    DEFAULT_SIZE,
    DOOR_HINGE,
    DOOR_LEN,
    HEAD_R,
    LAYOUT,
    PARTITION_Y,
    SCENARIOS,
    SEAT,
    PersonState,
    SceneScript,
    Snapshot,
    map_box,
    map_point,
    person_box,
)

__all__ = [
    "ScenarioBundle",
    "scenario_names",
    "make_zones",
    "make_detections",
    "make_tracks",
    "make_pose_frames",
    "make_scene_states",
    "make_expected_events",
    "make_catalog",
    "make_evidence_b64",
    "make_analysis_result",
    "make_bundle",
    "signals_at",
]

TYPE_NAMES = {
    "V01": "Посторонний в кассовом помещении",
    "V02": "Кассир покинул рабочее место, оставив ценности без присмотра",
    "V03": "Сейф открыт и оставлен без присмотра",
    "V04": "Дверь кассового помещения открыта",
    "V05": "Использование мобильного телефона при работе с ценностями",
    "V06": "Приём пищи или напитков на рабочем месте с наличными",
    "V07": "Нарушение порядка приёма-передачи инкассаторской сумки",
    "V08": "Операции с наличными вне обзора камеры",
    "V99": "Иное подозрительное действие",
}
_STATIC_TRACKS = {"safe": 101, "door": 102, "laptop": 103, "money_counter": 104}
_ITEM_TRACKS = {"cash": 201, "cash_bag": 202, "cell_phone": 203, "cup": 204, "food": 205}


def scenario_names() -> list[str]:
    return list(SCENARIOS)


# ----------------------------------------------------------------------------- зоны

def _rect_poly(b: tuple[float, float, float, float]) -> list[tuple[float, float]]:
    return [(b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])]


def make_zones(view: str = "default", size: tuple[int, int] = DEFAULT_SIZE) -> list[Zone]:
    """Зоны сцены (те, что A06 должен находить автоматически) в координатах ракурса ``view``."""
    py = LAYOUT["partition"]
    base = {
        "cashier_area": [(0.0, 0.0), (1.0, 0.0), (1.0, py[1]), (0.0, py[1])],
        "desk": _rect_poly(LAYOUT["desk"]),
        "safe": _rect_poly(LAYOUT["safe_stand"]),
        "door": _rect_poly((0.68, 0.50, 0.99, 0.62)),
        "client_area": [(0.0, py[3]), (1.0, py[3]), (1.0, 1.0), (0.0, 1.0)],
        "cash_tray": _rect_poly(LAYOUT["cash_tray"]),
    }
    out = []
    for name, poly in base.items():
        mapped = [tuple(min(1.0, max(0.0, c)) for c in map_point(p, view, size)) for p in poly]
        out.append(Zone(name=name, polygon=mapped, conf=0.9))  # type: ignore[arg-type]
    return out


# ----------------------------------------------------------------------------- детекции/треки

def _snapshot_boxes(snap: Snapshot, aspect: float) -> list[tuple[str, tuple, Optional[int], float]]:
    """(label, bbox базового ракурса, track_id, score) для всех видимых объектов снимка."""
    out: list[tuple[str, tuple, Optional[int], float]] = []
    for p in snap.persons:
        out.append(("person", person_box(p, aspect), p.pid, 0.93))
    out.append(("laptop", LAYOUT["laptop"], _STATIC_TRACKS["laptop"], 0.9))
    out.append(("money_counter", LAYOUT["money_counter"], _STATIC_TRACKS["money_counter"], 0.8))
    out.append(("document", LAYOUT["document"], None, 0.6))
    if snap.safe_open > 0.5:
        x1, y1, x2, y2 = LAYOUT["safe"]
        out.append(("safe_open", (x1, y1, x2 + 0.03, y2 + 0.09), _STATIC_TRACKS["safe"], 0.88))
    else:
        out.append(("safe", LAYOUT["safe"], _STATIC_TRACKS["safe"], 0.9))
    if snap.door_open > 0.5:
        th = math.radians(85.0 * snap.door_open)
        ex, ey = DOOR_HINGE[0] - DOOR_LEN * math.cos(th), DOOR_HINGE[1] + DOOR_LEN * math.sin(th) * aspect
        x1, y1, x2, y2 = LAYOUT["door"]
        out.append(("door_open", (min(x1, ex), y1, max(x2, ex + 0.02), max(y2, ey)), _STATIC_TRACKS["door"], 0.85))
    else:
        out.append(("door", LAYOUT["door"], _STATIC_TRACKS["door"], 0.85))
    for it in snap.items:
        out.append((it.label, it.bbox, _ITEM_TRACKS.get(it.label), 0.8))
    return out


def make_detections(scenario: str, duration: float = 20.0, sample_fps: float = 5.0, *, view: str = "default",
                    size: tuple[int, int] = DEFAULT_SIZE, tracked: bool = False, noise: float = 0.0,
                    dropout: float = 0.0, seed: int = 0) -> list[Detection]:
    """«Идеальный» детектор по сценарию. ``noise`` — σ дрожания bbox (доля кадра), ``dropout`` — доля пропусков."""
    script = SceneScript(scenario, duration, size)
    rng = random.Random(seed)
    dets: list[Detection] = []
    for t in sample_times(duration, sample_fps):
        snap = script.at(t)
        for label, box, tid, score in _snapshot_boxes(snap, script.aspect):
            if dropout and rng.random() < dropout:
                continue
            b = map_box(box, view, size)
            if noise:
                j = [rng.gauss(0, noise) for _ in range(4)]
                b = tuple(min(1.0, max(0.0, v + d)) for v, d in zip(b, j))  # type: ignore[assignment]
                b = (min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3]))
                score = min(1.0, max(0.05, score + rng.gauss(0, noise * 2)))
            if b[2] - b[0] <= 1e-4 or b[3] - b[1] <= 1e-4:
                continue
            dets.append(Detection(t=t, label=label, bbox=b, score=round(score, 3),
                                  track_id=tid if tracked else None, source="testkit"))
    return dets


def make_tracks(scenario: str, duration: float = 20.0, sample_fps: float = 5.0, *, view: str = "default",
                size: tuple[int, int] = DEFAULT_SIZE) -> list[Track]:
    """Треки людей (с ролями) и ключевых объектов (сейф, дверь, наличные, сумка, телефон...)."""
    script = SceneScript(scenario, duration, size)
    tracks: dict[int, Track] = {}
    roles = {1: "cashier", 2: "visitor", 3: "collector"}
    for t in sample_times(duration, sample_fps):
        for label, box, tid, _ in _snapshot_boxes(script.at(t), script.aspect):
            if tid is None:
                continue
            base_label = {"safe_open": "safe", "door_open": "door"}.get(label, label)
            tr = tracks.get(tid)
            if tr is None:
                role = roles.get(tid, "unknown") if base_label == "person" else "unknown"
                tr = tracks[tid] = Track(track_id=tid, label=base_label, role=role,  # type: ignore[arg-type]
                                         role_conf=0.9 if role != "unknown" else 0.0)
            tr.boxes.append((t, map_box(box, view, size)))
    return sorted(tracks.values(), key=lambda tr: tr.track_id)


def _pose_points(p: PersonState, aspect: float) -> list[tuple[float, float, float]]:
    a = math.radians(p.facing)
    fx, fy = math.sin(a), -math.cos(a)
    rx, ry = -fy, fx  # правая сторона человека

    def pt(dx_f: float, dx_r: float, score: float) -> tuple[float, float, float]:
        return (p.x + fx * dx_f + rx * dx_r, p.y + (fy * dx_f + ry * dx_r) * aspect, score)

    hand_s = 0.85 if p.hands_visible else 0.05
    (lhx, lhy), (rhx, rhy) = p.hands if len(p.hands) == 2 else ((p.x, p.y), (p.x, p.y))
    ls, rs = pt(0, -BODY_A * 0.8, 0.9), pt(0, BODY_A * 0.8, 0.9)
    return [
        pt(HEAD_R * 0.8, 0, 0.6),                      # nose
        pt(HEAD_R * 0.6, -HEAD_R * 0.35, 0.5),          # left_eye
        pt(HEAD_R * 0.6, HEAD_R * 0.35, 0.5),           # right_eye
        pt(0, -HEAD_R, 0.5), pt(0, HEAD_R, 0.5),        # ears
        ls, rs,                                          # shoulders
        ((ls[0] + lhx) / 2, (ls[1] + lhy) / 2, hand_s),  # left_elbow
        ((rs[0] + rhx) / 2, (rs[1] + rhy) / 2, hand_s),  # right_elbow
        (lhx, lhy, hand_s), (rhx, rhy, hand_s),          # wrists
        pt(-0.02, -BODY_A * 0.5, 0.3), pt(-0.02, BODY_A * 0.5, 0.3),  # hips
        pt(0.04, -BODY_A * 0.4, 0.1), pt(0.04, BODY_A * 0.4, 0.1),    # knees
        pt(0.06, -BODY_A * 0.4, 0.05), pt(0.06, BODY_A * 0.4, 0.05),  # ankles
    ]


def make_pose_frames(scenario: str, duration: float = 20.0, sample_fps: float = 5.0, *, view: str = "default",
                     size: tuple[int, int] = DEFAULT_SIZE) -> list[PoseFrame]:
    """COCO-17 позы людей (x, y нормированы к кадру ракурса, score; скрытые руки — score 0.05)."""
    script = SceneScript(scenario, duration, size)
    out = []
    for t in sample_times(duration, sample_fps):
        for p in script.at(t).persons:
            kps = []
            for x, y, s in _pose_points(p, script.aspect):
                mx, my = map_point((x, y), view, size)
                kps.append((min(1.0, max(0.0, mx)), min(1.0, max(0.0, my)), s))
            out.append(PoseFrame(t=t, track_id=p.pid, keypoints=kps))
    return out


# ----------------------------------------------------------------------------- сигналы сцены

def signals_at(snap: Snapshot) -> dict[str, float]:
    """Все 17 сигналов ``SIGNALS`` для снимка сцены (семантика не зависит от ракурса)."""
    cashier = snap.person("cashier")
    in_work = bool(cashier and abs(cashier.x - SEAT[0]) < 0.15 and abs(cashier.y - SEAT[1]) < 0.08)
    visitors_inside = [p for p in snap.persons if p.role == "visitor" and p.y < PARTITION_Y - 0.01]
    cash = [i for i in snap.items if i.label == "cash"]
    cash_on_desk = [i for i in cash if i.holder is None and _inside(i.bbox, LAYOUT["desk"])]
    phone = next((i for i in snap.items if i.label == "cell_phone" and i.holder == 1), None)
    eat = [i for i in snap.items if i.label in ("cup", "food", "bottle") and i.holder == 1]
    facing_away = bool(cashier and (cashier.facing % 360 < 60 or cashier.facing % 360 > 300))
    s = {
        "cashier_present": 1.0 if cashier else 0.0,
        "cashier_in_workplace": 1.0 if in_work else 0.0,
        "n_persons": float(len(snap.persons)),
        "outsider_in_cashier_area": 1.0 if visitors_inside else 0.0,
        "collector_present": 1.0 if snap.person("collector") else 0.0,
        "safe_open": round(snap.safe_open, 3),
        "door_open": round(snap.door_open, 3),
        "cash_visible": 1.0 if cash else 0.0,
        "cash_bag_visible": 1.0 if any(i.label == "cash_bag" for i in snap.items) else 0.0,
        "phone_in_hand": 1.0 if phone else 0.0,
        "phone_to_ear": 1.0 if phone and phone.near_head else 0.0,
        "eating_drinking": 1.0 if any(i.near_head for i in eat) else (0.3 if eat else 0.0),
        "hands_hidden": 1.0 if cashier and not cashier.hands_visible else 0.0,
        "cashier_facing_away": 1.0 if facing_away else 0.0,
        "workplace_unattended_with_cash": 1.0 if cash_on_desk and not in_work else 0.0,
        "handover_motion": round(snap.handover, 3),
        "motion_energy": round(min(1.0, 2.0 * sum(p.speed for p in snap.persons)), 3),
    }
    assert list(s) == SIGNALS
    return s


def _inside(b: tuple, outer: tuple, tol: float = 0.02) -> bool:
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return outer[0] - tol <= cx <= outer[2] + tol and outer[1] - tol <= cy <= outer[3] + tol


def make_scene_states(scenario: str, duration: float = 20.0, sample_fps: float = 5.0, *, noise: float = 0.0,
                      seed: int = 0) -> list[SceneState]:
    """Ряд ``SceneState`` (все ``SIGNALS``) — вход движка правил. ``noise`` — σ шума для 0..1-сигналов."""
    script = SceneScript(scenario, duration)
    rng = random.Random(seed)
    out = []
    for t in sample_times(duration, sample_fps):
        sig = signals_at(script.at(t))
        if noise:
            for k, v in sig.items():
                if k != "n_persons":
                    sig[k] = round(min(1.0, max(0.0, v + rng.gauss(0, noise))), 3)
        out.append(SceneState(t=t, signals=sig))
    return out


# ----------------------------------------------------------------------------- события, каталог, результат

def make_expected_events(scenario: str, duration: float = 20.0) -> list[Event]:
    """Эталонные события сценария (что должны найти rules/temporal/fusion)."""
    gt = SceneScript(scenario, duration).gt
    if gt.verdict != "violation":
        return []
    return [Event(type_id=gt.type_id or "V99", t_start=gt.t_start or 0.0, t_end=gt.t_end or 0.0,
                  t_peak=gt.t or 0.0, confidence=0.9, source="testkit", facts=[gt.description],
                  extra={"scenario": scenario})]


def make_catalog() -> dict:
    """Каталог-заглушка той же формы, что norms/catalog.yaml, но БЕЗ норм (rules: []) — только для тестов."""
    types = {}
    for tid, name in TYPE_NAMES.items():
        types[tid] = {
            "name": name,
            "observable": [],
            "not_observable": ["наличие допуска у лица", "личность человека", "умысел"],
            "description_template": "На {t} с: {facts}. Признаки возможного нарушения — требуется проверка человеком.",
            "severity": "high" if tid in ("V01", "V02", "V03", "V07") else "medium",
            "rules": [],
        }
    return {"version": 1, "_testkit_stub": True, "types": types}


_FALLBACK_B64: Optional[str] = None


def _fallback_jpeg_b64() -> str:
    global _FALLBACK_B64
    if _FALLBACK_B64 is None:
        ex = Path(__file__).resolve().parents[1] / "contracts" / "examples" / "job_done_tz.json"
        _FALLBACK_B64 = json.loads(ex.read_text(encoding="utf-8"))["violations"][0]["evidence_frames"][0][
            "image_base64"]
    return _FALLBACK_B64


def make_evidence_b64(scenario: str, t: float, duration: float = 20.0, size: tuple[int, int] = (232, 424),
                      quality: int = 70) -> str:
    """JPEG base64 синтетического кадра сценария в момент ``t`` (без cv2 — крошечный запасной JPEG)."""
    try:
        import cv2  # noqa: F401
    except ImportError:
        return _fallback_jpeg_b64()
    import cv2

    from .synth_video import render_snapshot

    img = render_snapshot(SceneScript(scenario, duration, size).at(t), size)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else _fallback_jpeg_b64()


def make_analysis_result(scenario: str, job_id: str = "testkit-job", duration: float = 20.0,
                         with_images: bool = True) -> AnalysisResult:
    """Готовый ``AnalysisResult`` «как от fusion» для тестов API/UI/eval/сериализации."""
    gt = SceneScript(scenario, duration).gt
    meta: dict[str, Any] = {
        "duration_sec": duration,
        "completeness": Completeness(complete=True, analyzed_until_sec=duration, duration_sec=duration,
                                     stages={"testkit": "mock"}).to_dict(),
        "provenance": Provenance(pipeline_version="testkit", mock=True, sources=["testkit"]).to_dict(),
        "warnings": [],
    }
    if gt.verdict != "violation":
        return AnalysisResult(job_id=job_id, status="done", verdict="no_violation", meta=meta,
                              timeline=[(t, 0.05) for t in sample_times(duration, 1.0)])
    t = gt.t or 0.0
    frames = [EvidenceFrame(timestamp_sec=t, image_base64=make_evidence_b64(scenario, t, duration)
                            if with_images else _fallback_jpeg_b64())]
    name = TYPE_NAMES.get(gt.type_id or "V99", "Возможное нарушение")
    v = Violation(type=name, timecode_sec=t, description=f"На {t:.1f} с: {gt.description}. Признаки возможного "
                  f"нарушения — требуется проверка человеком.", evidence_frames=frames, type_id=gt.type_id or "V99",
                  confidence=0.9, t_start=gt.t_start or t, t_end=gt.t_end or t, rules=[])
    timeline = [(x, 0.9 if (gt.t_start or 0) <= x <= (gt.t_end or 0) else 0.05) for x in sample_times(duration, 1.0)]
    return AnalysisResult(job_id=job_id, status="done", verdict="violation", violations=[v], timeline=timeline,
                          meta=meta)


@dataclass
class ScenarioBundle:
    scenario: str
    gt: dict
    zones: list[Zone]
    detections: list[Detection]
    tracks: list[Track]
    poses: list[PoseFrame]
    states: list[SceneState]
    events: list[Event]
    catalog: dict = field(default_factory=make_catalog)


def make_bundle(scenario: str, duration: float = 20.0, sample_fps: float = 5.0, *, view: str = "default",
                size: tuple[int, int] = DEFAULT_SIZE, noise: float = 0.0, seed: int = 0) -> ScenarioBundle:
    """Всё сразу: GT, зоны, детекции (с track_id), треки, позы, сигналы, эталонные события, каталог-заглушка."""
    gt = SceneScript(scenario, duration, size).gt.as_dict()
    assert set(LABELS) >= {"person", "cash", "cash_bag", "safe", "safe_open", "door", "door_open"}
    return ScenarioBundle(
        scenario=scenario, gt=gt, zones=make_zones(view, size),
        detections=make_detections(scenario, duration, sample_fps, view=view, size=size, tracked=True, noise=noise,
                                   seed=seed),
        tracks=make_tracks(scenario, duration, sample_fps, view=view, size=size),
        poses=make_pose_frames(scenario, duration, sample_fps, view=view, size=size),
        states=make_scene_states(scenario, duration, sample_fps, noise=noise, seed=seed),
        events=make_expected_events(scenario, duration),
    )
