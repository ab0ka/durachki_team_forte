"""pose_signals: позы → SceneState с семантикой unknown.

Ключ сигнала присутствует в ``signals`` только если сигнал определим в момент t:
  * значение ≥ 0.5 — сигнал активен (после гистерезиса и требований к длительности);
  * значение < 0.5 — наблюдаемое «нет» (нужные точки видны, предмет-подтверждение есть или не нужен);
  * ключа нет — unknown (окклюзия, низкая уверенность, нет нужного предмета, нет кассира в кадре).

Жест без предмета не превращается в предметный вывод: «рука у уха» без cell_phone, «кисть ко рту» без
cup/bottle/food, «кисти не видны» — пишутся только в ``extra["gestures"]`` как подсказка, не как сигнал.
Предметы берутся ТОЛЬКО из аргумента ``tracks`` (треки с label из LABELS, например cell_phone, cup).
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from ._contract import PoseFrame, PoseSceneState, Track, Zone
from .config import load_config
from .filters import gated_values, ramp, runs_from_mask, smooth_keypoint_track
from .keypoints import (
    L_KNEE, L_SHOULDER, L_WRIST, R_KNEE, R_SHOULDER, R_WRIST, WRISTS,
    PersonGeom, bbox_to_iso, kp_array, point_box_dist, point_in_polygon, point_polygon_dist,
    polygon_centroid, polygon_iso, to_iso, track_scale,
)

POSE_SIGNALS = ("phone_to_ear", "phone_in_hand", "eating_drinking", "hands_hidden",
                "cashier_facing_away", "handover_motion")
STAFF_SIGNALS = ("phone_to_ear", "phone_in_hand", "eating_drinking", "hands_hidden", "cashier_facing_away")
_TKEY = 1000.0  # ключ времени: round(t * _TKEY)


def _tkey(t: float) -> int:
    return int(round(float(t) * _TKEY))


@dataclass
class _Series:
    """Временной ряд позы одного трека."""

    tid: int
    role: str
    times: np.ndarray
    kps: np.ndarray                          # (T, 17, 3) нормированные
    bboxes: list[Optional[tuple[float, float, float, float]]]
    iso: np.ndarray = field(default_factory=lambda: np.zeros((0, 17, 3)))
    scale: float = 0.08
    geoms: list[PersonGeom] = field(default_factory=list)
    values: dict[str, list[Optional[float]]] = field(default_factory=dict)
    gestures: dict[str, list[Optional[float]]] = field(default_factory=dict)
    extras: dict[str, list[Optional[float]]] = field(default_factory=dict)


class _ObjectIndex:
    """Объект-треки (не person) из аргумента tracks: поиск bbox-ов по метке и времени."""

    def __init__(self, tracks: Sequence[Track], aspect: float) -> None:
        self.by_label: dict[str, tuple[np.ndarray, list[np.ndarray]]] = {}
        tmp: dict[str, list[tuple[float, np.ndarray]]] = defaultdict(list)
        for tr in tracks or []:
            if tr.label == "person":
                continue
            for t, bb in tr.boxes:
                tmp[tr.label].append((float(t), bbox_to_iso(bb, aspect)))
        for label, items in tmp.items():
            items.sort(key=lambda x: x[0])
            self.by_label[label] = (np.array([x[0] for x in items]), [x[1] for x in items])

    def has_any(self, labels: Sequence[str]) -> bool:
        return any(lb in self.by_label for lb in labels)

    def boxes(self, labels: Sequence[str], t0: float, t1: float) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for lb in labels:
            if lb not in self.by_label:
                continue
            ts, bxs = self.by_label[lb]
            i0 = int(np.searchsorted(ts, t0, side="left"))
            i1 = int(np.searchsorted(ts, t1, side="right"))
            out.extend(bxs[i0:i1])
        return out


def _aspect_from_poses(poses: Sequence[PoseFrame], default: float) -> float:
    for p in poses:
        wh = (getattr(p, "extra", None) or {}).get("frame_wh")
        if wh and len(wh) == 2 and wh[0] and wh[1]:
            return float(wh[0]) / float(wh[1])
    return float(default)


def _box_at(track: Optional[Track], t: float, tol: float) -> Optional[tuple[float, float, float, float]]:
    if track is None or not track.boxes:
        return None
    best, best_dt = None, tol
    for bt, bb in track.boxes:
        dt = abs(float(bt) - t)
        if dt <= best_dt:
            best, best_dt = bb, dt
    return tuple(float(v) for v in best) if best is not None else None


def _assign_pseudo_ids(poses: Sequence[PoseFrame], aspect: float) -> list[int]:
    """track_id=None → псевдо-id (отрицательные) жадной ассоциацией по центру точек."""
    ids: list[int] = []
    active: dict[int, tuple[float, np.ndarray]] = {}
    next_id = -1
    for p in poses:
        if p.track_id is not None:
            ids.append(int(p.track_id))
            continue
        k = kp_array(p.keypoints)
        vis = k[:, 2] > 0.1
        c = (to_iso(k[vis], aspect)[:, :2].mean(axis=0) if vis.any() else np.array([0.5 * aspect, 0.5]))
        best, best_d = None, 0.08
        for pid, (lt, lc) in active.items():
            if 0 < p.t - lt <= 1.0:
                d = float(np.linalg.norm(c - lc))
                if d < best_d:
                    best, best_d = pid, d
        if best is None:
            best = next_id
            next_id -= 1
        active[best] = (float(p.t), c)
        ids.append(best)
    return ids


def _build_series(poses: Sequence[PoseFrame], tracks: Sequence[Track], aspect: float,
                  cfg: dict[str, Any]) -> list[_Series]:
    person_tracks = {int(t.track_id): t for t in tracks or [] if t.label == "person"}
    order = sorted(range(len(poses)), key=lambda i: (poses[i].t, -1 if poses[i].track_id is None else poses[i].track_id))
    poses_sorted = [poses[i] for i in order]
    ids = _assign_pseudo_ids(poses_sorted, aspect)
    tol = float(cfg["selection"].get("box_time_tolerance_s", 0.25))
    groups: dict[int, list[PoseFrame]] = defaultdict(list)
    for tid, p in zip(ids, poses_sorted):
        groups[tid].append(p)
    smooth_cfg = cfg.get("smoothing", {}) or {}
    sc = cfg["signals"]
    out: list[_Series] = []
    for tid, plist in groups.items():
        # дубликаты по времени (один трек, один t) — берём с большей средней уверенностью
        uniq: dict[int, PoseFrame] = {}
        for p in plist:
            key = _tkey(p.t)
            if key not in uniq or kp_array(p.keypoints)[:, 2].mean() > kp_array(uniq[key].keypoints)[:, 2].mean():
                uniq[key] = p
        plist = [uniq[k] for k in sorted(uniq)]
        times = np.array([float(p.t) for p in plist], dtype=np.float64)
        kps = np.stack([kp_array(p.keypoints) for p in plist]) if plist else np.zeros((0, 17, 3), np.float32)
        already = all(bool((getattr(p, "extra", None) or {}).get("smoothed")) for p in plist)
        if smooth_cfg.get("enabled", True) and not already and len(plist) > 1:
            kps = smooth_keypoint_track(times, kps, smooth_cfg)
        track = person_tracks.get(tid)
        bboxes = []
        for p in plist:
            bb = (getattr(p, "extra", None) or {}).get("bbox")
            bboxes.append(tuple(float(v) for v in bb) if bb else _box_at(track, float(p.t), tol))
        s = _Series(tid=tid, role=(track.role if track is not None else "unknown"), times=times, kps=kps, bboxes=bboxes)
        s.iso = to_iso(kps, aspect)
        min_score = float(sc.get("kp_min_score", 0.3))
        s.scale = track_scale(list(s.iso), [bbox_to_iso(b, aspect) if b else None for b in bboxes],
                              sc.get("scale", {}) or {}, min_score)
        s.geoms = [PersonGeom(s.iso[j], s.scale, min_score) for j in range(len(times))]
        out.append(s)
    return out


# ---------------------------------------------------------------- per-signal evidence


def _apply_gate(series: _Series, name: str, ev: Sequence[Optional[float]], p: dict[str, Any]) -> None:
    """Гистерезис по известным кадрам; unknown внутри активного отрезка (≤ max_gap) — активны."""
    n = len(series.times)
    known = np.array([e is not None for e in ev], dtype=bool)
    out: list[Optional[float]] = [None] * n
    if known.any():
        idx = np.flatnonzero(known)
        vals, mask = gated_values(series.times[idx], [ev[i] for i in idx], p)
        for k, i in enumerate(idx):
            out[i] = float(vals[k])
        for a, b in runs_from_mask(mask):
            t_a, t_b = series.times[idx[a]], series.times[idx[b]]
            fill = float(vals[a:b + 1].mean())
            for j in range(n):
                if out[j] is None and t_a <= series.times[j] <= t_b:
                    out[j] = fill
    series.values[name] = out


def _phone_to_ear(series: _Series, objs: _ObjectIndex, cfg: dict[str, Any]) -> None:
    sc = cfg["signals"]
    p = sc["phone_to_ear"]
    tol = float(sc.get("obj_time_tolerance_s", 0.5))
    n = len(series.times)
    gesture: list[Optional[float]] = [None] * n
    n_vis = [0] * n
    phone_near = [False] * n
    for j, g in enumerate(series.geoms):
        heads = g.ears()
        wrists = g.wrists()
        n_vis[j] = len(wrists)
        if not heads or not wrists:
            continue
        mouth = g.mouth()
        vals = []
        for _, w in wrists:
            d_ear = min(g.dist(w, e) for e in heads)
            e = ramp(d_ear, float(p["near"]), float(p["far"]))
            if mouth is not None and g.dist(w, mouth) < float(p["mouth_pref_ratio"]) * d_ear:
                e *= float(p["mouth_penalty"])
            vals.append(e)
        best = max(vals)
        if len(vals) == 2 and min(vals) > 0.5:
            best *= float(p["both_hands_penalty"])
        gesture[j] = best
        t = float(series.times[j])
        head = g.head()
        for box in objs.boxes(sc.get("phone_labels", ["cell_phone"]), t - tol, t + tol):
            cands = [w for _, w in wrists] + ([head] if head is not None else [])
            if cands and min(point_box_dist(c, box) for c in cands) / g.s <= float(p["phone_near"]):
                phone_near[j] = True
                break
    # отрезки жеста «рука у головы» (гистерезис + ≥ min_duration_s по известным кадрам)
    known = [i for i in range(n) if gesture[i] is not None]
    out: list[Optional[float]] = [None] * n
    gest_out: list[Optional[float]] = [None] * n
    if known:
        times_k = series.times[known]
        vals, mask = gated_values(times_k, [gesture[i] for i in known], p)
        in_run = np.zeros(n, dtype=bool)
        for a, b in runs_from_mask(mask):
            run_idx = [known[k] for k in range(a, b + 1)]
            t_a, t_b = series.times[run_idx[0]], series.times[run_idx[-1]]
            span = [j for j in range(n) if t_a <= series.times[j] <= t_b]
            confirmed = any(phone_near[j] for j in span)
            fill = float(vals[a:b + 1].mean())
            for j in span:
                in_run[j] = True
                gest_out[j] = fill
                if confirmed:  # предмет подтверждён — сигнал; иначе только жест (unknown)
                    out[j] = float(vals[known.index(j)]) if j in known else fill
        for k, i in enumerate(known):
            if not in_run[i]:
                gest_out[i] = float(vals[k])
                if n_vis[i] == 2 and gesture[i] <= float(p["gesture_neg"]):
                    out[i] = float(vals[k])  # обе кисти видны и далеко от головы — наблюдаемое «нет»
    series.values["phone_to_ear"] = out
    series.gestures["hand_at_head"] = gest_out


def _phone_in_hand(series: _Series, objs: _ObjectIndex, cfg: dict[str, Any]) -> None:
    sc = cfg["signals"]
    p = sc["phone_in_hand"]
    tol = float(sc.get("obj_time_tolerance_s", 0.5))
    ev: list[Optional[float]] = [None] * len(series.times)
    for j, g in enumerate(series.geoms):
        t = float(series.times[j])
        phones = objs.boxes(sc.get("phone_labels", ["cell_phone"]), t - tol, t + tol)
        if not phones:
            continue  # нет телефона в кадре — unknown (детектор мог не увидеть)
        wrists = g.wrists()
        center = g.body_center()
        far_from_body = center is not None and all(
            point_box_dist(center, b) / g.s >= float(p["body_far"]) for b in phones)
        if wrists:
            d = min(point_box_dist(w, b) for _, w in wrists for b in phones) / g.s
            e = ramp(d, float(p["near"]), float(p["far"]))
            if e >= float(p["thr_off"]) or len(wrists) == 2 or far_from_body:
                ev[j] = e
        elif far_from_body:
            ev[j] = 0.0
    _apply_gate(series, "phone_in_hand", ev, p)


def _eating(series: _Series, objs: _ObjectIndex, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    sc = cfg["signals"]
    p = sc["eating_drinking"]
    tol = float(sc.get("obj_time_tolerance_s", 0.5))
    near, far = float(p["near"]), float(p["far"])
    mid = 0.5 * (near + far)
    n = len(series.times)
    cycles: list[dict[str, Any]] = []
    for widx in WRISTS:
        last_far: Optional[tuple[float, np.ndarray, Optional[np.ndarray]]] = None
        cur: Optional[dict[str, Any]] = None
        last_t = None
        for j, g in enumerate(series.geoms):
            t = float(series.times[j])
            mouth = g.mouth()
            if not g.ok(widx) or mouth is None:
                continue
            if last_t is not None and t - last_t > float(p["max_approach_s"]):
                last_far, cur = None, None  # длинный пропуск — сброс
            last_t = t
            w = g.xy(widx)
            d = g.dist(w, mouth)
            head = g.head()
            if cur is None:
                if d >= far:
                    last_far = (t, w.copy(), None if head is None else head.copy())
                elif d <= near and last_far is not None and t - last_far[0] <= float(p["max_approach_s"]):
                    travel_w = g.dist(w, last_far[1])
                    travel_h = g.dist(head, last_far[2]) if (head is not None and last_far[2] is not None) else 0.0
                    if travel_w >= float(p["min_wrist_travel"]) and travel_w >= float(p["wrist_vs_head_ratio"]) * travel_h:
                        cur = {"wrist": widx, "t_start": last_far[0], "t_enter": t, "t_last_near": t}
            else:
                if d <= mid:
                    cur["t_last_near"] = t
                    if d <= near:
                        cur["t_last_near"] = t
                if d > mid:
                    dwell = cur["t_last_near"] - cur["t_enter"]
                    if float(p["min_dwell_s"]) - 1e-9 <= dwell + _period(series.times) <= float(p["max_dwell_s"]) + 1e-9:
                        cur["t_end"] = t
                        cycles.append(cur)
                    cur = None
                    last_far = (t, w.copy(), None if head is None else head.copy()) if d >= far else None
    # предмет у кисти/рта во время подхода
    food_labels = sc.get("food_labels", ["cup", "bottle", "food"])
    for c in cycles:
        c["object"] = False
        for j, g in enumerate(series.geoms):
            t = float(series.times[j])
            if not (c["t_start"] - tol <= t <= c["t_end"] + tol):
                continue
            pts = [w for _, w in g.wrists()]
            m = g.mouth()
            if m is not None:
                pts.append(m)
            for box in objs.boxes(food_labels, t - tol, t + tol):
                if pts and min(point_box_dist(q, box) for q in pts) / g.s <= float(p["object_near"]):
                    c["object"] = True
                    break
            if c["object"]:
                break
    cycles.sort(key=lambda c: c["t_enter"])
    clusters: list[list[dict[str, Any]]] = []
    for c in cycles:
        if clusters and c["t_enter"] - clusters[-1][-1]["t_enter"] <= float(p["window_s"]):
            clusters[-1].append(c)
        else:
            clusters.append([c])
    ev: list[Optional[float]] = [None] * n
    gest: list[Optional[float]] = [None] * n
    covered = np.zeros(n, dtype=bool)
    for cl in clusters:
        t0 = min(c["t_start"] for c in cl)
        t1 = max(c["t_end"] for c in cl)
        k = len(cl)
        has_obj = any(c["object"] for c in cl)
        val = None
        if has_obj:
            val = float(p["evidence_single_object"]) if k == 1 else min(
                1.0, float(p["evidence_multi_object"]) + float(p["evidence_per_extra_cycle"]) * (k - 2))
        g_val = min(1.0, 0.3 * k)
        for j in range(n):
            if t0 <= series.times[j] <= t1:
                covered[j] = True
                ev[j] = val  # без предмета — None (unknown): «кисть ко рту» ≠ еда
                gest[j] = max(gest[j] or 0.0, g_val)
    for j, g in enumerate(series.geoms):
        if covered[j]:
            continue
        m = g.mouth()
        wr = g.wrists()
        if m is not None and len(wr) == 2 and min(g.dist(w, m) for _, w in wr) > near:
            ev[j] = 0.0  # рот и обе кисти видны, кисти не у рта — наблюдаемое «нет»
    _apply_gate(series, "eating_drinking", ev, p)
    series.gestures["hand_to_mouth"] = gest
    return cycles


def _period(times: np.ndarray) -> float:
    if len(times) < 2:
        return 0.2
    d = np.diff(times)
    d = d[d > 1e-6]
    return float(np.median(d)) if len(d) else 0.2


def _hands_hidden(series: _Series, cfg: dict[str, Any]) -> None:
    p = cfg["signals"]["hands_hidden"]
    vis_thr = float(p["wrist_visible_score"])
    torso_thr = float(p["torso_min_score"])
    ev: list[Optional[float]] = [None] * len(series.times)
    gest: list[Optional[float]] = [None] * len(series.times)
    for j, g in enumerate(series.geoms):
        ws = (g.score(L_WRIST), g.score(R_WRIST))
        torso = min(g.score(L_SHOULDER), g.score(R_SHOULDER)) >= torso_thr
        if min(ws) >= vis_thr:
            ev[j] = 0.0  # обе кисти видны — наблюдаемое «руки не скрыты»
            gest[j] = 0.0
        elif torso and max(ws) < vis_thr:
            gest[j] = 1.0  # кисти не видны при видимом корпусе: подсказка, НЕ доказательство сокрытия
            if bool(p.get("allow_positive_from_pose", False)):
                ev[j] = 1.0
    _apply_gate(series, "hands_hidden", ev, {"thr_on": 0.6, "thr_off": 0.4, **p})
    series.gestures["hands_not_visible"] = gest


def _zones_by_name(zones: Sequence[Zone], aspect: float) -> dict[str, list[list[tuple[float, float]]]]:
    out: dict[str, list[list[tuple[float, float]]]] = defaultdict(list)
    for z in zones or []:
        if len(z.polygon) >= 3:
            out[z.name].append(polygon_iso(z.polygon, aspect))
    return out


def _facing_away(series: _Series, zones_iso: dict[str, list], cfg: dict[str, Any]) -> None:
    p = cfg["signals"]["cashier_facing_away"]
    refs = zones_iso.get("desk", []) or zones_iso.get("cash_tray", [])
    doors = zones_iso.get("door", [])
    ev: list[Optional[float]] = [None] * len(series.times)
    if not refs and bool(p.get("require_desk_zone", True)):
        series.values["cashier_facing_away"] = ev
        return
    for j, g in enumerate(series.geoms):
        f, fc = g.facing(float(p["nose_weight"]), float(p["min_nose_offset"]))
        c = g.body_center()
        if f is None or c is None or fc < float(p["min_facing_conf"]) or not refs:
            continue
        ref = min((polygon_centroid(r) for r in refs), key=lambda q: float(np.linalg.norm(q - c)))
        v = ref - c
        if float(np.linalg.norm(v)) / g.s < 0.2:
            continue
        cos = float(np.dot(f, v / np.linalg.norm(v)))
        e = ramp(cos, float(p["away_cos"]), float(p["toward_cos"]))
        for d in doors:
            vd = polygon_centroid(d) - c
            if float(np.linalg.norm(vd)) > 1e-6:
                cos_d = float(np.dot(f, vd / np.linalg.norm(vd)))
                e = max(e, 0.8 * ramp(cos_d, 1.0, float(p["door_cos"])))
        ev[j] = e
    _apply_gate(series, "cashier_facing_away", ev, p)


def _bending_and_posture(series: _Series, zones_iso: dict[str, list], cfg: dict[str, Any]) -> None:
    sc = cfg["signals"]
    pb = sc["bending_to_safe"]
    pp = sc["posture"]
    safes = zones_iso.get("safe", [])
    n = len(series.times)
    head_hip = []
    for g in series.geoms:
        h, hm = g.head(), g.hip_mid()
        head_hip.append(g.dist(h, hm) if (h is not None and hm is not None) else None)
    valid = [v for v in head_hip if v is not None]
    med = float(np.median(valid)) if len(valid) >= 3 else None
    heights = [b[3] - b[1] for b in series.bboxes if b]
    h90 = float(np.percentile(heights, 90)) if len(heights) >= 3 else None
    ev: list[Optional[float]] = [None] * n
    sit: list[Optional[float]] = [None] * n
    for j, g in enumerate(series.geoms):
        if safes:
            pts = [w for _, w in g.wrists()]
            head = g.head()
            if pts or head is not None:
                margin = float(pb["zone_margin"])
                w_in = any(point_polygon_dist(w, s) <= margin for w in pts for s in safes)
                h_in = head is not None and any(point_polygon_dist(head, s) <= margin for s in safes)
                near_v = 1.0 if w_in else (0.7 if h_in else 0.0)
                bend = 0.5
                if med and head_hip[j] is not None:
                    bend = ramp(head_hip[j] / med, float(pb["bend_ratio"]), 1.0)
                ev[j] = near_v * (float(pb["base"]) + (1.0 - float(pb["base"])) * bend)
        # поза сидя/стоя (эвристика)
        sm, hm = g.shoulder_mid(), g.hip_mid()
        knees = [g.xy(k) for k in (L_KNEE, R_KNEE) if g.ok(k)]
        val, conf = None, 0.0
        if sm is not None and hm is not None and knees:
            torso = sm - hm
            thigh = np.mean(knees, axis=0) - hm
            nt, nh = float(np.linalg.norm(torso)), float(np.linalg.norm(thigh))
            if nt > 1e-6 and nh > 1e-6:
                ang = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(torso, -thigh)) / (nt * nh)))))
                val = ramp(ang, float(pp["sit_angle_hi_deg"]), float(pp["sit_angle_lo_deg"]))
                conf = 0.7
        elif h90 and series.bboxes[j]:
            b = series.bboxes[j]
            r = (b[3] - b[1]) / h90
            thr = float(pp["height_ratio_sit"])
            val = ramp(r, thr - 0.15, thr + 0.1)
            conf = 0.4
        if val is not None and conf >= float(pp["min_conf"]):
            sit[j] = val
    if safes:
        _apply_gate(series, "_bending_to_safe", ev, pb)
        series.extras["bending_to_safe"] = series.values.pop("_bending_to_safe")
    else:
        series.extras["bending_to_safe"] = [None] * n
    series.extras["sitting"] = sit


def _handover(series_list: list[_Series], zones_iso: dict[str, list], cfg: dict[str, Any]
              ) -> dict[tuple[int, int], tuple[np.ndarray, list[Optional[float]]]]:
    p = cfg["signals"]["handover_motion"]
    zone_polys = [poly for name in p.get("zone_names", []) for poly in zones_iso.get(name, [])]
    by_time: dict[int, list[tuple[_Series, int]]] = defaultdict(list)
    for s in series_list:
        for j, t in enumerate(s.times):
            by_time[_tkey(t)].append((s, j))
    pair_ev: dict[tuple[int, int], dict[int, Optional[float]]] = defaultdict(dict)
    for key, items in by_time.items():
        for ia in range(len(items)):
            for ib in range(ia + 1, len(items)):
                (sa, ja), (sb, jb) = items[ia], items[ib]
                if sa.tid == sb.tid:
                    continue
                ga, gb = sa.geoms[ja], sb.geoms[jb]
                pair = (min(sa.tid, sb.tid), max(sa.tid, sb.tid))
                ca, cb = ga.body_center(), gb.body_center()
                wa, wb = ga.wrists(), gb.wrists()
                if ca is None or cb is None or not wa or not wb:
                    pair_ev[pair][key] = None
                    continue
                s_mean = 0.5 * (ga.s + gb.s)
                if float(np.linalg.norm(ca - cb)) / s_mean < float(p["same_person_dist"]):
                    continue  # вероятно дубликат одного человека
                best, mid = math.inf, None
                for _, xa in wa:
                    if ga.dist(xa, ca) < float(p["min_reach"]):
                        continue
                    for _, xb in wb:
                        if gb.dist(xb, cb) < float(p["min_reach"]):
                            continue
                        d = float(np.linalg.norm(xa - xb)) / s_mean
                        if d < best:
                            best, mid = d, 0.5 * (xa + xb)
                e = 0.0 if not math.isfinite(best) else ramp(best, float(p["near"]), float(p["far"]))
                if e > 0 and zone_polys and mid is not None:
                    inside = any(point_polygon_dist(mid, z) <= float(p["zone_margin"]) for z in zone_polys)
                    if not inside:
                        e *= float(p["outside_zone_factor"])
                if e >= float(p["thr_off"]) or (len(wa) == 2 and len(wb) == 2):
                    pair_ev[pair][key] = e
                else:
                    pair_ev[pair][key] = None  # не все кисти видны — «нет передачи» не утверждаем
    out: dict[tuple[int, int], tuple[np.ndarray, list[Optional[float]]]] = {}
    for pair, d in pair_ev.items():
        keys = sorted(d)
        fake = _Series(tid=-999, role="unknown", times=np.array([k / _TKEY for k in keys]),
                       kps=np.zeros((0, 17, 3)), bboxes=[])
        _apply_gate(fake, "handover_motion", [d[k] for k in keys], p)
        out[pair] = (np.array(keys), fake.values["handover_motion"])
    return out


# ---------------------------------------------------------------- public


def compute_pose_signals(poses: Sequence[PoseFrame], tracks: Sequence[Track], zones: Sequence[Zone],
                         cfg: Optional[dict[str, Any]] = None) -> list[PoseSceneState]:
    """Реализация pose_signals с явным конфигом (для тестов/тюнинга)."""
    cfg = cfg or load_config()
    sc = cfg["signals"]
    poses = list(poses or [])
    tracks = list(tracks or [])
    aspect = _aspect_from_poses(poses, float(sc.get("default_aspect_wh", 0.5472)))
    objs = _ObjectIndex(tracks, aspect)
    zones_iso = _zones_by_name(zones, aspect)
    series_list = _build_series(poses, tracks, aspect, cfg)
    staff_roles = set(sc.get("staff_roles", ["cashier"]))
    for s in series_list:
        _phone_to_ear(s, objs, cfg)
        _phone_in_hand(s, objs, cfg)
        _eating(s, objs, cfg)
        _hands_hidden(s, cfg)
        _facing_away(s, zones_iso, cfg)
        _bending_and_posture(s, zones_iso, cfg)
    pairs = _handover(series_list, zones_iso, cfg)
    roles = {s.tid: s.role for s in series_list}

    # все моменты времени: позы + наблюдения person-треков (там, где позы нет, всё unknown)
    tkeys = {_tkey(p.t) for p in poses}
    for tr in tracks:
        if tr.label == "person":
            tkeys.update(_tkey(t) for t, _ in tr.boxes)
    index: dict[int, list[tuple[_Series, int]]] = defaultdict(list)
    for s in series_list:
        for j, t in enumerate(s.times):
            index[_tkey(t)].append((s, j))
    pair_index: dict[int, list[tuple[tuple[int, int], Optional[float]]]] = defaultdict(list)
    for pair, (keys, vals) in pairs.items():
        for k, v in zip(keys, vals):
            pair_index[int(k)].append((pair, v))

    from .status import status as _pose_status

    pose_status = _pose_status().get("state", "unknown")
    states: list[PoseSceneState] = []
    for key in sorted(tkeys):
        t = key / _TKEY
        signals: dict[str, float] = {}
        extra: dict[str, Any] = {"n_posed": len(index.get(key, [])), "pose_status": pose_status,
                                 "by_track": {}, "gestures": {}, "track_ids": {}, "boxes": {},
                                 "bending_to_safe": None, "sitting": None, "standing": None,
                                 "handover_pairs": []}
        staff_items = [(s, j) for s, j in index.get(key, []) if s.role in staff_roles]
        extra["staff_track_ids"] = [s.tid for s, _ in staff_items]
        for s, j in index.get(key, []):
            entry = {"role": s.role,
                     "signals": {nm: s.values[nm][j] for nm in STAFF_SIGNALS if s.values.get(nm) and s.values[nm][j] is not None},
                     "gestures": {gn: gv[j] for gn, gv in s.gestures.items() if gv[j] is not None},
                     "bending_to_safe": s.extras.get("bending_to_safe", [None] * (j + 1))[j],
                     "sitting": s.extras.get("sitting", [None] * (j + 1))[j],
                     "bbox": s.bboxes[j]}
            extra["by_track"][s.tid] = entry
        for nm in STAFF_SIGNALS:
            best, best_s = None, None
            for s, j in staff_items:
                v = s.values.get(nm, [None] * (j + 1))[j]
                if v is not None and (best is None or v > best):
                    best, best_s = v, (s, j)
            if best is not None:
                signals[nm] = round(float(best), 4)
                if best >= 0.5 and best_s is not None:
                    extra["track_ids"][nm] = [best_s[0].tid]
                    if best_s[0].bboxes[best_s[1]]:
                        extra["boxes"][nm] = best_s[0].bboxes[best_s[1]]
        for gn in ("hand_at_head", "hand_to_mouth", "hands_not_visible"):
            vals = [s.gestures.get(gn, [None] * (j + 1))[j] for s, j in staff_items]
            vals = [v for v in vals if v is not None]
            if vals:
                extra["gestures"][gn] = round(float(max(vals)), 4)
        bvals = [s.extras["bending_to_safe"][j] for s, j in staff_items if s.extras["bending_to_safe"][j] is not None]
        if bvals:
            extra["bending_to_safe"] = round(float(max(bvals)), 4)
        svals = [s.extras["sitting"][j] for s, j in staff_items if s.extras["sitting"][j] is not None]
        if svals:
            extra["sitting"] = round(float(svals[0]), 4)
            extra["standing"] = round(1.0 - float(svals[0]), 4)
        hv = [(pair, v) for pair, v in pair_index.get(key, []) if v is not None]
        if hv:
            pair, v = max(hv, key=lambda x: x[1])
            signals["handover_motion"] = round(float(v), 4)
            for pr, vv in hv:
                if vv >= 0.5:
                    extra["handover_pairs"].append([pr[0], pr[1], roles.get(pr[0], "unknown"),
                                                    roles.get(pr[1], "unknown"), round(float(vv), 4)])
            if v >= 0.5:
                extra["track_ids"]["handover_motion"] = [pair[0], pair[1]]
        extra["unknown"] = [nm for nm in POSE_SIGNALS if nm not in signals]
        states.append(PoseSceneState(t=t, signals=signals, extra=extra))
    return states


def pose_signals(poses: Sequence[PoseFrame], tracks: Sequence[Track], zones: Sequence[Zone]) -> list[PoseSceneState]:
    """Публичная функция контракта: ml.pose.pose_signals(poses, tracks, zones) -> list[SceneState]."""
    return compute_pose_signals(poses, tracks, zones)
