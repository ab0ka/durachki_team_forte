"""Роли людей БЕЗ лиц: cashier / collector / visitor / unknown, role_conf и provenance.

Принципы (важно для A08/A09):
* Кассир выбирается по НЕЗАВИСИМЫМ от зон признакам: близость к объектам стола (desk строится
  из детекций предметов, не людей), присутствие с начала, неподвижность, сторона сейфа.
  cashier_area строится ПОСЛЕ этого из позиций кассира и для повторного «назначения кассиром»
  сама по себе не используется (нет цикла «кассир, потому что в зоне кассира»).
* Без опор-предметов роль кассира — fallback (самый неподвижный, присутствующий с начала),
  role_conf ограничен fallback_cashier_conf_cap.
* non-cashier ≠ «посторонний/неуполномоченный». Второй человек в зоне кассира получает
  second_person_role (по умолчанию unknown) — у него может быть допуск.
* Сумка сама по себе роль не подтверждает: collector = контакт с сумкой + (пришёл в ходе ролика
  или сумка появилась вместе с ним). Только контакт с сумкой -> unknown + причина.
* Возврат «на то же место» без подтверждения внешностью -> unknown (possible_cashier_return).
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Optional, Sequence

import numpy as np

from .appearance import bhattacharyya
from .contract import Zone
from .geometry import box_center, dist_point_polygon, iou, iso_dist, point_in_polygon, polygon_centroid, scale_box
from .views import TrackView


def person_features(v: TrackView, desk: Optional[Zone], safe: Optional[Zone], cfg: dict, aspect: float,
                    t0: float, t1: float, duration: float) -> dict[str, Any]:
    rc = cfg["roles"]
    f: dict[str, Any] = {"presence_s": round(v.presence, 3)}
    f["start"] = 1.0 if v.t_start <= t0 + rc["early_s"] else 0.0
    f["end"] = 1.0 if v.t_end >= t1 - rc["early_s"] else 0.0
    if len(v.times) > 1:
        dt = np.maximum(np.diff(v.times), 1e-3)
        dd = np.hypot(np.diff(v.pts[:, 0]) * aspect, np.diff(v.pts[:, 1])) / v.diag
        med_speed = float(np.median(dd / dt))
    else:
        med_speed = 0.0
    f["speed_diag_s"] = round(med_speed, 4)
    f["still"] = math.exp(-med_speed / max(rc["stationary_speed_diag_s"], 1e-6))
    dur = max(duration, 1e-6)
    f["presence_frac"] = min(1.0, v.presence / dur)
    if desk is not None:
        near = np.array([dist_point_polygon(p, desk.polygon, aspect) <= rc["near_desk_diag"] * v.diag for p in v.pts])
        f["near_desk_s"] = round(float(v.w[near].sum()), 3)
        f["desk_frac"] = min(1.0, float(v.w[near].sum()) / dur)
    else:
        f["desk_frac"] = None
    if safe is not None:
        sc = polygon_centroid(safe.polygon)
        if desk is not None:
            dc = polygon_centroid(desk.polygon)
            mp = v.median_pt
            proj = (mp[0] - dc[0]) * (sc[0] - dc[0]) * aspect ** 2 + (mp[1] - dc[1]) * (sc[1] - dc[1])
            f["safe_side"] = 1.0 if proj > 0 else 0.0
        else:
            f["safe_side"] = 1.0 - min(1.0, iso_dist(v.median_pt, sc, aspect) / (3.0 * v.diag))
    else:
        f["safe_side"] = None
    return f


def cashier_score(f: dict[str, Any], cfg: dict) -> float:
    rc = cfg["roles"]
    terms = [(rc["w_start"], f["start"]), (rc["w_still"], f["still"]), (rc["w_presence"], f["presence_frac"])]
    if f.get("desk_frac") is not None:
        terms.append((rc["w_desk"], f["desk_frac"]))
    else:  # без стола его вес уходит в «присутствие»
        terms.append((rc["w_desk"], f["presence_frac"]))
    if f.get("safe_side") is not None:
        terms.append((rc["w_safe_side"], f["safe_side"]))
    wsum = sum(w for w, _ in terms)
    return float(sum(w * x for w, x in terms) / max(wsum, 1e-9))


def pick_cashier(views: Sequence[TrackView], desk: Optional[Zone], safe: Optional[Zone], cfg: dict,
                 aspect: float, t0: float, t1: float, duration: float) -> dict[str, Any]:
    rc = cfg["roles"]
    feats = {v.tid: person_features(v, desk, safe, cfg, aspect, t0, t1, duration) for v in views}
    scores = {tid: cashier_score(f, cfg) for tid, f in feats.items()}
    cands = [v for v in views if feats[v.tid]["presence_s"] >= rc["min_presence_s"]
             and scores[v.tid] >= rc["min_cashier_score"]]
    res: dict[str, Any] = {"primary": None, "conf": 0.0, "independent": False, "features": feats,
                           "scores": {k: round(s, 4) for k, s in scores.items()}}
    if not cands:
        res["reason"] = "no_candidate_above_threshold"
        return res
    cands.sort(key=lambda v: -scores[v.tid])
    best = cands[0]
    s1 = scores[best.tid]
    conf = s1
    if len(cands) > 1:
        margin = s1 - scores[cands[1].tid]
        if margin < 0.15:  # два похожих кандидата — роль менее уверенна
            conf *= 0.6 + 0.4 * margin / 0.15
            res["ambiguous_with"] = cands[1].tid
    independent = desk is not None
    if not independent:
        conf = min(conf, rc["fallback_cashier_conf_cap"])
    support = []
    if feats[best.tid].get("desk_frac"):
        support.append("near_desk_objects")
    if feats[best.tid]["start"]:
        support.append("present_from_start")
    if feats[best.tid]["still"] > 0.5:
        support.append("stationary")
    if feats[best.tid].get("safe_side"):
        support.append("safe_side")
    if not independent:
        support.append("fallback_no_desk_objects")
    res.update(primary=best.tid, conf=round(float(conf), 3), independent=independent, support=support)
    return res


def _frac_inside(v: TrackView, zone: Optional[Zone]) -> float:
    if zone is None or len(v.pts) == 0:
        return 0.0
    ins = np.array([point_in_polygon(p, zone.polygon) for p in v.pts])
    return float(v.w[ins].sum() / max(v.w.sum(), 1e-9))


def _bag_index(objs: Sequence[TrackView], cfg: dict) -> tuple[dict, list]:
    rc = cfg["roles"]
    by_t: dict[float, list] = defaultdict(list)
    tracks = []
    for o in objs:
        if o.label == "cash_bag":
            wgt = 1.0
        elif o.label == "bag":
            wgt = float(rc["generic_bag_weight"])
        else:
            continue
        for t, b in zip(o.times, o.boxes):
            by_t[round(float(t), 3)].append((b, wgt))
        tracks.append((o, wgt))
    return by_t, tracks


def _near(person_box: Sequence[float], bag_box: Sequence[float], expand: float) -> bool:
    pb = scale_box(person_box, expand)
    c = box_center(bag_box)
    return (pb[0] <= c[0] <= pb[2] and pb[1] <= c[1] <= pb[3]) or iou(person_box, bag_box) > 0.0


def bag_contact(v: TrackView, bag_by_t: dict, bag_tracks: list, cfg: dict, t0: float) -> tuple[float, bool]:
    rc = cfg["roles"]
    contact = 0.0
    for t, b, w in zip(v.times, v.boxes, v.w):
        best = 0.0
        for bb, wgt in bag_by_t.get(round(float(t), 3), ()):
            if wgt > best and _near(b, bb, rc["bag_near_expand"]):
                best = wgt
        contact += float(w) * best
    arrived = False
    win = rc["bag_arrival_window_s"]
    for o, _ in bag_tracks:
        if o.t_start <= t0 + rc["early_s"] or abs(o.t_start - v.t_start) > win:
            continue
        j = int(np.argmin(np.abs(v.times - o.t_start)))
        if abs(v.times[j] - o.t_start) <= win and _near(v.boxes[j], o.boxes[0], rc["bag_near_expand"] * 1.3):
            arrived = True
            break
    return contact, arrived


def assign_roles(views: Sequence[TrackView], objs: Sequence[TrackView], zones: dict[str, Zone],
                 cashier: dict[str, Any], app: dict[int, np.ndarray], cfg: dict, aspect: float,
                 t0: float, t1: float) -> dict[int, dict[str, Any]]:
    rc = cfg["roles"]
    out: dict[int, dict[str, Any]] = {}
    by = {v.tid: v for v in views}
    primary = cashier.get("primary")
    pv = by.get(primary) if primary is not None else None
    ca, cl = zones.get("cashier_area"), zones.get("client_area")
    bag_by_t, bag_tracks = _bag_index(objs, cfg)
    cashiers: list[TrackView] = []
    if pv is not None:
        out[pv.tid] = {"role": "cashier", "conf": float(cashier["conf"]), "reasons": list(cashier.get("support", []))}
        cashiers.append(pv)

    def overlap_with_cashiers(v: TrackView) -> float:
        return max([v.overlap_s(c) for c in cashiers] or [0.0])

    for v in sorted(views, key=lambda x: (x.t_start, x.tid)):
        if v.tid in out:
            continue
        if v.presence < rc["min_presence_s"]:
            out[v.tid] = {"role": "unknown", "conf": 0.2, "reasons": ["too_short"]}
            continue
        in_ca, in_cl = _frac_inside(v, ca), _frac_inside(v, cl)
        entered = v.t_start > t0 + rc["early_s"]
        ov = overlap_with_cashiers(v)
        info: dict[str, Any] = {"in_cashier_area": round(in_ca, 3), "in_client_area": round(in_cl, 3),
                                "entered_mid_clip": entered, "overlap_with_cashier_s": round(ov, 3)}
        # 1) кассир вернулся / смена — только без пересечения во времени с кассиром
        if pv is not None and ov <= rc["overlap_tol_s"] and in_ca >= rc["return_min_in_area"]:
            ad = bhattacharyya(app.get(v.tid), app.get(pv.tid))
            same_seat = iso_dist(v.median_pt, pv.median_pt, aspect) / max(pv.diag, 1e-6) <= rc["return_same_seat_diag"]
            after = v.t_start >= pv.t_end - rc["overlap_tol_s"]
            long_stay = after and v.presence >= rc["shift_min_s"] and v.t_end >= t1 - rc["early_s"]
            info.update(app_dist_to_cashier=None if ad is None else round(ad, 3), same_seat=same_seat)
            if ad is not None and ad <= rc["app_same_thr"]:
                out[v.tid] = {"role": "cashier", "conf": round(min(float(cashier["conf"]), 0.85) * (1 - 0.5 * ad), 3),
                              "reasons": ["cashier_return", "clothes_appearance_match"], **info}
                cashiers.append(v)
                continue
            if ad is not None and ad < rc["app_diff_thr"] and same_seat:
                out[v.tid] = {"role": "cashier", "conf": 0.5,
                              "reasons": ["cashier_return", "clothes_appearance_partial", "same_seat"], **info}
                cashiers.append(v)
                continue
            if ad is not None and ad >= rc["app_diff_thr"]:
                if long_stay:
                    out[v.tid] = {"role": "unknown", "conf": 0.45,
                                  "reasons": ["different_appearance", "possible_shift_change"], **info}
                    continue
                info["note"] = "different_appearance_in_cashier_area"
            elif same_seat:
                out[v.tid] = {"role": "unknown", "conf": float(rc["position_only_return_conf"]),
                              "reasons": ["possible_cashier_return", "position_only_no_appearance"], **info}
                continue
        # 2) инкассатор: сумка + независимый признак прихода
        contact, arrived = bag_contact(v, bag_by_t, bag_tracks, cfg, t0)
        info.update(bag_contact_s=round(contact, 3), bag_arrived_with_person=arrived)
        if contact > 0:
            score = (rc["w_bag"] * min(1.0, contact / rc["bag_contact_full_s"]) + rc["w_entered"] * float(entered)
                     + rc["w_bag_arrived"] * float(arrived))
            if score >= rc["collector_thr"] and (entered or arrived):
                reasons = ["bag_contact"] + (["entered_mid_clip"] if entered else []) + (
                    ["bag_arrived_with_person"] if arrived else [])
                out[v.tid] = {"role": "collector", "conf": round(min(rc["collector_conf_cap"], score), 3),
                              "reasons": reasons, **info}
                continue
            out[v.tid] = {"role": "unknown", "conf": 0.35, "reasons": ["bag_contact_only"], **info}
            continue
        # 3) второй человек в зоне кассира одновременно с кассиром — НЕ обязательно посторонний
        if pv is not None and ov > rc["overlap_tol_s"] and in_ca >= 0.3:
            out[v.tid] = {"role": rc["second_person_role"], "conf": 0.5,
                          "reasons": ["co_present_in_cashier_area"], **info}
            continue
        if in_ca >= 0.3:
            out[v.tid] = {"role": "unknown", "conf": 0.4, "reasons": ["in_cashier_area_without_cashier"], **info}
            continue
        # 4) посетитель (клиент)
        conf = min(0.85, 0.5 + 0.3 * in_cl + 0.05 * float(entered))
        reasons = ["not_cashier"] + (["in_client_area"] if in_cl > 0.3 else [])
        out[v.tid] = {"role": "visitor", "conf": round(conf, 3), "reasons": reasons, **info}

    # спутник инкассатора (пришли вместе) — без своей сумки роль не подтверждаем
    collectors = [by[t] for t, r in out.items() if r["role"] == "collector"]
    for v in views:
        r = out.get(v.tid)
        if r is None or r["role"] not in ("visitor",) or not r.get("entered_mid_clip"):
            continue
        for c in collectors:
            if abs(v.t_start - c.t_start) <= rc["companion_window_s"] and \
                    iso_dist(v.pts[0], c.pts[0], aspect) / max(c.diag, 1e-6) <= rc["companion_dist_diag"]:
                r.update(role="unknown", conf=0.4, reasons=r["reasons"] + ["arrived_with_collector"])
                break
    return out
