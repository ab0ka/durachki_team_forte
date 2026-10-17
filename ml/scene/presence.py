"""Сигналы присутствия с разделением «нет» / «неизвестно».

* Человек с неизвестной ролью ≠ посторонний: outsider подтверждается только явной ролью visitor от
  трекера или историей трека в client_area; иначе — observed.unknown_person_in_cashier_area + unknown.
* Пропавший трек ≠ ушёл: уход подтверждается только если трек был виден вне рабочего места или
  исчез у края кадра / в зоне двери. Трек, «растворившийся» посреди рабочего места — unknown.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Optional

import numpy as np

from .geometry import BBox, zone_membership
from .timeline import Timeline, sample_dt

Polys = list[list[tuple[float, float]]]


@dataclass
class Sig:
    p: np.ndarray
    c: np.ndarray
    prov: list[str] = field(default_factory=list)


@lru_cache(maxsize=1 << 16)
def _member_one(b: BBox, poly: tuple[tuple[float, float], ...], anchor_y: float, ofull: float) -> float:
    return zone_membership(b, poly, anchor_y, ofull)


def member(b: BBox, polys: Optional[Polys], anchor_y: float, ofull: float) -> float:
    if not polys:
        return 0.0
    bb = tuple(float(v) for v in b)
    return max(_member_one(bb, tuple((float(x), float(y)) for x, y in p), anchor_y, ofull) for p in polys)


def _near_exit(b: BBox, door: Optional[Polys], border: float, anchor_y: float, ofull: float) -> bool:
    if b[0] <= border or b[1] <= border or b[2] >= 1 - border or b[3] >= 1 - border:
        return True
    return bool(door) and member(b, door, anchor_y, ofull) > 0.3


def presence_signals(tl: Timeline, zones: dict[str, Polys], workplace: Optional[Polys], quality: np.ndarray, det_ran: np.ndarray, cfg: dict[str, Any]) -> tuple[dict[str, Sig], dict[str, np.ndarray]]:
    pc = cfg["presence"]
    ay, of = float(pc["anchor_y"]), float(pc["overlap_full"])
    border = float(pc.get("border_margin", 0.04))
    n = len(tl)
    dt = sample_dt(tl.times)
    door = zones.get("door")
    client = zones.get("client_area")
    q = np.clip(quality, 0.0, 1.0) * det_ran.astype(float)

    # --- статус каждого трека на каждом сэмпле ---
    tracks: dict[int, dict[int, Any]] = {}
    for i in range(n):
        for o in tl.persons[i]:
            if o.track_id is not None:
                tracks.setdefault(int(o.track_id), {})[i] = o
    has_tracks = bool(tracks)
    ws_occ = np.zeros(n)            # кто-то на рабочем месте (известно)
    ws_unknown = np.zeros(n)        # возможно на рабочем месте, но не видим (потерянный трек)
    cash_in = np.zeros(n)
    cash_conf = np.zeros(n)
    cashier_vis = np.zeros(n)
    cashier_gone = np.ones(n)       # все «кассироподобные» треки подтверждённо вне рабочего места
    cashier_absent_known = np.ones(n)
    track_info: dict[int, dict[str, Any]] = {}
    backfill_s = float(pc.get("appear_backfill_s", 1.5))
    # проход 1: вход/выход каждого трека (выход подтверждён у края кадра / в двери / вне рабочего места)
    geo: dict[int, dict[str, Any]] = {}
    for tid, obs in tracks.items():
        idx = sorted(obs)
        first, last = idx[0], idx[-1]
        m = {i: member(obs[i].bbox, workplace, ay, of) for i in idx} if workplace else {i: 0.5 for i in idx}
        geo[tid] = {
            "idx": idx, "first": first, "last": last, "m": m,
            "exit_end": workplace is not None and (m[last] < 0.3 or _near_exit(obs[last].bbox, door, border, ay, of)),
            "enter_start": workplace is not None and (m[first] < 0.3 or _near_exit(obs[first].bbox, door, border, ay, of)),
        }
    exit_times = sorted(float(tl.times[g["last"]]) for g in geo.values() if g["exit_end"])
    # проход 2: статус трека на каждом сэмпле
    for tid, obs in tracks.items():
        g = geo[tid]
        idx, first, last, m = g["idx"], g["first"], g["last"], g["m"]
        o0 = obs[first]
        frac_client = float(np.mean([member(obs[i].bbox, client, ay, of) >= 0.5 for i in idx])) if client else 0.0
        track_info[tid] = {"p_cashier": o0.p_cashier, "p_collector": o0.p_collector, "explicit": o0.explicit_role, "frac_client": frac_client}
        cashierish = o0.p_cashier >= 0.4
        t_first = float(tl.times[first])
        for i in range(n):
            if i in obs:
                mi = m[i]
                if mi >= 0.5:
                    ws_occ[i] = max(ws_occ[i], mi)
                if cashierish:
                    cashier_vis[i] = max(cashier_vis[i], o0.p_cashier)
                    cash_in[i] = max(cash_in[i], o0.p_cashier * mi)
                    cash_conf[i] = max(cash_conf[i], 1.0 if (o0.explicit_role or o0.p_cashier >= 0.6) else 0.4)
                    if mi >= 0.5:
                        cashier_gone[i] = 0.0
                continue
            ti = float(tl.times[i])
            if i < first:
                # трек «возник» посреди рабочего места: человек мог быть там и раньше (пропуски детектора).
                # Неизвестно в окне backfill_s; раньше — «нет», только если до этого кто-то подтверждённо ушёл.
                gone = g["enter_start"] or (t_first - ti > backfill_s and any(e <= ti for e in exit_times))
            elif i > last:
                gone = g["exit_end"]
            else:  # разрыв внутри трека
                prev = max(j for j in idx if j < i)
                nxt = min(j for j in idx if j > i)
                if m[prev] >= 0.5 and m[nxt] >= 0.5:
                    ws_occ[i] = max(ws_occ[i], 0.8)
                    if cashierish:
                        cash_in[i] = max(cash_in[i], o0.p_cashier * 0.8)
                        cash_conf[i] = max(cash_conf[i], 0.6)
                        cashier_gone[i] = 0.0
                    continue
                gone = m[prev] < 0.3 and m[nxt] < 0.3
            if not gone:
                ws_unknown[i] = 1.0
                if cashierish:
                    cashier_gone[i] = 0.0
                    cashier_absent_known[i] = 0.0

    obs_unknown = np.zeros(n)
    out: dict[str, Sig] = {}
    npers = np.zeros(n)
    outsider_p = np.zeros(n)
    outsider_c = np.zeros(n)
    coll_p = np.zeros(n)
    coll_c = np.zeros(n)
    for i in range(n):
        ps = tl.persons[i]
        npers[i] = float(len(ps))
        best_out, best_out_c, unk = 0.0, 0.0, 0.0
        staff_only = True
        for o in ps:
            mi = member(o.bbox, workplace, ay, of) if workplace else 0.0
            info = track_info.get(int(o.track_id)) if o.track_id is not None else None
            # доказательства «не сотрудник»: явная роль visitor или история в зоне клиентов
            if o.explicit_role:
                p_vis = max(0.0, 1.0 - o.p_cashier - o.p_collector)
                c_vis = 1.0
            elif info is not None and info["frac_client"] >= float(pc.get("client_history_frac", 0.2)):
                p_vis, c_vis = 0.75 * (1.0 - o.p_collector), 0.8
            else:
                p_vis, c_vis = 0.15, 0.3
            if mi >= 0.5:
                if not (o.explicit_role and p_vis < 0.3) and o.p_cashier < 0.7:
                    staff_only = False
                if c_vis < 0.5:
                    unk = max(unk, mi)
                if pc.get("outsider_excludes_collectors", True):
                    p_vis *= 1.0 - o.p_collector
                if mi * p_vis > best_out:
                    best_out, best_out_c = mi * p_vis, c_vis
            if o.p_collector > coll_p[i]:
                coll_p[i] = o.p_collector
                coll_c[i] = 1.0 if o.explicit_role else 0.6
        obs_unknown[i] = unk
        outsider_p[i] = best_out
        if best_out >= 0.5:
            outsider_c[i] = best_out_c * q[i]
        elif staff_only:
            outsider_c[i] = q[i] * (1.0 if workplace else 0.0)   # в зоне только сотрудники/никого — известно «нет»
        else:
            outsider_c[i] = 0.3 * q[i]                            # есть человек неизвестной роли — неизвестно
        if coll_p[i] < 0.5:
            all_known = all(o.explicit_role or o.p_cashier >= 0.7 for o in ps)
            coll_c[i] = q[i] * (0.8 if all_known else 0.3)
        else:
            coll_c[i] *= q[i]

    if not has_tracks:
        # только детекции: присутствие видно, но уход подтвердить нельзя
        for i in range(n):
            for o in tl.persons[i]:
                mi = member(o.bbox, workplace, ay, of) if workplace else 0.0
                if mi >= 0.5:
                    ws_occ[i] = max(ws_occ[i], mi)
                    cash_in[i] = max(cash_in[i], o.p_cashier * mi)
                    cash_conf[i] = max(cash_conf[i], 0.5)
                    cashier_gone[i] = 0.0
                cashier_vis[i] = max(cashier_vis[i], o.p_cashier)
        cashier_absent_known[:] = 0.0
        ws_unknown[ws_occ < 0.5] = 1.0

    # рабочее место пусто: известно, только если никто не «потерян» в зоне, детектор работал и кадр годный
    empty_p = np.where(ws_occ >= 0.5, 0.0, 1.0)
    empty_c = np.where(ws_occ >= 0.5, 1.0, np.where(ws_unknown > 0, 0.0, q))
    if not workplace:
        empty_c[:] = 0.0
    ciw_p = np.where(cash_in >= 0.5, cash_in, np.where(cashier_gone > 0, 0.0, cash_in))
    ciw_c = np.where(cash_in >= 0.5, cash_conf, np.where((cashier_gone > 0) & (ws_unknown == 0), q, 0.0))
    if not workplace:
        ciw_c = np.minimum(ciw_c, 0.3)
    cp_p = cashier_vis
    cp_c = np.where(cashier_vis >= 0.5, 1.0, np.where(cashier_absent_known > 0, q, 0.0)) if has_tracks else np.where(cashier_vis >= 0.5, 0.6, 0.0)

    out["cashier_present"] = Sig(cp_p, cp_c, ["track" if has_tracks else "det"])
    out["cashier_in_workplace"] = Sig(ciw_p, ciw_c, ["track" if has_tracks else "det"])
    out["outsider_in_cashier_area"] = Sig(outsider_p, outsider_c, ["track"])
    out["collector_present"] = Sig(coll_p, coll_c, ["track"])
    out["n_persons"] = Sig(npers, q.copy(), ["track" if has_tracks else "det"])
    extra = {
        "workplace_empty_p": empty_p,
        "workplace_empty_c": empty_c,
        "unknown_person_in_cashier_area": obs_unknown,
        "workplace_lost": ws_unknown,
        # для предикатов: кассир наблюдается на рабочем месте / подтверждённо отсутствует (уход виден)
        "cashier_in_obs": (cash_in >= 0.5).astype(float),
        "cashier_gone_known": ((cashier_gone > 0) & (ws_unknown == 0) & (cash_in < 0.5)).astype(float) if has_tracks and workplace else np.zeros(n),
    }
    _ = dt
    return out, extra
