"""Один главный номер для итераций: аппроксимация 100-балльной шкалы жюри.

Критерии ТЗ: Детекция 25 · Таймкод+кадр 15 · Тип 15 · Нормы 15 · Устойчивость 10 · Продукт/UI 10 ·
Архитектура/локальность 10. Формулы каждого компонента (доля 0..1 × вес) — в README и в ``explain``.

CLI::

    python -m eval.score_like_jury --metrics runs/eval/metrics.json [--robustness robustness.json] [--jury-sim jury_sim.json]
    python -m eval.score_like_jury --labels data/labels.csv --preds runs/eval/preds
"""
from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path
from typing import Any, Optional

from .common import load_config, read_json, write_json

_TYPE_RE = re.compile(r"^V(0[1-9]|[1-9]\d)$")


def _step(dt_abs: Optional[float], steps: list[list[float]]) -> float:
    if dt_abs is None:
        return 0.0
    for thr, credit in sorted(steps, key=lambda x: x[0]):
        if dt_abs <= float(thr) + 1e-9:
            return float(credit)
    return 0.0


def _avg(xs: list[float]) -> Optional[float]:
    return statistics.fmean(xs) if xs else None


def _norm_score(n: dict) -> float:
    has = 1.0 if n.get("n_rules", 0) > 0 else 0.0
    src = n.get("in_catalog") if n.get("in_catalog") is not None else n.get("url_ok")
    return 0.4 * has + 0.3 * (1.0 if src else 0.0) + 0.3 * float(n.get("verified_share") or 0.0)


def component_detection(summary: dict, cfg: dict) -> tuple[Optional[float], str]:
    v = summary["video"]
    j = cfg.get("jury") or {}
    cfn, cfp = float(j.get("fn_cost", 2.0)), float(j.get("fp_cost", 1.0))
    P, N = v["n_pos"], v["n_neg"]
    if not P + N:
        return None, "нет видео"
    failed_neg = sum(1 for _ in [])  # failed на негативе — тоже ошибка (жюри не получит ответ)
    fp_eff = v["fp"] + len([x for x in v.get("failed_ids", []) if x not in set(v.get("fn_ids", []))])
    cost = cfn * v["fn"] + cfp * fp_eff + failed_neg
    val = max(0.0, 1.0 - cost / (cfn * P + cfp * N))
    return val, f"1 − ({cfn:g}·FN + {cfp:g}·(FP+failed⁻)) / ({cfn:g}·P + {cfp:g}·N): FN={v['fn']} FP={v['fp']} " \
                f"failed⁻={fp_eff - v['fp']} P={P} N={N}"


def component_timecode(per_video: list[dict], cfg: dict) -> tuple[Optional[float], str]:
    j = cfg.get("jury") or {}
    tp_part = float(j.get("time_part", 0.667))
    steps = j.get("time_steps") or [[1.0, 1.0], [2.0, 0.75], [3.0, 0.5], [5.0, 0.25]]
    vals = []
    for e in per_video:
        if e["gt_label"] != "violation" or not e["gt_times"]:
            continue
        if e["status"] != "done" or e["verdict"] != "violation":
            vals.append(0.0)
            continue
        vals.append(tp_part * _step(e.get("dt_abs"), steps) + (1 - tp_part) * (1.0 if e.get("evidence_ok") else 0.0))
    return _avg(vals), f"среднее по {len(vals)} позитивам с GT-таймкодом: {tp_part:g}·ступень(|Δt|) + " \
                       f"{1 - tp_part:g}·[кадр в ±tol]; FN → 0"


def component_type(summary: dict, per_video: list[dict], cfg: dict, catalog_types: Optional[set] = None
                   ) -> tuple[Optional[float], str, str]:
    ty = summary.get("types") or {}
    if ty.get("available"):
        return ty.get("accuracy"), f"точность типа по {ty.get('n_events')} GT-событиям (пропуск → 0)", "measured"
    credit99 = float((cfg.get("jury") or {}).get("v99_type_credit", 0.5))
    vals = []
    for e in per_video:
        if e["gt_label"] != "violation":
            continue
        if e["status"] != "done" or e["verdict"] != "violation":
            vals.append(0.0)
            continue
        t = (e.get("pred_type") or "").upper()
        if t == "V99":
            vals.append(credit99)
        elif _TYPE_RE.match(t) and (catalog_types is None or t in catalog_types):
            vals.append(1.0)
        else:
            vals.append(0.0)
    return _avg(vals), f"PROXY (нет labels_types): валидный type_id из каталога = 1, V99 = {credit99:g}, " \
                       f"нет/неизвестный = 0; среднее по {len(vals)} позитивам", "proxy"


def component_norms(per_video: list[dict]) -> tuple[Optional[float], str]:
    vals = []
    for e in per_video:
        if e["gt_label"] != "violation":
            continue
        if e["status"] != "done" or e["verdict"] != "violation" or not e.get("norms"):
            vals.append(0.0)
            continue
        vals.append(statistics.fmean(_norm_score(n) for n in e["norms"]))
    return _avg(vals), "среднее по позитивам: 0.4·[≥1 rule] + 0.3·[все rules из каталога (или adilet URL)] + " \
                       "0.3·доля verified; FN → 0"


def component_robustness(rob: Optional[dict]) -> tuple[Optional[float], str]:
    if not rob:
        return None, "нет robustness.json"
    s = rob.get("summary") or rob
    flip = s.get("flip_rate")
    stab = s.get("timecode_stable_rate")
    if flip is None:
        return None, "в robustness.json нет flip_rate"
    val = 0.7 * (1 - float(flip)) + 0.3 * (float(stab) if stab is not None else 1 - float(flip))
    return val, f"0.7·(1 − доля смены вердикта={flip}) + 0.3·доля стабильного таймкода={stab}"


def component_product(summary: dict) -> tuple[Optional[float], str]:
    sch, ev, lng = summary["schema"], summary["evidence"], summary["language"]
    parts = []
    if sch.get("valid_rate") is not None:
        parts.append((0.4, sch["valid_rate"]))
    if ev.get("valid_jpeg_rate") is not None:
        parts.append((0.2, ev["valid_jpeg_rate"]))
    if ev.get("frames_per_violation_ok") is not None:
        parts.append((0.2, ev["frames_per_violation_ok"]))
    if lng.get("n"):
        parts.append((0.2, float(lng["hedged_rate"] or 0) * (1 - float(lng["forbidden_rate"] or 0))))
    if not parts:
        return None, "нет ответов"
    w = sum(p[0] for p in parts)
    return sum(a * b for a, b in parts) / w, "PROXY: 0.4·схема ТЗ + 0.2·валидный JPEG + 0.2·1–3 кадра + " \
                                             "0.2·hedged·(1−forbidden) (UI жюри оценит само)"


def component_architecture(summary: dict, jury_sim: Optional[dict]) -> tuple[Optional[float], str]:
    lat, v = summary["latency"], summary["video"]
    parts = []
    if lat.get("within_sla_rate") is not None:
        parts.append((0.5, lat["within_sla_rate"]))
    if v["n"]:
        parts.append((0.3, 1 - v["failed"] / v["n"]))
    if jury_sim:
        js = jury_sim.get("summary") or jury_sim
        if js.get("pass_rate") is not None:
            parts.append((0.2, float(js["pass_rate"])))
    if not parts:
        return None, "нет данных"
    w = sum(p[0] for p in parts)
    return sum(a * b for a, b in parts) / w, "PROXY: 0.5·доля в SLA + 0.3·(1 − доля failed) + 0.2·jury_sim pass_rate"


def score(metrics: dict, robustness: Optional[dict] = None, jury_sim: Optional[dict] = None,
          cfg: Optional[dict] = None, catalog_types: Optional[set] = None) -> dict:
    cfg = cfg or load_config()
    j = cfg.get("jury") or {}
    w = j.get("weights") or {}
    defaults = j.get("defaults") or {}
    summary, per_video = metrics["summary"], metrics["per_video"]
    comps: dict[str, dict[str, Any]] = {}

    def put(name: str, val: Optional[float], note: str, source: str = "measured") -> None:
        if val is None and defaults.get(name) is not None:
            val, source, note = float(defaults[name]), "default", f"значение по умолчанию из config ({note})"
        elif val is None:
            source = "n/a"
        weight = float(w.get(name, 0))
        comps[name] = {"weight": weight, "value": None if val is None else round(float(val), 4),
                       "points": None if val is None else round(weight * float(val), 2), "source": source,
                       "note": note}

    put("detection", *component_detection(summary, cfg))
    put("timecode", *component_timecode(per_video, cfg))
    tv, tn, ts = component_type(summary, per_video, cfg, catalog_types)
    put("type", tv, tn, ts)
    put("norms", *component_norms(per_video))
    put("robustness", *component_robustness(robustness))
    pv, pn = component_product(summary)
    put("product", pv, pn, "proxy")
    av, an = component_architecture(summary, jury_sim)
    put("architecture", av, an, "proxy")
    avail = [c for c in comps.values() if c["points"] is not None]
    raw = sum(c["points"] for c in avail)
    wmax = sum(c["weight"] for c in avail)
    return {"jury_score": round(100.0 * raw / wmax, 2) if wmax else 0.0,
            "points": round(raw, 2), "max_points_available": wmax,
            "missing": [k for k, c in comps.items() if c["points"] is None],
            "components": comps}


def format_score(res: dict) -> str:
    lines = [f"JURY_SCORE = {res['jury_score']:.1f} / 100   ({res['points']:.1f} из {res['max_points_available']:g} "
             f"доступных баллов" + (f"; нет данных: {', '.join(res['missing'])}" if res["missing"] else "") + ")"]
    for k, c in res["components"].items():
        pts = "  n/a" if c["points"] is None else f"{c['points']:5.1f}"
        lines.append(f"  {k:<13} {pts} / {c['weight']:>4g}  [{c['source']}]  {c['note']}")
    return "\n".join(lines)


def _catalog_types(cfg: dict) -> Optional[set]:
    from .metrics import load_catalog

    cat = load_catalog(cfg)
    return set((cat or {}).get("types") or {}) or None


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="100-балльная аппроксимация оценки жюри")
    ap.add_argument("--metrics", default=None, help="metrics.json от eval.metrics / eval.run")
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--preds", default=None)
    ap.add_argument("--types", default="data/labels_types.csv")
    ap.add_argument("--robustness", default=None)
    ap.add_argument("--jury-sim", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.metrics:
        metrics = read_json(args.metrics)
    elif args.preds:
        from .metrics import evaluate_dirs

        metrics = evaluate_dirs(args.labels, args.preds, args.types, cfg)
    else:
        ap.error("нужен --metrics или --preds")
    rob = read_json(args.robustness) if args.robustness else None
    js = read_json(args.jury_sim) if args.jury_sim else None
    res = score(metrics, rob, js, cfg, _catalog_types(cfg))
    print(format_score(res))
    if args.out:
        write_json(args.out, res)
    elif args.metrics:
        write_json(Path(args.metrics).with_name("jury_score.json"), res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
