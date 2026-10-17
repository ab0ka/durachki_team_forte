"""Быстрый цикл тюнинга правил по кэшу состояний (без CV) + сравнение с разметкой.

    python -m ml.rules.replay --cache cache/ --labels data/labels.csv
    python -m ml.rules.replay --cache cache/ --labels data/labels.csv --assume-verified \
        --set rules.V01_outsider_in_cashier_area.min_duration_s=3 --sweep global.ffill_max_s=0.5,1,2
    python -m ml.rules.replay --rules-table

Вердикт по видео = есть ли событие ``candidate_violation`` (``--count-review`` — диагностика:
считать и ``review_observation``). Таймкод — ``t_peak`` ближайшего к разметке события.
``--assume-verified`` — ТОЛЬКО для калибровки детекции на train/calibration: считает нормы всех
типов проверенными (в проде статус норм берётся из norms/catalog.yaml A13).
Пороги, найденные здесь, применять только если они подобраны на train/calibration, а не на тесте.
"""
from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from .cache_io import load_cache_dir, load_labels
from .catalog import FALLBACK_CATALOG, resolve_catalog
from .config import apply_overrides, parse_rules_config, read_raw_config
from .engine import CANDIDATE, INSUFFICIENT, REVIEW, assessment_of, evaluate

MAX_SWEEP = 500


def _load_catalog(path: Optional[str], assume_verified: bool) -> tuple[dict, str]:
    if path:
        import yaml

        with open(path, "r", encoding="utf-8") as fh:
            cat = yaml.safe_load(fh) or {}
        src = path
    else:
        cat, src = resolve_catalog(None)
    cat = copy.deepcopy(dict(cat))
    if assume_verified:
        types = dict(cat.get("types") or {})
        for tid in set(types) | set(FALLBACK_CATALOG["types"]):
            entry = dict(types.get(tid) or FALLBACK_CATALOG["types"].get(tid, {"name": tid}))
            entry["rules"] = [{"document": "CALIBRATION-ONLY", "clause": None, "url": "calibration://assume-verified",
                               "quote": "", "verified": True}]
            types[tid] = entry
        cat["types"] = types
        src += " + assume-verified (калибровка)"
    return cat, src


def _fmt(v: Any, nd: int = 1) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "да" if v else "нет"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _table(rows: list[list[Any]], header: list[str]) -> str:
    cells = [[_fmt(c) for c in r] for r in rows]
    widths = [max(len(h), *(len(r[i]) for r in cells)) if cells else len(h) for i, h in enumerate(header)]
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths))
    out = [line, "  ".join("-" * w for w in widths)]
    out += ["  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in cells]
    return "\n".join(out)


def run_replay(cache: dict[str, dict[str, Any]], labels: dict[str, list[dict[str, Any]]], cfg: Any, catalog: dict,
               tol: float = 2.0, min_conf: float = 0.0, count_review: bool = False) -> dict[str, Any]:
    videos = []
    rule_fires: dict[str, dict[str, int]] = {}
    for key in sorted(set(cache) | set(labels)):
        item = cache.get(key)
        gt_list = labels.get(key) or [{}]
        gt = gt_list[0]
        row: dict[str, Any] = {"id": key, "name": (item or gt).get("name", key), "gt": gt.get("violation"),
                               "gt_t": gt.get("t"), "gt_all_t": [g.get("t") for g in gt_list if g.get("t") is not None],
                               "cached": item is not None}
        if item is None:
            row["note"] = "нет кэша"
            videos.append(row)
            continue
        rep = evaluate(item["states"], item.get("tracks") or [], item.get("zones") or [], catalog, cfg)
        evs = [e for e in rep.events if e.confidence >= min_conf]
        cands = [e for e in evs if assessment_of(e) == CANDIDATE]
        reviews = [e for e in evs if assessment_of(e) == REVIEW]
        insuff = [e for e in evs if assessment_of(e) == INSUFFICIENT]
        pool = cands + (reviews if count_review else [])
        row.update(pred=bool(pool), n_cand=len(cands), n_review=len(reviews), n_insuff=len(insuff),
                   warnings=rep.warnings[:5],
                   events=[{"type_id": e.type_id, "t_start": e.t_start, "t_end": e.t_end, "t_peak": e.t_peak,
                            "confidence": e.confidence, "assessment": assessment_of(e),
                            "rule_ids": e.extra.get("rule_ids", [])} for e in evs])
        best = None
        if pool:
            if row["gt_t"] is not None:
                best = min(pool, key=lambda e: (abs(e.t_peak - row["gt_t"]), -e.confidence))
            else:
                best = max(pool, key=lambda e: e.confidence)
            row.update(type_id=best.type_id, t_peak=best.t_peak, t_start=best.t_start, conf=best.confidence,
                       rule=",".join(best.extra.get("rule_ids", [])))
            if row["gt_t"] is not None:
                row["dt"] = round(best.t_peak - row["gt_t"], 2)
                row["dt_onset"] = round(best.t_start - row["gt_t"], 2)
                row["hit"] = abs(row["dt"]) <= tol
        bucket = "violation" if row["gt"] else ("clean" if row["gt"] is False else "unlabeled")
        for e in evs:
            for rid in e.extra.get("rule_ids", []):
                d = rule_fires.setdefault(rid, {"violation": 0, "clean": 0, "unlabeled": 0})
                d[bucket] += 1
        videos.append(row)
    labeled = [v for v in videos if v.get("cached") and v.get("gt") is not None]
    tp = sum(1 for v in labeled if v["gt"] and v["pred"])
    fn = sum(1 for v in labeled if v["gt"] and not v["pred"])
    fp = sum(1 for v in labeled if not v["gt"] and v["pred"])
    tn = sum(1 for v in labeled if not v["gt"] and not v["pred"])
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if prec is not None and rec is not None else None)
    tp_rows = [v for v in labeled if v["gt"] and v["pred"] and v.get("dt") is not None]
    abs_dt = [abs(v["dt"]) for v in tp_rows]
    summary = {
        "n_videos": len(videos), "n_cached": sum(1 for v in videos if v.get("cached")), "n_labeled": len(labeled),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec, "f1": f1,
        "accuracy": (tp + tn) / len(labeled) if labeled else None,
        "time_hit_rate": {str(t): (sum(1 for d in abs_dt if d <= t) / len(abs_dt) if abs_dt else None)
                          for t in sorted({1.0, 2.0, 3.0, 5.0, tol})},
        "mean_abs_dt": statistics.mean(abs_dt) if abs_dt else None,
        "median_abs_dt": statistics.median(abs_dt) if abs_dt else None,
        "median_gt_minus_t_peak": statistics.median([-v["dt"] for v in tp_rows]) if tp_rows else None,
        "rule_fires": rule_fires,
        "count_review": count_review,
    }
    return {"videos": videos, "summary": summary}


def _print_report(res: dict[str, Any], tol: float, show_events: bool, catalog_src: str, cfg_src: str) -> None:
    rows = []
    for v in res["videos"]:
        rows.append([v["name"], v.get("gt"), v.get("gt_t"), v.get("pred"), v.get("n_cand"), v.get("n_review"),
                     v.get("n_insuff"), v.get("type_id"), v.get("t_peak"), v.get("dt"), v.get("hit"),
                     v.get("rule") or v.get("note")])
    print(_table(rows, ["видео", "GT", "GT_t", "pred", "cand", "review", "insuff", "тип", "t_peak", "Δt", f"≤{tol:g}с",
                        "правило/прим."]))
    if show_events:
        for v in res["videos"]:
            for e in v.get("events", []):
                print(f"  {v['name']}: {e['type_id']} {e['assessment']} [{e['t_start']:.1f}–{e['t_end']:.1f}] "
                      f"t_peak={e['t_peak']:.1f} conf={e['confidence']:.2f} {','.join(e['rule_ids'])}")
    s = res["summary"]
    print()
    print(f"rules: {cfg_src} | catalog: {catalog_src} | review считается нарушением: {_fmt(s['count_review'])}")
    print(f"видео: {s['n_videos']} (кэш: {s['n_cached']}, размечено: {s['n_labeled']})")
    print(f"TP={s['tp']} FP={s['fp']} FN={s['fn']} TN={s['tn']}  precision={_fmt(s['precision'], 3)} "
          f"recall={_fmt(s['recall'], 3)} F1={_fmt(s['f1'], 3)} accuracy={_fmt(s['accuracy'], 3)}")
    hits = ", ".join(f"≤{k}с: {_fmt(v, 2)}" for k, v in s["time_hit_rate"].items())
    print(f"таймкод (TP): {hits}; mean|Δt|={_fmt(s['mean_abs_dt'], 2)} median|Δt|={_fmt(s['median_abs_dt'], 2)}")
    if s["median_gt_minus_t_peak"] is not None:
        print(f"медиана (GT − t_peak) = {s['median_gt_minus_t_peak']:+.2f} с — справочно; сдвиг применять только "
              f"если подобран на train/calibration")
    if s["rule_fires"]:
        print(_table([[rid, d["violation"], d["clean"], d["unlabeled"]] for rid, d in sorted(s["rule_fires"].items())],
                     ["правило", "на нарушениях", "на чистых", "без разметки"]))


def _sweep(cache: dict, labels: dict, raw: dict, base_sets: list[str], sweeps: list[str], catalog: dict,
           tol: float, min_conf: float, count_review: bool, src: str) -> list[dict[str, Any]]:
    axes = []
    for sw in sweeps:
        key, _, vals = sw.partition("=")
        axes.append([(key.strip(), v.strip()) for v in vals.split(",") if v.strip()])
    combos = list(itertools.product(*axes))[:MAX_SWEEP]
    results = []
    for combo in combos:
        sets = base_sets + [f"{k}={v}" for k, v in combo]
        cfg = parse_rules_config(apply_overrides(raw, sets), src)
        s = run_replay(cache, labels, cfg, catalog, tol, min_conf, count_review)["summary"]
        results.append({"params": dict(combo), **{k: s[k] for k in ("tp", "fp", "fn", "tn", "f1", "recall",
                                                                    "precision", "median_abs_dt")},
                        "hit": s["time_hit_rate"].get(str(tol))})
    results.sort(key=lambda r: (-(r["f1"] or 0), r["fn"], -(r["hit"] or 0)))
    return results


def rules_table_markdown(cfg: Any) -> str:
    lines = ["| Правило | Тип | Вид | Включено | По бездействию | Причина |", "|---|---|---|---|---|---|"]
    for r in cfg.rules_table():
        lines.append(f"| `{r['id']}` | {r['type_id']} | {r['kind']} | {'да' if r['enabled'] else 'нет'} | "
                     f"{'да' if r['inaction'] else '—'} | {r['reason']} |")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ml.rules.replay", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", help="папка кэша состояний (states/tracks/zones JSON)")
    ap.add_argument("--labels", help="CSV разметки: файл, нарушение/без нарушения, секунда")
    ap.add_argument("--rules", default=None, help="путь к rules.yaml (по умолчанию ml/rules/rules.yaml)")
    ap.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE",
                    help="переопределение параметра, напр. rules.V01_outsider_in_cashier_area.on=0.6")
    ap.add_argument("--sweep", action="append", default=[], metavar="KEY=v1,v2",
                    help="перебор значений (декартово произведение, ≤500 комбинаций)")
    ap.add_argument("--catalog", default=None, help="norms/catalog.yaml (иначе norms.load_catalog() или fallback)")
    ap.add_argument("--assume-verified", action="store_true",
                    help="КАЛИБРОВКА: считать нормы всех типов проверенными")
    ap.add_argument("--count-review", action="store_true", help="диагностика: review_observation = нарушение")
    ap.add_argument("--tol", type=float, default=2.0, help="допуск таймкода, с")
    ap.add_argument("--min-conf", type=float, default=0.0)
    ap.add_argument("--events", action="store_true", help="печатать все события")
    ap.add_argument("--json", dest="json_out", help="сохранить результаты в JSON")
    ap.add_argument("--csv", dest="csv_out", help="сохранить таблицу по видео в CSV")
    ap.add_argument("--rules-table", action="store_true", help="напечатать таблицу правил (markdown) и выйти")
    args = ap.parse_args(argv)

    raw, src = read_raw_config(args.rules)
    cfg = parse_rules_config(apply_overrides(raw, args.sets), src)
    if args.rules_table:
        print(rules_table_markdown(cfg))
        return 0
    if not args.cache:
        ap.error("нужен --cache (или --rules-table)")
    cache = load_cache_dir(args.cache)
    labels = load_labels(args.labels) if args.labels else {}
    if not cache:
        print(f"[replay] в {args.cache} не найдено кэшей состояний", file=sys.stderr)
        return 2
    catalog, cat_src = _load_catalog(args.catalog, args.assume_verified)
    if args.sweep:
        results = _sweep(cache, labels, raw, args.sets, args.sweep, catalog, args.tol, args.min_conf,
                         args.count_review, src)
        print(_table([[json.dumps(r["params"], ensure_ascii=False), r["tp"], r["fp"], r["fn"], r["tn"],
                       r["f1"], r["hit"], r["median_abs_dt"]] for r in results],
                     ["параметры", "TP", "FP", "FN", "TN", "F1", f"hit≤{args.tol:g}", "med|Δt|"]))
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        return 0
    res = run_replay(cache, labels, cfg, catalog, args.tol, args.min_conf, args.count_review)
    _print_report(res, args.tol, args.events, cat_src, src)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    if args.csv_out:
        keys = ["name", "gt", "gt_t", "pred", "n_cand", "n_review", "n_insuff", "type_id", "t_peak", "t_start", "dt",
                "dt_onset", "hit", "conf", "rule"]
        with open(args.csv_out, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(keys)
            for v in res["videos"]:
                w.writerow([v.get(k) for k in keys])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
