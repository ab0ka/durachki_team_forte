"""Метрики «как у жюри» по предсказаниям (preds/*.json) и разметке (data/labels.csv).

* по видео: confusion, precision / recall / F1, FP / FN / failed;
* локализация: |Δt| ближайшего нарушения до GT, hit@0.5/1/2/3 с, знак сдвига (рано/поздно);
* кадр-доказательство в ±evidence_tol_s от GT (и валидный JPEG);
* типы (если есть data/labels_types.csv): точность, confusion; иначе — распределение;
* нормы: у каждого нарушения ≥1 rule, доля verified, URL adilet, наличие в norms/catalog.yaml;
* схема ТЗ; язык описаний (без домыслов); латентность; согласованность md5-дубликатов.

CLI::

    python -m eval.metrics --labels data/labels.csv --preds runs/eval/preds [--types data/labels_types.csv]
"""
from __future__ import annotations

import argparse
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from .common import load_config, normalize_video_id, read_json, video_sort_key, write_json
from .labels import LABEL_VIOLATION, GroundTruth, load_ground_truth
from .schema import check_api_response, decode_jpeg_b64

CATEGORIES = ("TP", "TN", "FP", "FN", "early", "late", "wrong_type", "failed")


# ----------------------------------------------------------------------------- загрузка
def unwrap_prediction(d: Any) -> tuple[dict, dict]:
    """preds/*.json бывают «голым» ответом API или обёрткой {"response": ..., "latency_s": ...}."""
    if isinstance(d, dict) and isinstance(d.get("response"), dict):
        info = {k: v for k, v in d.items() if k != "response"}
        return d["response"], info
    return (d if isinstance(d, dict) else {}), {}


def load_predictions(preds_dir: str | Path) -> tuple[dict[str, dict], dict[str, dict]]:
    """→ ({video_id: api_dict}, {video_id: runinfo}). Файлы, начинающиеся с «_», — служебные."""
    preds: dict[str, dict] = {}
    info: dict[str, dict] = {}
    pdir = Path(preds_dir)
    runinfo_path = pdir / "_runinfo.json"
    if runinfo_path.exists():
        for k, v in (read_json(runinfo_path) or {}).items():
            info[normalize_video_id(k) or k] = v
    for p in sorted(pdir.glob("*.json")):
        if p.name.startswith("_"):
            continue
        try:
            api, extra = unwrap_prediction(read_json(p))
        except Exception as e:  # noqa: BLE001
            api, extra = {}, {"error": f"битый JSON: {e}"}
        vid = normalize_video_id(extra.get("video_id") or p.stem) or p.stem
        preds[vid] = api
        if extra:
            info.setdefault(vid, {}).update(extra)
    return preds, info


def load_catalog(cfg: Optional[dict] = None) -> Optional[dict]:
    """norms.load_catalog() (A13) → иначе YAML по cfg.norms.catalog_path → иначе None."""
    try:
        from norms import load_catalog as _lc  # type: ignore

        cat = _lc()
        if isinstance(cat, dict) and cat:
            return cat
    except Exception:  # noqa: BLE001
        pass
    from .common import REPO_ROOT

    path = Path(((cfg or {}).get("norms") or {}).get("catalog_path") or "norms/catalog.yaml")
    if not path.is_absolute():
        path = REPO_ROOT / path
    if path.exists():
        try:
            import yaml

            return yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None
    return None


# ----------------------------------------------------------------------------- per-video
@dataclass
class VideoEval:
    video_id: str
    gt_label: str
    gt_times: list[float]
    gt_types: list[tuple[float, str]]
    members: list[str]
    conflict: bool
    pred_id: Optional[str] = None          # чьё предсказание взяли (член md5-группы)
    status: str = "missing"                # done | failed | processing | missing
    verdict: Optional[str] = None
    n_violations: int = 0
    category: str = "failed"
    pred_t: Optional[float] = None         # таймкод ближайшего к GT нарушения (или топового)
    pred_type: Optional[str] = None
    pred_conf: Optional[float] = None
    matched_gt_t: Optional[float] = None
    dt_signed: Optional[float] = None      # pred - gt
    dt_abs: Optional[float] = None
    hits: dict[str, bool] = field(default_factory=dict)
    evidence_ok: Optional[bool] = None
    evidence_dt: Optional[float] = None
    type_results: list[dict] = field(default_factory=list)
    norms: list[dict] = field(default_factory=list)
    schema_errors: list[str] = field(default_factory=list)
    schema_warnings: list[str] = field(default_factory=list)
    frames_total: int = 0
    frames_valid: int = 0
    language: list[dict] = field(default_factory=list)
    latency_s: Optional[float] = None
    duration: Optional[float] = None
    error: Optional[str] = None
    violations_brief: list[dict] = field(default_factory=list)

    @property
    def gt_positive(self) -> bool:
        return self.gt_label == LABEL_VIOLATION

    @property
    def pred_positive(self) -> bool:
        return self.status == "done" and self.verdict == "violation"


def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _latency(api: dict, info: dict) -> Optional[float]:
    for v in (info.get("latency_s"), info.get("elapsed_s")):
        if _num(v) is not None:
            return _num(v)
    meta = api.get("meta") or {}
    t = meta.get("timings") or {}
    for k in ("total_s", "total", "wall_s", "elapsed_s"):
        if isinstance(t, dict) and _num(t.get(k)) is not None:
            return _num(t.get(k))
    for k in ("elapsed_s", "latency_s", "processing_s"):
        if _num(meta.get(k)) is not None:
            return _num(meta.get(k))
    return None


def _duration(api: dict, info: dict) -> Optional[float]:
    meta = api.get("meta") or {}
    for v in (info.get("duration"), meta.get("duration"), meta.get("duration_s"), (meta.get("video") or {}).get("duration")
              if isinstance(meta.get("video"), dict) else None):
        if _num(v) is not None and _num(v) > 0:
            return _num(v)
    return None


def _rule_key(r: dict) -> tuple:
    return (str(r.get("url") or "").strip().rstrip("/"), str(r.get("clause") or "").strip())


def _catalog_rules(catalog: Optional[dict], type_id: str) -> Optional[set]:
    if not catalog:
        return None
    t = ((catalog.get("types") or {}).get(type_id) or {})
    return {_rule_key(r) for r in (t.get("rules") or []) if isinstance(r, dict)}


def evaluate_video(gt: GroundTruth, api: Optional[dict], info: dict, cfg: dict,
                   catalog: Optional[dict] = None, pred_id: Optional[str] = None) -> VideoEval:
    ve = VideoEval(video_id=gt.video_id, gt_label=gt.label, gt_times=list(gt.times), gt_types=list(gt.types),
                   members=list(gt.members or [gt.video_id]), conflict=gt.conflict, pred_id=pred_id)
    ve.latency_s = _latency(api or {}, info or {})
    ve.error = (info or {}).get("error")
    if not api:
        ve.status, ve.category = "missing", "failed"
        return ve
    ve.status = api.get("status") or "missing"
    ve.duration = _duration(api, info or {})
    rep = check_api_response(api, duration=ve.duration, max_frames=int(cfg.get("max_evidence_frames", 3)))
    ve.schema_errors, ve.schema_warnings = rep.errors, rep.warnings
    ve.frames_total, ve.frames_valid = rep.frames_total, rep.frames_valid_jpeg
    if ve.status != "done":
        ve.category = "failed"
        return ve
    ve.verdict = api.get("verdict")
    vios = [v for v in (api.get("violations") or []) if isinstance(v, dict)]
    ve.n_violations = len(vios)
    ve.violations_brief = [{"t": _num(v.get("timecode_sec")), "type_id": v.get("type_id") or "",
                            "type": v.get("type") or "", "conf": _num(v.get("confidence")),
                            "t_start": _num(v.get("t_start")), "t_end": _num(v.get("t_end")),
                            "n_frames": len(v.get("evidence_frames") or []), "n_rules": len(v.get("rules") or [])}
                           for v in vios]
    lang = cfg.get("language") or {}
    hedges = [h.lower() for h in lang.get("hedge_markers", [])]
    forbidden = [w.lower() for w in lang.get("forbidden", [])]
    allowed = tuple((cfg.get("norms") or {}).get("allowed_url_prefixes") or ())
    for v in vios:
        desc = str(v.get("description") or "").lower()
        ve.language.append({"hedged": any(h in desc for h in hedges),
                            "forbidden": [w for w in forbidden if w in desc]})
        rules = [r for r in (v.get("rules") or []) if isinstance(r, dict)]
        cat_rules = _catalog_rules(catalog, str(v.get("type_id") or ""))
        ve.norms.append({
            "type_id": v.get("type_id") or "",
            "n_rules": len(rules),
            "verified_share": (sum(bool(r.get("verified")) for r in rules) / len(rules)) if rules else 0.0,
            "url_ok": bool(rules) and all(str(r.get("url") or "").startswith(allowed) for r in rules) if allowed
            else bool(rules),
            "in_catalog": (bool(rules) and all(_rule_key(r) in cat_rules for r in rules)) if cat_rules is not None
            else None,
            "type_in_catalog": (str(v.get("type_id") or "") in (catalog.get("types") or {})) if catalog else None,
        })

    # --- вердикт
    pred_pos = ve.verdict == "violation"
    if not gt.positive:
        ve.category = "FP" if pred_pos else "TN"
    elif not pred_pos:
        ve.category = "FN"
    else:
        ve.category = "TP"

    # --- локализация: ближайшее (по времени) нарушение к любому GT-таймкоду
    timed = [(i, _num(v.get("timecode_sec"))) for i, v in enumerate(vios)]
    timed = [(i, t) for i, t in timed if t is not None]
    best: Optional[tuple[float, int, float]] = None
    if timed and gt.times:
        for i, t in timed:
            for g in gt.times:
                cand = (abs(t - g), i, g)
                if best is None or cand < best:
                    best = cand
    if best is not None:
        _, bi, g = best
        v = vios[bi]
        ve.pred_t, ve.matched_gt_t = _num(v.get("timecode_sec")), g
        ve.dt_signed = round(ve.pred_t - g, 3)
        ve.dt_abs = abs(ve.dt_signed)
    elif timed:  # GT без таймкода или GT-негатив: показываем самое уверенное нарушение
        bi = max(range(len(timed)), key=lambda k: _num(vios[timed[k][0]].get("confidence")) or 0.0)
        bi = timed[bi][0]
        v = vios[bi]
        ve.pred_t = _num(v.get("timecode_sec"))
    else:
        v = None
    if v is not None:
        ve.pred_type = v.get("type_id") or None
        ve.pred_conf = _num(v.get("confidence"))
    if gt.positive and pred_pos and gt.times and ve.dt_abs is not None:
        for k in cfg.get("hit_windows_s", [0.5, 1, 2, 3]):
            ve.hits[f"{float(k):g}"] = ve.dt_abs <= float(k) + 1e-9
        tol = float(cfg.get("evidence_tol_s", 1.0))
        frames = [f for f in (v.get("evidence_frames") or []) if isinstance(f, dict)]
        dts = [abs(_num(f.get("timestamp_sec")) - ve.matched_gt_t) for f in frames
               if _num(f.get("timestamp_sec")) is not None and decode_jpeg_b64(f.get("image_base64")) is not None]
        ve.evidence_dt = round(min(dts), 3) if dts else None
        ve.evidence_ok = bool(dts) and min(dts) <= tol + 1e-9
        el = float(cfg.get("early_late_tol_s", 2.0))
        if ve.dt_abs > el:
            ve.category = "early" if ve.dt_signed < 0 else "late"

    # --- типы
    if gt.types:
        win = float(cfg.get("type_match_window_s", 3.0))
        for gt_t, gt_type in gt.types:
            cands = [(abs(t - gt_t), i) for i, t in timed] if gt_t == gt_t else []
            cands = [c for c in cands if c[0] <= win]
            if not pred_pos or not cands:
                ve.type_results.append({"gt_t": gt_t, "gt_type": gt_type, "pred_type": None, "ok": False,
                                        "reason": "missed" if not pred_pos else "not_localized"})
                continue
            _, i = min(cands)
            pt = vios[i].get("type_id") or None
            ve.type_results.append({"gt_t": gt_t, "gt_type": gt_type, "pred_type": pt, "ok": pt == gt_type,
                                    "reason": "ok" if pt == gt_type else "wrong_type"})
        if ve.category == "TP" and any(r["reason"] == "wrong_type" for r in ve.type_results):
            ve.category = "wrong_type"
    return ve


# ----------------------------------------------------------------------------- агрегаты
def _rate(a: float, b: float) -> Optional[float]:
    return round(a / b, 4) if b else None


def _stats(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0, "mean": None, "median": None, "p90": None, "max": None}
    s = sorted(xs)
    p90 = s[min(len(s) - 1, int(math.ceil(0.9 * len(s))) - 1)]
    return {"n": len(s), "mean": round(statistics.fmean(s), 3), "median": round(statistics.median(s), 3),
            "p90": round(p90, 3), "max": round(s[-1], 3)}


def summarize(evals: list[VideoEval], cfg: dict, dup_check: Optional[list[dict]] = None) -> dict:
    pos = [e for e in evals if e.gt_positive]
    neg = [e for e in evals if not e.gt_positive]
    tp = [e for e in pos if e.pred_positive]
    fn = [e for e in pos if not e.pred_positive]
    fp = [e for e in neg if e.pred_positive]
    tn = [e for e in neg if not e.pred_positive]
    failed = [e for e in evals if e.status != "done"]
    prec = _rate(len(tp), len(tp) + len(fp))
    rec = _rate(len(tp), len(pos))
    f1 = round(2 * prec * rec / (prec + rec), 4) if prec and rec else (0.0 if (prec is not None and rec is not None) else None)
    confusion: dict[str, dict[str, int]] = {}
    for e in evals:
        pred = e.verdict if e.status == "done" else "failed"
        confusion.setdefault(e.gt_label, {}).setdefault(str(pred), 0)
        confusion[e.gt_label][str(pred)] += 1
    video = {
        "n": len(evals), "n_pos": len(pos), "n_neg": len(neg),
        "tp": len(tp), "tn": len(tn), "fp": len(fp), "fn": len(fn), "failed": len(failed),
        "precision": prec, "recall": rec, "f1": f1,
        "accuracy": _rate(len(tp) + len(tn) - sum(1 for e in tn if e.status != "done"), len(evals)),
        "specificity": _rate(len(tn), len(neg)),
        "fp_ids": [e.video_id for e in fp], "fn_ids": [e.video_id for e in fn],
        "failed_ids": [e.video_id for e in failed], "confusion": confusion,
    }
    # локализация
    timed_pos = [e for e in pos if e.gt_times]
    loc_tp = [e for e in timed_pos if e.dt_abs is not None]
    windows = [f"{float(k):g}" for k in cfg.get("hit_windows_s", [0.5, 1, 2, 3])]
    localization = {
        "n_pos_with_time": len(timed_pos), "n_tp_localized": len(loc_tp),
        "abs_dt": _stats([e.dt_abs for e in loc_tp]),
        "mean_signed_dt": round(statistics.fmean([e.dt_signed for e in loc_tp]), 3) if loc_tp else None,
        "hit_tp": {k: _rate(sum(e.hits.get(k, False) for e in loc_tp), len(loc_tp)) for k in windows},
        "hit_all": {k: _rate(sum(e.hits.get(k, False) for e in loc_tp), len(timed_pos)) for k in windows},
        "early_ids": [e.video_id for e in evals if e.category == "early"],
        "late_ids": [e.video_id for e in evals if e.category == "late"],
    }
    frames_total = sum(e.frames_total for e in evals)
    n_vios = sum(e.n_violations for e in evals)
    frames_per_vio_ok = sum(1 for e in evals for v in e.violations_brief
                            if 1 <= v["n_frames"] <= int(cfg.get("max_evidence_frames", 3)))
    evidence = {
        "ok_tp": _rate(sum(bool(e.evidence_ok) for e in loc_tp), len(loc_tp)),
        "ok_all": _rate(sum(bool(e.evidence_ok) for e in loc_tp), len(timed_pos)),
        "valid_jpeg_rate": _rate(sum(e.frames_valid for e in evals), frames_total),
        "frames_per_violation_ok": _rate(frames_per_vio_ok, n_vios),
        "evidence_dt": _stats([e.evidence_dt for e in loc_tp if e.evidence_dt is not None]),
    }
    # типы
    trs = [r for e in evals for r in e.type_results]
    tconf: dict[str, dict[str, int]] = {}
    per_type: dict[str, dict[str, int]] = {}
    for r in trs:
        tconf.setdefault(r["gt_type"], {}).setdefault(str(r["pred_type"] or "—"), 0)
        tconf[r["gt_type"]][str(r["pred_type"] or "—")] += 1
        pt = per_type.setdefault(r["gt_type"], {"n": 0, "correct": 0})
        pt["n"] += 1
        pt["correct"] += int(r["ok"])
    pred_dist: dict[str, int] = {}
    for e in evals:
        if e.pred_positive:
            for v in e.violations_brief:
                pred_dist[v["type_id"] or "?"] = pred_dist.get(v["type_id"] or "?", 0) + 1
    types = {"available": bool(trs), "n_events": len(trs),
             "accuracy": _rate(sum(r["ok"] for r in trs), len(trs)),
             "accuracy_localized": _rate(sum(r["ok"] for r in trs), sum(r["reason"] in ("ok", "wrong_type") for r in trs)),
             "confusion": tconf, "per_type": per_type, "pred_distribution": dict(sorted(pred_dist.items()))}
    # нормы
    nrm = [n for e in evals for n in e.norms]
    nrm_tp = [n for e in tp for n in e.norms]
    cat_known = [n for n in nrm if n["in_catalog"] is not None]
    norms = {"n_violations": len(nrm),
             "with_rule_rate": _rate(sum(n["n_rules"] > 0 for n in nrm), len(nrm)),
             "with_rule_rate_tp": _rate(sum(n["n_rules"] > 0 for n in nrm_tp), len(nrm_tp)),
             "verified_share": round(statistics.fmean([n["verified_share"] for n in nrm]), 4) if nrm else None,
             "url_ok_rate": _rate(sum(bool(n["url_ok"]) for n in nrm), len(nrm)),
             "in_catalog_rate": _rate(sum(bool(n["in_catalog"]) for n in cat_known), len(cat_known)) if cat_known else None,
             "type_in_catalog_rate": _rate(sum(bool(n["type_in_catalog"]) for n in nrm if n["type_in_catalog"] is not None),
                                           sum(n["type_in_catalog"] is not None for n in nrm))}
    schema = {"n": len([e for e in evals if e.status != "missing"]),
              "valid_rate": _rate(sum(not e.schema_errors for e in evals if e.status != "missing"),
                                  len([e for e in evals if e.status != "missing"])),
              "errors": {e.video_id: e.schema_errors for e in evals if e.schema_errors},
              "warnings": {e.video_id: e.schema_warnings for e in evals if e.schema_warnings}}
    lng = [x for e in evals for x in e.language]
    language = {"n": len(lng), "hedged_rate": _rate(sum(x["hedged"] for x in lng), len(lng)),
                "forbidden_rate": _rate(sum(bool(x["forbidden"]) for x in lng), len(lng)),
                "forbidden_examples": sorted({w for x in lng for w in x["forbidden"]})}
    lat_cfg = cfg.get("latency") or {}
    sla = float(lat_cfg.get("sla_s", 60.0))
    ref = float(lat_cfg.get("ref_duration_s", 145.0))

    def _sla(e: VideoEval) -> float:
        if lat_cfg.get("scale_with_duration") and e.duration:
            return sla * max(1.0, e.duration / ref)
        return sla

    lats = [e for e in evals if e.latency_s is not None]
    latency = {**_stats([e.latency_s for e in lats]), "sla_s": sla,
               "over_sla_ids": [e.video_id for e in lats if e.latency_s > _sla(e)],
               "within_sla_rate": _rate(sum(e.latency_s <= _sla(e) for e in lats), len(lats)),
               "rt_factor": _stats([e.latency_s / e.duration for e in lats if e.duration])}
    cats = {c: [e.video_id for e in evals if e.category == c] for c in CATEGORIES}
    out = {"video": video, "localization": localization, "evidence": evidence, "types": types, "norms": norms,
           "schema": schema, "language": language, "latency": latency, "categories": cats}
    if dup_check is not None:
        out["duplicates"] = {"groups": dup_check,
                             "consistent_rate": _rate(sum(d["consistent"] for d in dup_check), len(dup_check))}
    return out


def evaluate(gt: dict[str, GroundTruth], preds: dict[str, dict], runinfo: Optional[dict[str, dict]] = None,
             cfg: Optional[dict] = None, catalog: Optional[dict] = None) -> dict:
    """Главная функция: → {"summary": ..., "per_video": [...]}."""
    cfg = cfg or load_config()
    runinfo = runinfo or {}
    evals: list[VideoEval] = []
    dup_check: list[dict] = []
    for key, g in gt.items():
        members = g.members or [key]
        order = [key] + [m for m in members if m != key]
        pid = next((m for m in order if m in preds), None)
        evals.append(evaluate_video(g, preds.get(pid) if pid else None, runinfo.get(pid or key, {}), cfg, catalog, pid))
        have = [m for m in members if m in preds]
        if len(have) > 1:
            verdicts = {m: (preds[m] or {}).get("verdict") for m in have}
            tcs = {m: sorted(_num(v.get("timecode_sec")) or 0.0 for v in (preds[m] or {}).get("violations") or []
                             if isinstance(v, dict)) for m in have}
            dup_check.append({"group": key, "members": have, "verdicts": verdicts, "timecodes": tcs,
                              "consistent": len(set(verdicts.values())) == 1})
    extra = sorted(set(preds) - {m for g in gt.values() for m in (g.members or [g.video_id])}, key=video_sort_key)
    summary = summarize(evals, cfg, dup_check)
    summary["unlabeled_predictions"] = extra
    return {"summary": summary, "per_video": [asdict(e) for e in evals]}


def evaluate_dirs(labels_csv: str | Path, preds_dir: str | Path, types_csv: Optional[str | Path] = None,
                  cfg: Optional[dict] = None) -> dict:
    cfg = cfg or load_config()
    gt = load_ground_truth(labels_csv, types_csv, dedupe=bool(cfg.get("dedupe_groups", True)))
    preds, info = load_predictions(preds_dir)
    return evaluate(gt, preds, info, cfg, load_catalog(cfg))


def format_summary(s: dict) -> str:
    v, loc, ev, ty, nr, lat = s["video"], s["localization"], s["evidence"], s["types"], s["norms"], s["latency"]

    def f(x: Any) -> str:
        return "—" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))

    lines = [
        f"Видео: n={v['n']} (+{v['n_pos']} / -{v['n_neg']})  TP={v['tp']} TN={v['tn']} FP={v['fp']} FN={v['fn']} "
        f"failed={v['failed']}",
        f"  precision={f(v['precision'])} recall={f(v['recall'])} F1={f(v['f1'])} accuracy={f(v['accuracy'])}",
        f"  FN: {', '.join(v['fn_ids']) or '—'}   FP: {', '.join(v['fp_ids']) or '—'}   failed: {', '.join(v['failed_ids']) or '—'}",
        f"Локализация: |Δt| median={f(loc['abs_dt']['median'])} mean={f(loc['abs_dt']['mean'])} "
        f"signed={f(loc['mean_signed_dt'])}  hit(TP): " + " ".join(f"@{k}={f(x)}" for k, x in loc["hit_tp"].items()),
        f"  hit(все позитивы): " + " ".join(f"@{k}={f(x)}" for k, x in loc["hit_all"].items()),
        f"Кадр ±tol: TP={f(ev['ok_tp'])} все={f(ev['ok_all'])}  валидный JPEG={f(ev['valid_jpeg_rate'])} "
        f"1–3 кадра={f(ev['frames_per_violation_ok'])}",
        (f"Типы: acc={f(ty['accuracy'])} (локализованные {f(ty['accuracy_localized'])}), событий={ty['n_events']}"
         if ty["available"] else f"Типы: нет labels_types; распределение: {ty['pred_distribution']}"),
        f"Нормы: ≥1 rule={f(nr['with_rule_rate'])} verified={f(nr['verified_share'])} adilet={f(nr['url_ok_rate'])} "
        f"в каталоге={f(nr['in_catalog_rate'])}",
        f"Схема ТЗ: valid={f(s['schema']['valid_rate'])}  ошибок у {len(s['schema']['errors'])} видео; "
        f"язык: hedged={f(s['language']['hedged_rate'])} forbidden={f(s['language']['forbidden_rate'])}",
        f"Латентность: mean={f(lat['mean'])} p90={f(lat['p90'])} max={f(lat['max'])} с; > SLA {lat['sla_s']} с: "
        f"{', '.join(lat['over_sla_ids']) or '—'}",
    ]
    if s.get("duplicates") and s["duplicates"]["groups"]:
        lines.append(f"Дубликаты md5: согласованность вердиктов={f(s['duplicates']['consistent_rate'])}")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Метрики Kassa Sentinel по preds/*.json")
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--preds", required=True)
    ap.add_argument("--types", default="data/labels_types.csv")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None, help="metrics.json (по умолчанию <preds>/../metrics.json)")
    ap.add_argument("--no-dedupe", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, {"dedupe_groups": False} if args.no_dedupe else None)
    res = evaluate_dirs(args.labels, args.preds, args.types, cfg)
    out = Path(args.out) if args.out else Path(args.preds).resolve().parent / "metrics.json"
    write_json(out, res)
    print(format_summary(res["summary"]))
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
