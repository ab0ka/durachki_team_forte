"""Nested grouped CV для baseline и temporal head.

    python -m ml.temporal.cv [--kind baseline|temporal|both] [--outer loo|5] [--inner 3] [--holdout 0.2]

Внешний сплит — оценка; внутренний (только на train внешнего фолда) — калибровка порога. Скейлер/PCA/модель
подгоняются на train каждого фолда. Валидация — только оригиналы (без аугментированных копий); дубликаты —
одна группа. Отчёт: reports/temporal_cv.json. Порог, подобранный по всем OOF-оценкам («deploy»), — НЕ
независимая оценка: независимые цифры — threshold_free (AUC, таймкод) и nested (порог из внутреннего CV).
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np

from .config import load_config, reports_dir
from .dataset import Sample, fit_spec, prepare_samples
from .features import TextBank
from .metrics import recommend_threshold, summarize
from .postproc import risk_curve, video_score
from .predictors import fit_predictor
from .splits import check_split, grouped_folds, holdout_split

log = logging.getLogger("sentinel.temporal")


def _parse_folds(v: Union[str, int, None]) -> Union[str, int]:
    if v is None:
        return "loo"
    return "loo" if str(v).lower() == "loo" else int(v)


def fit_and_score(kind: str, train: Sequence[Sample], val: Sequence[Sample], bank: TextBank, cfg: dict,
                  seed: int, epochs: Optional[int] = None) -> list[tuple[float, Optional[float]]]:
    """Подогнать spec+модель на train (все версии), оценить val ТОЛЬКО по оригиналу (версия 0)."""
    spec = fit_spec([fs for s in train for fs in s.versions], bank, cfg)
    prepare_samples(list(train), bank, spec)
    val_only = [Sample(name=s.name, group=s.group, label=s.label, times=s.times, versions=s.versions[:1])
                for s in val]
    prepare_samples(val_only, bank, spec)
    model = fit_predictor(kind, train, spec, cfg, seed=seed, epochs=epochs)
    out = []
    for s in val_only:
        p, _ = model.predict(s.X[0], spec.fps)
        t, r = risk_curve(p, s.grid[0], cfg)
        out.append((video_score(r), float(t[int(np.argmax(r))]) if len(r) else None))
    return out


def nested_cv(samples: Sequence[Sample], bank: TextBank, cfg: dict, kind: str,
              outer: Union[str, int] = "loo", inner: int = 3, seed: int = 0,
              epochs: Optional[int] = None) -> dict:
    t0 = time.monotonic()
    cv = cfg["cv"]
    folds = grouped_folds(samples, outer, seed=seed)
    n = len(samples)
    scores = np.zeros(n, np.float32)
    peaks: list[Optional[float]] = [None] * n
    thr_inner = np.full(n, np.nan, np.float32)
    for fi, (tr, va) in enumerate(folds):
        check_split(samples, tr, va)
        train = [samples[i] for i in tr]
        # внутренний CV только на train внешнего фолда -> порог
        in_scores, in_labels = [], []
        n_in = min(inner, len({s.group for s in train}))
        if n_in >= 2:
            for itr, iva in grouped_folds(train, n_in, seed=seed + fi + 1):
                check_split(train, itr, iva)
                res = fit_and_score(kind, [train[i] for i in itr], [train[i] for i in iva], bank, cfg,
                                    seed=seed + fi, epochs=epochs)
                in_scores += [r[0] for r in res]
                in_labels += [train[i].label for i in iva]
        thr = recommend_threshold(in_labels, in_scores, cv.get("threshold_margin", 0.02),
                                  cv.get("min_recall_fp0", 0.8)) if len(set(in_labels)) == 2 else \
            float(cfg["infer"]["event_threshold"])
        res = fit_and_score(kind, train, [samples[i] for i in va], bank, cfg, seed=seed + fi, epochs=epochs)
        for i, (sc, tp) in zip(va, res):
            scores[i], peaks[i], thr_inner[i] = sc, tp, thr
        log.info("cv[%s] фолд %d/%d готов (%.1f с)", kind, fi + 1, len(folds), time.monotonic() - t0)

    labels = [s.label for s in samples]
    tol = cv.get("hit_tolerances_s", [1, 2, 3])
    free = summarize(labels, scores.tolist(), peaks, [s.times for s in samples], tol)
    free["note"] = ("recall_at_fp*/thr_fp* — с «оракульным» порогом по тем же OOF (оптимистично); "
                    "auc и таймкод — независимы от порога")
    pred = scores > thr_inner
    y = np.asarray(labels, bool)
    tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    fn, tn = int((~pred & y).sum()), int((~pred & ~y).sum())
    nested = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
              "recall": tp / max(tp + fn, 1), "fpr": fp / max(fp + tn, 1),
              "precision": tp / max(tp + fp, 1) if tp + fp else None, "accuracy": (tp + tn) / max(n, 1),
              "note": "порог каждого внешнего фолда подобран внутренним CV только на его train — независимая оценка"}
    per = []
    for i, s in enumerate(samples):
        dt = min((abs(peaks[i] - t) for t in s.times), default=None) if (s.label and peaks[i] is not None) else None
        per.append({"name": s.name, "group": s.group, "label": s.label, "times": s.times,
                    "sources": [{"name": x["name"], "label": x["label"], "times": x["times"]} for x in s.sources],
                    "conflict": s.conflict, "score": round(float(scores[i]), 4), "t_peak": peaks[i],
                    "dt": None if dt is None else round(float(dt), 2), "thr_inner": round(float(thr_inner[i]), 4),
                    "pred": bool(pred[i])})
    return {"kind": kind, "protocol": "nested_grouped_cv", "outer": outer, "inner": inner, "seed": seed,
            "n_folds": len(folds), "n_samples": n, "elapsed_s": round(time.monotonic() - t0, 1),
            "threshold_free": free, "nested": nested,
            "deploy_threshold": recommend_threshold(labels, scores.tolist(), cv.get("threshold_margin", 0.02),
                                                    cv.get("min_recall_fp0", 0.8)),
            "deploy_threshold_note": "подобран по всем OOF-оценкам — для прод-порога, не для оценки качества",
            "per_video": per}


def _select(reports: dict) -> str:
    def key(r: dict) -> tuple:
        f = r["threshold_free"]
        return (f.get("auc") or 0.0, r["nested"]["recall"] - r["nested"]["fpr"], -(f.get("dt_median") or 99))
    kinds = [k for k in ("baseline", "temporal") if k in reports]
    best = max(kinds, key=lambda k: key(reports[k]))
    # temporal выбираем, только если он строго лучше baseline по AUC (иначе — проще и стабильнее baseline)
    if best == "temporal" and "baseline" in reports and key(reports["temporal"])[0] <= key(reports["baseline"])[0]:
        best = "baseline"
    return best


def run(cfg: dict, kinds: Sequence[str], outer: Union[str, int, None], inner: int, holdout: float, seed: int,
        index: Optional[str] = None, max_aug: Optional[int] = None, epochs: Optional[int] = None,
        samples: Optional[list[Sample]] = None, bank: Optional[TextBank] = None) -> dict:
    if samples is None or bank is None:
        from .data_index import bank_for_index, load_samples

        samples, idx = load_samples(cfg, index, max_aug=max_aug)
        bank = bank_for_index(cfg, idx)
    rest, hold = holdout_split(samples, holdout, seed=seed)
    if hold:
        check_split(samples, rest, hold)
    rest_s = [samples[i] for i in rest]
    report: dict[str, Any] = {"n_groups": len(samples), "n_pos": sum(s.label for s in samples),
                              "conflicts": [s.name for s in samples if s.conflict]}
    for kind in kinds:
        o = outer if outer is not None else (cfg["cv"]["outer_temporal"] if "temporal" in kinds else cfg["cv"]["outer"])
        report[kind] = nested_cv(rest_s, bank, cfg, kind, _parse_folds(o), inner, seed,
                                 epochs=epochs or (cfg["cv"].get("epochs") if kind == "temporal" else None))
    report["selected"] = _select({k: report[k] for k in kinds})
    if hold:  # нетронутый holdout: одна оценка выбранной модели, порог — из CV на rest
        kind = report["selected"]
        res = fit_and_score(kind, rest_s, [samples[i] for i in hold], bank, cfg, seed=seed, epochs=epochs)
        thr = report[kind]["deploy_threshold"]
        hs = [samples[i] for i in hold]
        report["holdout"] = {
            "kind": kind, "threshold": thr, "names": [s.name for s in hs],
            **summarize([s.label for s in hs], [r[0] for r in res], [r[1] for r in res], [s.times for s in hs],
                        cfg["cv"].get("hit_tolerances_s", [1, 2, 3])),
            "pred": [bool(r[0] > thr) for r in res], "labels": [s.label for s in hs],
            "note": "holdout не участвовал ни в подгонке, ни в выборе порога/модели",
        }
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="ml.temporal: nested grouped CV")
    ap.add_argument("--index", default=None, help="cache/features/index.json")
    ap.add_argument("--kind", default="both", choices=["baseline", "temporal", "both"])
    ap.add_argument("--outer", default=None, help="loo | число фолдов")
    ap.add_argument("--inner", type=int, default=None)
    ap.add_argument("--holdout", type=float, default=None)
    ap.add_argument("--max-aug", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config()
    kinds = ["baseline", "temporal"] if a.kind == "both" else [a.kind]
    if "temporal" in kinds:
        try:
            import torch  # noqa: F401
        except ImportError:
            log.warning("torch не установлен — temporal head пропущен, только baseline")
            kinds = ["baseline"]
    rep = run(cfg, kinds, a.outer, a.inner or int(cfg["cv"]["inner"]),
              cfg["cv"]["holdout_frac"] if a.holdout is None else a.holdout,
              cfg["train"]["seed"] if a.seed is None else a.seed, a.index, a.max_aug, a.epochs)
    out = Path(a.out) if a.out else reports_dir(cfg) / "temporal_cv.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    for k in kinds:
        f, nst = rep[k]["threshold_free"], rep[k]["nested"]
        print(f"[{k}] AUC={f['auc']} recall@FP0(oracle)={f['recall_at_fp0']:.2f} "
              f"nested: recall={nst['recall']:.2f} FPR={nst['fpr']:.2f} |dt| median={f['dt_median']} "
              f"hit@2s={f.get('hit@2s')} ({rep[k]['elapsed_s']} с)")
    print(f"selected={rep['selected']} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
