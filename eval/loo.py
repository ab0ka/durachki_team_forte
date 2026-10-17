"""Leave-one-group-out и nested CV для подбора порога вердикта по сырым скорам (scores_raw.jsonl из eval.run --raw).

* LOGO по данным, на ошибках которых много раз меняли пайплайн, — **exploratory**, НЕ независимый тест;
* nested CV: внешние групповые фолды, порог подбирается только на outer-train → оценка процедуры подбора;
* итоговые цифры — только нетронутый holdout (eval.run --subset holdout --final).

Скор видео = max по кандидатам (``scores_raw.candidates[].score`` / ``scores_raw.max`` / confidence нарушений).
Если у ml.fusion.tune (A12) есть ``fit(rows) -> params`` и ``predict(params, row) -> bool``, используется он.

CLI::

    python -m eval.loo --scores runs/eval/dev1/scores_raw.jsonl [--folds 5]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Callable, Optional

from .common import write_json

EXPLORATORY = "exploratory: LOGO на данных, по ошибкам которых настраивали пайплайн, — не независимый тест"


def video_score(row: dict) -> float:
    sr = row.get("scores_raw")
    vals: list[float] = []
    if isinstance(sr, dict):
        for k in ("max", "score", "risk_max"):
            if isinstance(sr.get(k), (int, float)):
                vals.append(float(sr[k]))
        for c in sr.get("candidates") or sr.get("events") or []:
            if isinstance(c, dict) and isinstance(c.get("score", c.get("confidence")), (int, float)):
                vals.append(float(c.get("score", c.get("confidence"))))
    elif isinstance(sr, list):
        vals += [float(c.get("score", c.get("confidence", 0)) or 0) for c in sr if isinstance(c, dict)]
    vals += [float(v.get("score") or 0) for v in row.get("violations") or [] if isinstance(v, dict)]
    return max(vals) if vals else 0.0


def fit_threshold(rows: list[dict], fn_cost: float = 2.0, fp_cost: float = 1.0) -> float:
    """Порог с минимальной ценой ошибок (пропуск дороже); середина между соседними скорами."""
    pts = sorted({video_score(r) for r in rows})
    cands = [pts[0] - 1e-6] + [(a + b) / 2 for a, b in zip(pts, pts[1:])] + [pts[-1] + 1e-6] if pts else [0.5]

    def cost(th: float) -> float:
        c = 0.0
        for r in rows:
            pos, pred = r.get("label") == "violation", video_score(r) >= th
            c += fn_cost * (pos and not pred) + fp_cost * ((not pos) and pred)
        return c

    return min(cands, key=lambda th: (cost(th), th))


def _builtin() -> tuple[Callable, Callable]:
    return (lambda rows: {"threshold": fit_threshold(rows)},
            lambda params, row: video_score(row) >= params["threshold"])


def _tuner() -> tuple[Callable, Callable, str]:
    try:
        from ml.fusion import tune as t  # type: ignore

        if callable(getattr(t, "fit", None)) and callable(getattr(t, "predict", None)):
            return t.fit, t.predict, "ml.fusion.tune"
    except Exception:  # noqa: BLE001
        pass
    f, p = _builtin()
    return f, p, "builtin-threshold"


def _metrics(pairs: list[tuple[bool, bool]]) -> dict:
    tp = sum(g and p for g, p in pairs)
    fn = sum(g and not p for g, p in pairs)
    fp = sum((not g) and p for g, p in pairs)
    tn = sum((not g) and (not p) for g, p in pairs)
    return {"n": len(pairs), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "recall": round(tp / (tp + fn), 4) if tp + fn else None,
            "precision": round(tp / (tp + fp), 4) if tp + fp else None}


def logo(rows: list[dict]) -> dict:
    fit, predict, name = _tuner()
    groups = sorted({r.get("group") or r["video_id"] for r in rows if r.get("label") in ("violation", "no_violation")})
    pairs = []
    for g in groups:
        train = [r for r in rows if (r.get("group") or r["video_id"]) != g and r.get("label")]
        test = [r for r in rows if (r.get("group") or r["video_id"]) == g and r.get("label")]
        params = fit(train)
        pairs += [(r["label"] == "violation", bool(predict(params, r))) for r in test[:1]]  # группа = одно видео
    return {"status": "exploratory", "note": EXPLORATORY, "tuner": name, **_metrics(pairs)}


def nested_cv(rows: list[dict], folds: int = 5, seed: int = 2026) -> dict:
    fit, predict, name = _tuner()
    lab = [r for r in rows if r.get("label") in ("violation", "no_violation")]
    by_g: dict[str, list[dict]] = {}
    for r in lab:
        by_g.setdefault(r.get("group") or r["video_id"], []).append(r)
    strata = {True: [], False: []}
    for g, rs in by_g.items():
        strata[any(r["label"] == "violation" for r in rs)].append(g)
    rng = random.Random(seed)
    fold_of: dict[str, int] = {}
    for gs in strata.values():
        gs.sort()
        rng.shuffle(gs)
        for i, g in enumerate(gs):
            fold_of[g] = i % folds
    per_fold, pairs = [], []
    for k in range(folds):
        train = [r for g, rs in by_g.items() if fold_of[g] != k for r in rs]
        test = [rs[0] for g, rs in by_g.items() if fold_of[g] == k]
        if not test or not train:
            continue
        params = fit(train)
        fp = [(r["label"] == "violation", bool(predict(params, r))) for r in test]
        pairs += fp
        per_fold.append({"fold": k, **_metrics(fp), "params": params})
    return {"status": "nested_cv", "note": "порог подбирается только на outer-train; оценка процедуры, не holdout",
            "tuner": name, "folds": folds, **_metrics(pairs), "per_fold": per_fold}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="LOGO (exploratory) и nested CV порога вердикта")
    ap.add_argument("--scores", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    rows: list[dict[str, Any]] = [json.loads(x) for x in Path(args.scores).read_text(encoding="utf-8").splitlines() if x.strip()]
    res = {"logo": logo(rows), "nested_cv": nested_cv(rows, args.folds)}
    write_json(args.out or Path(args.scores).with_name("loo.json"), res)
    print(EXPLORATORY)
    for k, v in res.items():
        print(f"{k}: n={v['n']} recall={v['recall']} precision={v['precision']} FN={v['fn']} FP={v['fp']} ({v['tuner']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
