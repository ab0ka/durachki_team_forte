from __future__ import annotations

from ml.temporal import cv as cvmod

from .conftest import make_bank, make_samples


def test_nested_cv_baseline_report(cfg):
    samples, bank = make_samples(n_pos=6, n_neg=6, T=80), make_bank()
    samples[1].group = samples[0].group                    # дубликат в одной группе
    rep = cvmod.nested_cv(samples, bank, cfg, "baseline", outer=4, inner=2, seed=0)
    assert rep["n_samples"] == 12 and rep["n_folds"] == 4
    f, n = rep["threshold_free"], rep["nested"]
    assert 0.0 <= f["auc"] <= 1.0 and f["auc"] >= 0.8
    assert n["tp"] + n["fp"] + n["fn"] + n["tn"] == 12
    assert {"hit@1s", "hit@2s", "hit@3s", "dt_median"} <= set(f)
    assert all(0.05 <= v["thr_inner"] <= 0.95 for v in rep["per_video"])
    assert set(rep["by_pooling"]) == {"max", "topk", "topk_null"}
    assert {"duration_only_auc", "spearman_neg", "warn"} <= set(rep["duration_watchdog"])
    assert {"peak", "onset"} <= set(rep["timecode_by_policy"])
    assert "разведочная" in rep["evaluation_status"]
    assert all(set(v["scores_by_pooling"]) == {"max", "topk", "topk_null"} for v in rep["per_video"])


def test_run_with_holdout_and_selection(cfg):
    samples, bank = make_samples(n_pos=6, n_neg=6, T=60), make_bank()
    rep = cvmod.run(cfg, ["baseline"], outer=3, inner=2, holdout=0.25, seed=0, samples=samples, bank=bank)
    assert rep["selected"] == "baseline"
    h = rep["holdout"]
    assert set(h["names"]).isdisjoint({v["name"] for v in rep["baseline"]["per_video"]})
    assert len(h["pred"]) == len(h["names"]) >= 2


def test_nested_cv_temporal_smoke(cfg):
    import pytest

    pytest.importorskip("torch")
    samples, bank = make_samples(n_pos=4, n_neg=4, T=60), make_bank()
    c = {**cfg, "train": {**cfg["train"], "steps_per_epoch": 3}}
    rep = cvmod.run(c, ["baseline", "temporal"], outer=2, inner=2, holdout=0.0, seed=0, epochs=2,
                    samples=samples, bank=bank)
    assert rep["selected"] in ("baseline", "temporal")
    assert rep["temporal"]["n_folds"] == rep["baseline"]["n_folds"] == 2   # одинаковые внешние фолды
    assert [v["name"] for v in rep["temporal"]["per_video"]] == [v["name"] for v in rep["baseline"]["per_video"]]
