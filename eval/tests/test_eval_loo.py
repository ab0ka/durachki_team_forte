from __future__ import annotations

from eval.loo import fit_threshold, logo, nested_cv, video_score


def _rows():
    rows = []
    for i in range(10):
        pos = i % 2 == 0
        rows.append({"video_id": f"X-{i}", "group": f"X-{i}", "label": "violation" if pos else "no_violation",
                     "scores_raw": {"max": 0.8 + i * 0.01 if pos else 0.2 + i * 0.01}})
    rows.append({"video_id": "X-47", "group": "X-0", "label": "violation", "scores_raw": {"max": 0.81}})  # дубль
    return rows


def test_threshold_separates():
    th = fit_threshold(_rows())
    assert 0.3 < th < 0.8
    assert video_score({"scores_raw": {"candidates": [{"score": 0.4}, {"score": 0.9}]}}) == 0.9


def test_logo_exploratory_and_nested():
    r = logo(_rows())
    assert r["status"] == "exploratory" and "не независимый" in r["note"]
    assert r["n"] == 10 and r["recall"] == 1.0 and r["fp"] == 0      # дубль X-47 не утекает и не двоится
    n = nested_cv(_rows(), folds=3)
    assert n["status"] == "nested_cv" and n["n"] == 10 and n["recall"] == 1.0
