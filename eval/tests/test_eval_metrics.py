from __future__ import annotations

from pathlib import Path

import pytest

from eval.common import load_config, write_json
from eval.labels import LabelRow, load_ground_truth, write_labels_csv
from eval.metrics import evaluate, evaluate_dirs, format_scorecard, format_summary, match_one_to_one, wilson
from eval.schema import check_api_response
from eval.score_like_jury import format_score, score

from .conftest import make_pred, make_violation


def _labels(tmp_path: Path) -> Path:
    rows = [
        LabelRow("X-1", "violation", [22.0], md5="a"),
        LabelRow("X-2", "violation", [30.0], md5="b"),
        LabelRow("X-3", "no_violation", [], md5="c"),
        LabelRow("X-6", "no_violation", [], md5="d"),
        LabelRow("X-4", "violation", [57.0], md5="dup", raw="57 секунда"),
        LabelRow("X-47", "violation", [68.0], md5="dup", raw="1минута 8 секунда"),
        LabelRow("X-7", "violation", [40.0], md5="e"),
        LabelRow("X-8", "violation", [15.0], md5="f"),
        LabelRow("X-9", "violation", [50.0], md5="g"),
        LabelRow("X-10", "violation", [5.0], md5="lab"),
        LabelRow("X-11", "no_violation", [], md5="lab"),
    ]
    from eval.labels import assign_groups

    assign_groups(rows)
    p = tmp_path / "labels.csv"
    write_labels_csv(rows, p)
    return p


def _preds() -> dict:
    return {
        "X-1": make_pred("j1", [make_violation(22.4)]),
        "X-2": make_pred("j2", []),
        "X-3": make_pred("j3", [make_violation(10.0)]),
        "X-6": make_pred("j6", []),
        "X-4": make_pred("j4", [make_violation(67.5, frames=[67.5])]),
        "X-7": make_pred("j7", [make_violation(48.0)]),
        "X-8": make_pred("j8", [make_violation(15.5), make_violation(16.0), make_violation(90.0)]),
        "X-9": {"job_id": "j9", "status": "failed", "verdict": None, "violations": [],
                "meta": {"error": "deadline exceeded"}},
        "X-10": make_pred("j10", [make_violation(5.0)]),
    }


@pytest.fixture
def result(tmp_path: Path) -> dict:
    gt = load_ground_truth(_labels(tmp_path))
    return evaluate(gt, _preds(), {"X-1": {"latency_s": 12.0}, "X-2": {"latency_s": 75.0}}, load_config())


def test_wilson():
    assert wilson(0, 0) is None
    lo, hi = wilson(19, 20)
    assert 0.7 < lo < 0.95 and hi <= 1.0


def test_binary(result):
    v = result["summary"]["video"]
    assert (v["tp"], v["fn"], v["fp"], v["tn"]) == (4, 2, 1, 1)
    assert v["failed"] == 1 and v["timeouts"] == 1 and v["timeout_ids"] == ["X-9"]
    assert v["label_conflict_ids"] == ["X-10"]          # конфликт меток — не выбираем, исключаем
    assert v["n"] == 8 and v["recall"] == pytest.approx(4 / 6, abs=1e-3)
    assert v["recall_ci95"][0] < v["recall"] < v["recall_ci95"][1]
    assert set(v["fn_ids"]) == {"X-2", "X-9"} and v["fp_ids"] == ["X-3"]


def test_one_to_one_and_conflict_alternatives(result):
    by = {e["video_id"]: e for e in result["per_video"]}
    x4 = by["X-4"]                       # одно событие с альтернативами [57, 68]
    assert x4["events_total"] == 1 and x4["events_matched"] == 1
    assert x4["dt_abs"] == pytest.approx(0.5) and x4["matched_gt_t"] == 68.0 and x4["evidence_ok"]
    x8 = by["X-8"]
    assert x8["events_matched"] == 1 and x8["extra_candidates"] == 2 and x8["dt_abs"] == pytest.approx(0.5)
    x7 = by["X-7"]                       # вне окна сопоставления: не hit, но «поздно» и справочный Δt
    assert x7["category"] == "late" and x7["events_matched"] == 0 and x7["nearest_dt_signed"] == 8.0
    assert by["X-1"]["category"] == "TP" and by["X-1"]["hits"]["0.5"]
    loc = result["summary"]["localization"]
    assert loc["extra_candidates_total"] == 3 and "неполная" in loc["caveat"]   # X-8: 2, X-7: 1 (вне окна)
    assert result["summary"]["localization_no_conflict"]["events_total"] == loc["events_total"] - 1


def test_match_one_to_one_unique():
    pairs = match_one_to_one([[10.0], [11.0]], [(0, 10.4)], 5.0)
    assert len(pairs) == 1 and pairs[0]["event"] == 0
    pairs = match_one_to_one([[10.0]], [(0, 10.4), (1, 9.9)], 5.0)
    assert len(pairs) == 1 and pairs[0]["pred"] == 1


def test_types_only_with_real_labels(result, tmp_path: Path):
    assert result["summary"]["types"]["available"] is False and result["summary"]["types"]["accuracy"] is None
    s = score(result)
    assert s["components"]["type"]["points"] is None and "type" in s["missing"]
    types = tmp_path / "labels_types.csv"
    types.write_text("video_id,t_sec,type_id\nX-1,22,V01\nX-8,15,V05\n", encoding="utf-8")
    gt = load_ground_truth(tmp_path / "labels.csv", types)
    res = evaluate(gt, _preds(), {}, load_config())
    ty = res["summary"]["types"]
    assert ty["available"] and ty["n_events"] == 2 and ty["accuracy"] == 0.5
    assert {e["video_id"]: e["category"] for e in res["per_video"]}["X-8"] == "wrong_type"
    assert score(res)["components"]["type"]["source"] == "measured"


def test_norms_and_language(result):
    nr = result["summary"]["norms"]
    assert nr["with_rule_rate"] == 1.0 and nr["url_ok_rate"] == 1.0 and "≠" in nr["note"]
    assert result["summary"]["language"]["hedged_rate"] == 1.0


def test_latency_and_scorecard(result):
    lat = result["summary"]["latency"]
    assert lat["over_sla_ids"] == ["X-2"] and lat["max"] == 75.0
    sc = result["scorecard"]
    assert sc["n_videos"] == 8 and sc["timeouts"] == 1 and sc["versions"]["eval_version"]
    txt = format_scorecard(sc) + format_summary(result["summary"])
    assert "SCORECARD" in txt and "recall" in txt


def test_proxy_score_perfect_and_disclaimer(tmp_path: Path):
    rows = [LabelRow("X-1", "violation", [22.0]), LabelRow("X-2", "no_violation", [])]
    p = tmp_path / "l.csv"
    write_labels_csv(rows, p)
    gt = load_ground_truth(p)
    res = evaluate(gt, {"X-1": make_pred("a", [make_violation(22.0)]), "X-2": make_pred("b", [])},
                   {"X-1": {"latency_s": 5}, "X-2": {"latency_s": 5}}, load_config())
    s = score(res)
    assert s["proxy_score"] == pytest.approx(100.0) and "неизвестна" in s["disclaimer"]
    assert set(s["missing"]) == {"type", "robustness"}
    assert "PROXY_SCORE" in format_score(s)
    bad = evaluate(gt, {"X-1": make_pred("a", []), "X-2": make_pred("b", [make_violation(3.0)])}, {}, load_config())
    assert score(bad)["components"]["detection"]["value"] == 0.0


def test_evaluate_dirs_and_wrapper(tmp_path: Path):
    lab = _labels(tmp_path)
    pdir = tmp_path / "preds"
    for vid, d in _preds().items():
        write_json(pdir / f"{vid}.json", {"video_id": vid, "latency_s": 3.0, "response": d} if vid == "X-1" else d)
    write_json(pdir / "_runinfo.json", {"X-2": {"latency_s": 4.0}})
    res = evaluate_dirs(lab, pdir, None, load_config(), subset={"X-1", "X-2"})
    assert res["summary"]["video"]["n"] == 2
    lat = {e["video_id"]: e["latency_s"] for e in res["per_video"]}
    assert lat == {"X-1": 3.0, "X-2": 4.0}
    assert res["scorecard"]["versions"]["labels_sha256"]


def test_schema_errors():
    assert check_api_response(make_pred("a", [make_violation(1.0)])).valid
    bad = make_pred("a", [make_violation(1.0)])
    del bad["violations"][0]["description"]
    bad["violations"][0]["evidence_frames"][0]["image_base64"] = "not-base64!!"
    rep = check_api_response(bad)
    assert not rep.valid and any("description" in e for e in rep.errors)
    assert any("JPEG" in w for w in rep.warnings)
    rep = check_api_response({"job_id": "x", "status": "done", "verdict": "maybe", "violations": []})
    assert any("verdict" in e for e in rep.errors)
