import json
from pathlib import Path

import pytest

from ml.vlm.compare import load_candidates, summarize
from ml.vlm.tests.test_vlm_cli_api import _write_video


def _ev(tid, t, conf, source="rules"):
    return {"type_id": tid, "t_start": t - 1, "t_end": t + 1, "t_peak": t, "confidence": conf, "source": source,
            "facts": [], "bboxes": {}, "track_ids": [], "extra": {}}


@pytest.fixture
def setup(tmp_path):
    vids = tmp_path / "videos"
    vids.mkdir()
    _write_video(vids / "X-1.avi", seconds=6.0)   # нарушение, кандидат слабый (FN без VLM)
    _write_video(vids / "X-2.avi", seconds=5.0)   # чисто, кандидат сильный (FP без VLM)
    _write_video(vids / "X-3.avi", seconds=4.0)   # чисто, кандидатов нет (TN)
    (tmp_path / "labels.csv").write_text("file,label,t\nX-1.mov,нарушение,3\nX-2.mov,без нарушения,\n"
                                         "X-3.mov,без нарушения,\n", encoding="utf-8")
    cands = tmp_path / "cands"
    for name, events in (("X-1", [_ev("V05", 3.5, 0.4)]), ("X-2", [_ev("V05", 2.0, 0.6)]), ("X-3", [])):
        d = cands / name
        d.mkdir(parents=True)
        (d / "events_rules.json").write_text(json.dumps(events), encoding="utf-8")
        (d / "events_temporal.json").write_text("[]", encoding="utf-8")
        (d / "zones.json").write_text(json.dumps([{"name": "desk", "polygon": [[0.1, 0.3], [0.9, 0.3], [0.9, 0.6]],
                                                   "conf": 0.9}]), encoding="utf-8")
    return tmp_path


def test_load_candidates_dir_and_jsonl(setup, tmp_path):
    c = load_candidates(str(setup / "cands"))
    assert set(c) == {"x-1", "x-2", "x-3"} and c["x-1"]["events"][0].confidence == 0.4
    assert c["x-1"]["zones"][0].name == "desk"
    jl = tmp_path / "c.jsonl"
    jl.write_text(json.dumps({"video": "Х-1.MOV", "events": [_ev("V02", 1, 0.5)], "zones": []}) + "\n",
                  encoding="utf-8")
    assert load_candidates(str(jl))["x-1"]["events"][0].type_id == "V02"


def test_compare_counts_fixes_and_breaks(setup, monkeypatch, capsys):
    from ml.vlm.compare import main
    monkeypatch.setenv("SENTINEL_VLM_MOCK_MODE", "confirm")
    rc = main(["--videos", str(setup / "videos"), "--labels", str(setup / "labels.csv"), "--candidates",
               str(setup / "cands"), "--modes", "none,json,qa", "--dump-dir", str(setup / "dump"),
               "--out", str(setup / "rep.json")])
    out = capsys.readouterr().out
    rep = json.loads(out)
    assert rc == 0 and rep["videos"] == 3
    none, js = rep["summary"]["none"], rep["summary"]["json"]
    assert (none["tp"], none["fn"], none["fp"], none["tn"]) == (0, 1, 1, 1)
    assert js["vs_none"]["fn_fixed"] == 1 and js["vs_none"]["fp_created"] == 0  # mock «confirm» поднял слабый
    assert js["timecode_hit"]["<=1s"] == 1.0 and js["vlm_calls_total"] >= 2
    assert "qa" in rep["summary"] and "note" in rep
    assert "X-1" not in out  # построчно — только в локальном дампе
    assert (Path(rep["dump_dir"]) / "rows.json").exists()


def test_compare_reject_fixes_fp(setup, monkeypatch, capsys):
    from ml.vlm.compare import main
    monkeypatch.setenv("SENTINEL_VLM_MOCK_MODE", "reject")
    main(["--videos", str(setup / "videos"), "--labels", str(setup / "labels.csv"), "--candidates",
          str(setup / "cands"), "--modes", "none,json", "--no-dump"])
    js = json.loads(capsys.readouterr().out)["summary"]["json"]
    assert js["vs_none"]["fp_fixed"] == 1 and js["fp"] == 0


def test_summarize_unknown_not_negative():
    rows = [{"gt": True, "t_gt": 3.0, "modes": {"none": {"verdict": None, "t": None, "vlm_s": 0.0}}}]
    s = summarize(rows, ["none"])["none"]
    assert s["unknown_verdicts"] == 1 and s["fn"] == 0 and s["n"] == 0
