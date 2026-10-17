from __future__ import annotations

import pytest

from contracts.sentinel_contracts import AnalysisResult, Detection, Event, PoseFrame, SceneState, Track, Zone
from testkit import contract_checks as cc
from testkit import e2e_stub


def test_checks_catch_bad_outputs():
    assert cc.check_detections([Detection(0.0, "dog", (0.1, 0.1, 0.2, 0.2), 0.5)])
    assert cc.check_detections([Detection(0.0, "person", (0.3, 0.1, 0.2, 1.5), 0.5)])
    assert cc.check_detections([Detection(0.0, "person", (0.1, 0.1, 0.2, 0.2), 1.5)])
    assert cc.check_detections("nope")
    assert cc.check_tracks_zones(([Track(1, "person", role="boss")], [Zone("kitchen", [(0, 0), (1, 0), (1, 1)])]))
    assert cc.check_tracks_zones(([], [Zone("desk", [(0, 0), (1, 0)])]))
    assert cc.check_pose_frames([PoseFrame(0.0, 1, [(0.1, 0.1, 0.9)] * 5)])
    assert cc.check_scene_states([SceneState(0.0, {"n_persons": 1.5})])
    assert cc.check_scene_states([SceneState(0.0, {"safe_open": 2.0, "bogus": 1.0})])
    assert cc.check_scene_states([SceneState(0.0, {"x_cashier_lost_in_view": 1.0, "scene_visibility": 0.2})]) == []
    assert cc.check_detections([Detection(0.0, "personal_item", (0.1, 0.1, 0.2, 0.2), 0.5)]) == []
    assert cc.check_scene_states([SceneState(1.0, {}), SceneState(0.5, {})])
    assert cc.check_events([Event("X1", 0, 1, 2, 0.5, "rules")])
    assert cc.check_events([Event("V01", 0, 1, 0.5, 0.5, "magic")])
    assert cc.check_events([Event("V01", 0, 1, 0.5, 0.5, "rules")], duration=10) == []
    assert cc.check_risk([(0.0, 1.5)]) and cc.check_risk([(0.0, 0.5)]) == []
    assert cc.check_result({"job_id": "x"})
    assert cc.check_result(AnalysisResult(job_id="j", status="done", verdict="no_violation")) == []


def test_fake_clip_passes_clip_check():
    pytest.importorskip("cv2")
    from testkit.fake_clip import FakeClip

    assert cc.check_clip(FakeClip.from_scenario("clean", duration=2, fps=10, size=(116, 212))) == []
    assert cc.check_clip(object())


@pytest.fixture
def out_dir(tmp_path):
    pytest.importorskip("cv2")
    return tmp_path


@pytest.mark.parametrize("scenario", ["clean", "V02", "V07"])
def test_e2e_fixtures_only(out_dir, scenario):
    rec = e2e_stub.run_scenario(scenario, out_dir, duration=6, fps=5, use_backend=False, prefer_real=False)
    assert rec["errors"] == [] and rec["verdict_ok"] and rec["type_ok"]
    assert all(s["source"] == "fixture" for s in rec["stages"])


def test_e2e_broken_module_falls_back_or_fails_strict(out_dir, monkeypatch):
    def broken(*a, **kw):
        raise RuntimeError("boom")

    real = e2e_stub._resolve
    monkeypatch.setattr(e2e_stub, "_resolve", lambda m, f: broken if m == "ml.detect" else real(m, f))
    rec = e2e_stub.run_scenario("V01", out_dir, duration=6, fps=5, use_backend=False)
    det = next(s for s in rec["stages"] if s["name"] == "detect")
    assert det["source"] == "real->fixture" and "boom" in det["error"] and rec["errors"] == []
    rec = e2e_stub.run_scenario("V01", out_dir, duration=6, fps=5, use_backend=False, strict=True)
    assert rec["errors"] and "detect" in rec["errors"][0]


def test_e2e_invalid_module_output_is_rejected(out_dir, monkeypatch):
    real = e2e_stub._resolve
    monkeypatch.setattr(e2e_stub, "_resolve",
                        lambda m, f: (lambda *a: [Detection(0.0, "dog", (0, 0, 1, 1), 0.9)]) if m == "ml.detect"
                        else real(m, f))
    rec = e2e_stub.run_scenario("clean", out_dir, duration=4, fps=5, use_backend=False)
    det = next(s for s in rec["stages"] if s["name"] == "detect")
    assert det["source"] == "real->fixture" and det["problems"]


def test_e2e_main(out_dir):
    assert e2e_stub.main(["--scenarios", "clean,V05", "--duration", "5", "--fps", "5", "--fixtures-only",
                          "--require-accuracy", "--out", str(out_dir)]) == 0
    assert (out_dir / "e2e_report.json").exists()
