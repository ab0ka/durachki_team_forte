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


# ----------------------------------------------------------------------------- аудит 2055fc4: окружение и полная сборка

def test_e2e_main_does_not_mutate_environ(out_dir, monkeypatch):
    monkeypatch.delenv("SENTINEL_MOCK", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    import os

    before = dict(os.environ)
    assert e2e_stub.main(["--scenarios", "clean", "--duration", "3", "--fps", "5", "--fixtures-only",
                          "--out", str(out_dir)]) == 0
    assert dict(os.environ) == before


def test_require_full_without_backend_is_not_run(out_dir, monkeypatch, capsys):
    real = e2e_stub._resolve
    monkeypatch.setattr(e2e_stub, "_resolve", lambda m, f: None if m == "backend.pipeline" else real(m, f))
    rc = e2e_stub.main(["--require-full", "--scenarios", "clean", "--duration", "3", "--fps", "5",
                        "--out", str(out_dir)])
    out = capsys.readouterr().out
    assert rc == e2e_stub.EXIT_NOT_RUN and "NOT RUN" in out and "ПОЛНАЯ СБОРКА НЕ ПРОВЕРЕНА" in out


def _fake_backend(meta: dict, verdict: str = "no_violation"):
    def run(video_path, job_id, progress=None, deadline_s=240):
        return AnalysisResult(job_id=job_id, status="done", verdict=verdict, meta=dict(meta))
    return run


def _with_backend(monkeypatch, fn):
    real = e2e_stub._resolve
    monkeypatch.setattr(e2e_stub, "_resolve", lambda m, f: fn if (m, f) == ("backend.pipeline", "run") else real(m, f))


def test_full_build_passes_with_coverage_and_explicit_mock(out_dir, monkeypatch):
    from contracts.extensions import compute_coverage

    _with_backend(monkeypatch, _fake_backend({"mock": True, "coverage": compute_coverage(3.0, 3.0)}))
    monkeypatch.setenv("SENTINEL_MOCK", "1")
    rec = e2e_stub.run_scenario("clean", out_dir, duration=3, fps=5, require_backend=True, expect_status="done")
    assert rec["runner"] == "backend.pipeline" and rec["errors"] == [] and rec["response"]["status"] == "done"


def test_backend_mock_without_explicit_mode_is_an_error(out_dir, monkeypatch):
    from contracts.extensions import compute_coverage

    _with_backend(monkeypatch, _fake_backend({"mock": True, "coverage": compute_coverage(3.0, 3.0)}))
    monkeypatch.delenv("SENTINEL_MOCK", raising=False)
    rec = e2e_stub.run_scenario("clean", out_dir, duration=3, fps=5, require_backend=True)
    assert any("production" in e for e in rec["errors"]) and rec["response"]["status"] == "failed"


def test_expect_status_catches_missing_coverage(out_dir, monkeypatch):
    _with_backend(monkeypatch, _fake_backend({}))  # оркестратор без meta.coverage
    monkeypatch.setenv("SENTINEL_MOCK", "1")
    rec = e2e_stub.run_scenario("clean", out_dir, duration=3, fps=5, require_backend=True, expect_status="done")
    assert any("ожидался status=done" in e and "meta.coverage" in e for e in rec["errors"])


def test_deadline_overrun_is_an_error(out_dir, monkeypatch):
    import time as _t

    from contracts.extensions import compute_coverage

    def slow(video_path, job_id, progress=None, deadline_s=240):
        _t.sleep(0.05)
        return AnalysisResult(job_id=job_id, status="done", verdict="no_violation",
                              meta={"coverage": compute_coverage(3.0, 3.0)})

    _with_backend(monkeypatch, slow)
    rec = e2e_stub.run_scenario("clean", out_dir, duration=3, fps=5, require_backend=True, deadline_s=0.001)
    assert any("дедлайн" in e for e in rec["errors"])
