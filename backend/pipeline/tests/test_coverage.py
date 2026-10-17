"""Полнота: meta.coverage, гейт покрытия, completeness в формате A01; статичное нарушение не теряется."""
from __future__ import annotations

from backend.pipeline import mocks, run
from backend.pipeline.contract import Event, SceneState
from backend.pipeline.coverage import interval_coverage


def test_interval_coverage_basics():
    assert interval_coverage([i * 0.2 for i in range(50)], 10.0) == 1.0
    assert interval_coverage([i * 0.2 for i in range(25)], 10.0) == 0.5          # только первая половина
    gap = [i * 0.2 for i in range(20)] + [8 + i * 0.2 for i in range(10)]        # провал 4..8 с
    assert 0.55 < interval_coverage(gap, 10.0) < 0.65
    assert interval_coverage([], 10.0) == 0.0 and interval_coverage([1.0], None) is None
    assert interval_coverage([i * 5.0 for i in range(3)], 15.0, max_gap_s=2.0) < 0.5  # редкие кадры ≠ покрытие


def test_meta_completeness_coverage_provenance(fake_ml, synth_video):
    fake_ml()
    res = run(synth_video, "job-cov", None, 60)
    m = res.meta
    assert res.status == "done" and m["completeness"]["complete"] is True
    assert set(m["completeness"]) == {"complete", "analyzed_until_sec", "duration_sec", "stages",
                                      "required_stages", "reason"}
    assert m["completeness"]["stages"]["detect"] in ("ok", "mock")
    assert m["duration_sec"] and m["duration_sec"] > 3
    assert m["coverage"]["ingest"] >= 0.9 and m["coverage"]["scene"] >= 0.9
    assert m["coverage"]["vlm_discover"] is None                      # неизвестно ≠ 0 и ≠ 1
    assert m["provenance"]["pipeline_version"] and "sources" in m["provenance"]


def _half_scene(clip, dets, tracks, zones, pose_states):
    half = clip.times[: len(clip.times) // 2]
    return [SceneState(t, {"n_persons": 1.0}) for t in half]


def test_partial_scene_coverage_blocks_no_violation(fake_ml, synth_video):
    fake_ml({"ml.scene.scene_states": _half_scene, "ml.rules.run_rules": lambda *a: [],
             "ml.temporal.score": lambda clip: ([], [])})
    res = run(synth_video, "job-halfcov", None, 60)
    assert res.meta["coverage"]["scene"] < 0.9
    assert res.status == "failed" and res.verdict is None
    assert "покрытие" in res.meta["error"] and res.meta["completeness"]["complete"] is False


def test_coverage_gate_can_be_disabled(fake_ml, synth_video):
    fake_ml({"ml.scene.scene_states": _half_scene, "ml.rules.run_rules": lambda *a: [],
             "ml.temporal.score": lambda clip: ([], [])})
    res = run(synth_video, "job-halfcov2", None, 60, overrides={"coverage": {"gate": False}})
    assert res.status == "done" and res.verdict == "no_violation"


def _static_rules(states, tracks, zones, catalog):
    # «сейф открыт без присмотра»: в кадре ничего не двигается — движение не является условием
    return [Event("V03", 1.0, 3.0, 1.0, 0.8, "rules", ["сейф открыт, у сейфа никого нет"])]


def test_static_violation_survives_zero_motion_and_low_risk(fake_ml, synth_video):
    verified = {}

    def verify(clip, events, budget_s):
        verified["n"] = len(events)
        return []

    fake_ml({"ml.scene.scene_states": lambda clip, *a: [SceneState(t, {"motion_energy": 0.0}) for t in clip.times],
             "ml.rules.run_rules": _static_rules,
             "ml.temporal.score": lambda clip: ([(t, 0.0) for t in clip.times], []),   # слабый SigLIP-риск
             "ml.vlm.verify": verify})
    res = run(synth_video, "job-static", None, 60)
    assert verified.get("n") == 1, "кандидат правил дошёл до VLM, несмотря на нулевой риск"
    assert res.status == "done" and res.verdict == "violation"
    assert res.violations[0].type_id == "V03"


def test_inference_does_not_read_labels(fake_ml, synth_video, tmp_path, monkeypatch):
    import builtins

    opened = []
    real_open = builtins.open

    def spy(file, *a, **k):
        opened.append(str(file))
        return real_open(file, *a, **k)

    fake_ml()
    monkeypatch.setattr(builtins, "open", spy)
    run(synth_video, "job-nogt", None, 60)
    assert not [f for f in opened if "label" in f.lower() or f.lower().endswith(".csv")]
