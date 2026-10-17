"""Слияние, дедуп, гейт «факт + норма», пороги, таймкоды, timeline."""
from __future__ import annotations

import json

import pytest

from ml.fusion import fuse
from ml.fusion.catalog import resolve_catalog
from ml.fusion.config import load_config, load_thresholds
from ml.fusion.core import fuse_scores, timecode, vlm_opinion

from .conftest import FakeClip, ev


def _types(catalog):
    return resolve_catalog(catalog)[0]


def run(events, catalog, risk=(), duration=60.0, cfg=None, thr=None):
    return fuse_scores(events, list(risk), duration, _types(catalog), cfg or load_config(), thr or load_thresholds())


def test_same_type_within_gap_merged(catalog):
    out = run([ev(t=10.0), ev(t=12.5, conf=0.6)], catalog)
    assert len(out.reported) == 1
    c = out.reported[0]
    assert c.n_events == 2 and c.t_start <= 9.5 and c.t_end >= 13.0


def test_same_type_far_apart_not_merged(catalog):
    out = run([ev(t=10.0), ev(t=20.0)], catalog)
    assert [c.t_peak for c in out.reported] == [10.0, 20.0]


def test_different_types_not_merged(catalog):
    out = run([ev("V01", t=10.0), ev("V03", t=30.0, facts=["дверца сейфа открыта"])], catalog)
    assert {c.type_id for c in out.reported} == {"V01", "V03"}


def test_correlated_sources_max_plus_bounded_bonus(catalog):
    cfg = load_config()
    assert cfg["combine"] == "max"
    out = run([ev(t=10.0, conf=0.7), ev(t=10.4, conf=0.6, source="temporal")], catalog)
    c = out.reported[0]
    # max(1.0*0.7, 0.8*0.6) + bonus 0.05 — не noisy-OR (0.844+)
    assert c.confidence == pytest.approx(0.75, abs=1e-6)
    assert c.bonus <= cfg["agreement"]["max_bonus"]


def test_vlm_reject_downgrades(catalog):
    base = run([ev(t=10.0, conf=0.6)], catalog).reported
    assert base and base[0].confidence == pytest.approx(0.6)
    rej = ev(t=10.0, conf=0.1, source="vlm", facts=[], extra={"vlm": {"verdict": "reject", "score": 0.9}})
    out = run([ev(t=10.0, conf=0.6), rej], catalog)
    assert not out.reported
    cand = out.candidates[0]
    assert cand.vlm_state == "reject" and cand.confidence == pytest.approx(0.36)


def test_vlm_opinion_conventions():
    assert vlm_opinion("rules", 0.5, {"vlm_verdict": "yes"}) == ("confirm", 1.0)
    assert vlm_opinion("vlm", 0.7, {"verdict": "confirm"}) == ("confirm", 0.7)
    assert vlm_opinion("rules", 0.5, {"verified": False}) == ("reject", 1.0)
    assert vlm_opinion("rules", 0.5, {"vlm": "нет"}) == ("reject", 1.0)
    assert vlm_opinion("rules", 0.5, {"vlm": {"answer": "unsure", "p": 0.4}}) == ("unsure", 0.4)
    assert vlm_opinion("rules", 0.5, {}) is None


def test_high_risk_curve_alone_is_hint_not_violation(catalog):
    risk = [(t / 5, 0.95 if 20 <= t / 5 <= 23 else 0.05) for t in range(300)]
    out = run([], catalog, risk=risk)
    assert out.verdict == "no_violation" and not out.reported
    hints = [c for c in out.candidates if c.origin == "risk"]
    assert hints and hints[0].gate == ["risk_only"]


def test_temporal_only_event_is_hint(catalog):
    out = run([ev(t=10.0, conf=0.95, source="temporal")], catalog)
    assert not out.reported
    assert set(out.candidates[0].gate) >= {"temporal_only", "no_observable_fact"}


def test_risk_supports_rules_event(catalog):
    risk = [(t / 5, 0.9) for t in range(300)]
    out = run([ev(t=10.0, conf=0.5)], catalog, risk=risk)
    c = out.reported[0]
    assert "risk_curve" in c.weighted and c.confidence > 0.5


def test_no_fact_no_violation(catalog):
    out = run([ev(t=10.0, conf=0.9, facts=[])], catalog)
    assert not out.reported and "no_observable_fact" in out.candidates[0].gate


def test_forbidden_only_fact_does_not_count(catalog):
    out = run([ev(t=10.0, conf=0.9, facts=["кассир украл деньги"])], catalog)
    assert not out.reported and "no_observable_fact" in out.candidates[0].gate


def test_type_without_norm_is_hint(catalog):
    out = run([ev("V06", t=10.0, facts=["на столе рядом с наличными стоит стакан"])], catalog)
    assert not out.reported and out.candidates[0].gate == ["no_norm"]


def test_unknown_type_maps_to_v99(catalog):
    out = run([ev("V42", t=10.0, facts=["клиент перелезает через стойку"])], catalog)
    c = out.candidates[0]
    assert c.type_id == "V99" and c.extra.get("orig_type_ids") == ["V42"]
    # у V99 норма есть (verified="false" строкой → не сверена, но норма применима) → нарушение
    assert c.reported


def test_no_catalog_reports_with_warning(no_frames):
    res = fuse(FakeClip(), [ev(t=5.0)], [], {}, "job")
    assert res.verdict == "violation" and res.violations[0].rules == []
    assert any("norms" in w for w in res.meta["warnings"])


def test_per_type_thresholds(tmp_path, monkeypatch, catalog, no_frames):
    thr = tmp_path / "thr.yaml"
    thr.write_text("default: 0.5\nper_type:\n  V01: 0.9\n", encoding="utf-8")
    monkeypatch.setenv("SENTINEL_FUSION_THRESHOLDS", str(thr))
    res = fuse(FakeClip(), [ev(t=5.0, conf=0.8)], [], catalog, "job")
    assert res.verdict == "no_violation"
    hint = res.meta["ui"]["review_hints"][0]
    assert hint["reasons"] == ["below_threshold"] and hint["threshold"] == 0.9
    thr.write_text("default: 0.5\nper_type:\n  V01: 0.7\n", encoding="utf-8")
    assert fuse(FakeClip(), [ev(t=5.0, conf=0.8)], [], catalog, "job").verdict == "violation"


def test_timecode_bounds_and_rounding():
    assert timecode(12.345, 60.0) == 12.3
    assert timecode(-3.0, 60.0) == 0.0
    assert timecode(75.0, 60.0) == 60.0
    assert timecode(10.06, 10.04) == 10.0


def test_event_beyond_duration_clamped(catalog, no_frames):
    clip = FakeClip(duration=10.0)
    res = fuse(clip, [ev(t=14.0, dur=2.0)], [], catalog, "job")
    v = res.violations[0]
    assert 0.0 <= v.t_start <= v.timecode_sec <= v.t_end <= 10.0


def test_timecode_snapped_to_frame_pts(catalog, no_frames):
    clip = FakeClip(duration=20.0)  # сэмплы по 0.2 с
    res = fuse(clip, [ev(t=6.13)], [], catalog, "job")
    assert res.violations[0].timecode_sec in (6.0, 6.2)


def test_max_10_sorted_by_time(catalog, no_frames):
    events = [ev(t=float(t), conf=0.5 + (t % 7) / 20) for t in range(5, 125, 8)]
    res = fuse(FakeClip(duration=130.0), events, [], catalog, "job")
    assert len(res.violations) == 10
    times = [v.timecode_sec for v in res.violations]
    assert times == sorted(times)
    kept = sorted(v.confidence for v in res.violations)
    dropped = [h for h in res.meta["ui"]["review_hints"] if h["reasons"] == ["over_limit"]]
    assert dropped and max(h["score"] for h in dropped) <= kept[0] + 1e-9


def test_cross_type_nms(catalog):
    out = run([ev("V01", t=10.0, conf=0.9), ev("V03", t=10.2, conf=0.3, facts=["сейф открыт"])], catalog)
    weak = next(c for c in out.candidates if c.type_id == "V03")
    assert weak.suppressed_by and not weak.reported


def test_timeline_step_and_max(catalog, no_frames):
    risk = [(t / 5, 0.3) for t in range(100)]
    res = fuse(FakeClip(duration=20.0), [ev(t=5.0, conf=0.8)], risk, catalog, "job")
    ts = [t for t, _ in res.timeline]
    assert ts[0] == 0.0 and ts[-1] == 20.0 and all(abs(b - a - 0.5) < 1e-9 for a, b in zip(ts, ts[1:]))
    by_t = dict(res.timeline)
    assert by_t[5.0] == pytest.approx(res.violations[0].confidence, abs=1e-3)
    assert by_t[15.0] == pytest.approx(0.3)


def test_scores_raw_roundtrip(catalog, no_frames):
    res = fuse(FakeClip(), [ev(t=5.0), ev(t=9.0, source="temporal")], [(1.0, 0.2)], catalog, "job")
    raw = res.meta["scores_raw"]
    json.dumps(raw)
    assert len(raw["inputs"]["events"]) == 2 and raw["inputs"]["risk"] == [[1.0, 0.2]]
    assert {c["type_id"] for c in raw["candidates"]} == {"V01"}


def test_malformed_events_skipped(catalog, no_frames):
    bad = [{"type_id": "V01"}, {"type_id": "V01", "t_peak": "abc"}, None]
    res = fuse(FakeClip(), bad + [ev(t=5.0)], [], catalog, "job")
    assert res.status == "done" and len(res.violations) == 1
    assert any("skipped" in w for w in res.meta["warnings"])


def test_dict_events_accepted(catalog, no_frames):
    d = {"type_id": "V01", "t_start": 4.0, "t_end": 6.0, "t_peak": 5.0, "confidence": 0.8, "source": "rules",
         "facts": ["посетитель за кассовым столом"]}
    assert fuse(FakeClip(), [d], [], catalog, "job").verdict == "violation"
