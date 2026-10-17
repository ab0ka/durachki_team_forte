"""Тест-кит v2: механизмы спецификации (не качество модели)."""
from __future__ import annotations

import statistics

import pytest

from contracts.extensions import TYPE_REGISTRY
from testkit import contract_checks as cc
from testkit.spec import (
    CASE_ORDER,
    CASES,
    CLEAN_CASES,
    LAYOUTS,
    CaseSpec,
    build_case,
    case_bundle,
    make_suite,
    sample_timing,
)

KEY = {"V01": ("outsider_in_cashier_area", "second_outsider"), "V02": ("workplace_unattended_with_cash",),
       "V03": ("safe_open",), "V05": ("phone_in_hand", "phone_to_ear"), "V06": ("eating_drinking",),
       "V07": ("cash_bag_visible",), "V09": ("personal_items_on_desk",)}
SUITE = make_suite(seed=7, orientations=("portrait", "landscape"))


def test_catalogue_pairs_and_registry():
    assert len(CASE_ORDER) == len(CASES) == 21
    assert set(CLEAN_CASES) == {"leave_clean", "client_opposite", "cashier_enters_first", "client_first",
                                "lights_off_clean", "hand_to_face", "desk_phone", "bag_handover_proper"}
    for spec in make_suite(seed=1, layouts=("A",), orientations=("portrait",)):
        for e in build_case(spec).gt.as_dict()["events"]:
            assert e["type_id"] in TYPE_REGISTRY and all(a in TYPE_REGISTRY for a in e["alt_type_ids"])
    assert "V09" in {e["type_id"] for s in SUITE for e in build_case(s).gt.as_dict()["events"]}


@pytest.mark.parametrize("spec", SUITE, ids=lambda s: f"{s.case}-{s.layout}-{s.orientation}")
def test_bundle_matches_contract(spec):
    b = case_bundle(spec)
    assert cc.check_detections(b.detections) == []
    assert cc.check_tracks_zones((b.tracks, b.zones)) == []
    assert cc.check_pose_frames(b.poses) == []
    assert cc.check_scene_states(b.states) == []
    assert cc.check_events(b.events, duration=spec.duration) == []
    assert (b.gt["verdict"] == "violation") == bool(b.gt["events"])


@pytest.mark.parametrize("spec", [s for s in SUITE if s.case not in CLEAN_CASES and s.orientation == "portrait"],
                         ids=lambda s: f"{s.case}-{s.layout}")
def test_key_signal_follows_action_onset(spec):
    b = case_bundle(spec)
    for e in b.gt["events"]:
        window = [s for s in b.states if e["t"] + 0.2 <= s.t <= e["t"] + 2.8]
        assert any(any((s.signals.get(k) or 0) >= 0.99 for k in KEY[e["type_id"]]) for s in window), e
        before = [s for s in b.states if s.t <= e["t"] - 0.6]
        if e["type_id"] in ("V02", "V05", "V09"):  # переход: до начала действия признака нет
            assert not any((s.signals.get(k) or 0) >= 0.99 for s in before for k in KEY[e["type_id"]]), e


@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_hard_negatives_have_no_violation_signal(layout):
    def states(case):
        return case_bundle(CaseSpec(case, layout, duration=30.0, onset=12.0)).states

    assert all(s.signals.get("phone_in_hand") != 1.0 for s in states("desk_phone"))       # стационарный — не мобильный
    hand = states("hand_to_face")
    assert all((s.signals.get("eating_drinking") or 0) < 0.5 and s.signals.get("phone_to_ear") != 1.0 for s in hand)
    for case in ("client_opposite", "client_first", "cashier_enters_first"):
        assert all(s.signals.get("outsider_in_cashier_area") != 1.0 for s in states(case)), case
    for case in ("leave_clean", "lights_off_clean"):
        for s in states(case):
            assert s.signals.get("workplace_unattended_with_cash") != 1.0, (case, s.t)
            if s.signals.get("cashier_present") == 0.0:
                assert (s.signals.get("safe_open") or 0) < 0.5


def test_unknown_is_absent_not_zero():
    st = case_bundle(CaseSpec("lights_off_cash", "A", duration=30.0, onset=12.0)).states
    dark = [s for s in st if s.t > 16.5]
    assert dark and all(set(s.signals) == {"scene_visibility"} for s in dark)
    lit_absent = [s for s in st if 15.0 <= s.t <= 15.6]
    assert lit_absent and all("phone_in_hand" not in s.signals and s.signals["cashier_present"] == 0.0
                              for s in lit_absent)
    assert any(s.signals.get("workplace_unattended_with_cash") == 1.0 for s in lit_absent)  # видно до темноты
    gt = build_case(CaseSpec("lights_off_cash", "A", duration=30.0, onset=12.0)).gt.as_dict()
    assert gt["unknown_spans"] == [[15.5, 30.0]] and gt["events"][0]["t"] == 12.0


def test_same_timing_distribution_for_clean_and_violation():
    big = make_suite(seed=3, n_per_case=6, orientations=("portrait",))
    clean = [s for s in big if s.case in CLEAN_CASES]
    viol = [s for s in big if s.case not in CLEAN_CASES]
    for attr in ("duration", "onset"):
        mc, mv = statistics.mean(getattr(s, attr) for s in clean), statistics.mean(getattr(s, attr) for s in viol)
        assert abs(mc - mv) / mv < 0.12, (attr, mc, mv)
    fc = statistics.mean(s.onset / s.duration for s in clean)
    fv = statistics.mean(s.onset / s.duration for s in viol)
    assert abs(fc - fv) < 0.06
    import inspect

    assert list(inspect.signature(sample_timing).parameters) == ["rng", "duration_range"]  # класс не передаётся


def test_multi_event_and_debounce():
    multi = build_case(CaseSpec("multi_event", "B", duration=40.0, onset=12.0)).gt.as_dict()
    assert [e["type_id"] for e in multi["events"]] == ["V05", "V02"] and multi["events"][1]["t"] > multi["events"][0]["t"]
    deb = build_case(CaseSpec("debounce_phone", "A", duration=40.0, onset=12.0)).gt.as_dict()
    assert len(deb["events"]) == 1 and deb["events"][0]["t_end"] - deb["events"][0]["t"] == 6.0
    leave_bag = build_case(CaseSpec("leave_bag", "A", duration=30.0, onset=12.0)).gt.as_dict()
    assert leave_bag["events"][0]["alt_type_ids"] == ["V07"]


def test_layouts_differ_and_landscape_swaps():
    a = case_bundle(CaseSpec("client_opposite", "A", duration=20.0, onset=9.0))
    b = case_bundle(CaseSpec("client_opposite", "B", duration=20.0, onset=9.0))
    za = {z.name: z.polygon for z in a.zones}
    zb = {z.name: z.polygon for z in b.zones}
    assert za["cashier_area"] != zb["cashier_area"] and za["desk"] != zb["desk"]
    land = case_bundle(CaseSpec("client_opposite", "A", "landscape", duration=20.0, onset=9.0))
    assert {z.name: z.polygon for z in land.zones}["desk"] != za["desk"]


def test_role_by_zone_not_order():
    spec = CaseSpec("client_first", "A", duration=30.0, onset=10.0)
    b = case_bundle(spec)
    first_seen = min(b.detections, key=lambda d: d.t if d.label == "person" else 1e9)
    roles = {t.track_id: t.role for t in b.tracks if t.label == "person"}
    assert roles[first_seen.track_id] == "visitor" and "cashier" in roles.values()


def test_case_video(tmp_path):
    pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    from testkit.fake_clip import clip_from_video
    from testkit.spec import make_case_video

    spec = CaseSpec("lights_off_cash", "B", "landscape", duration=14.0, onset=7.0)  # свет гаснет на 10.5 с
    gt = make_case_video(tmp_path / "c.mov", spec, fps=5, size=(116, 212))
    clip = clip_from_video(gt["path"], sample_fps=5)
    assert (clip.width, clip.height) == (212, 116) and gt["events"][0]["type_id"] == "V02"
    assert np.asarray(clip.frame(len(clip) - 1)).mean() < np.asarray(clip.frame(0)).mean() * 0.5  # свет выключен
