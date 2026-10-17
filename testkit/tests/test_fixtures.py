from __future__ import annotations

import pytest

from contracts.serialize import result_to_jury_api, result_to_ui_report
from contracts.validate import jury_response_errors
from testkit import contract_checks as cc
from testkit.fixtures import (
    make_analysis_result,
    make_bundle,
    make_catalog,
    make_detections,
    make_scene_states,
    scenario_names,
)
from testkit.scenarios import VIEWS

KEY_SIGNAL = {"V01": "outsider_in_cashier_area", "V02": "workplace_unattended_with_cash", "V03": "safe_open",
              "V04": "door_open", "V05": "phone_to_ear", "V06": "eating_drinking", "V07": "cash_bag_visible",
              "V08": "hands_hidden"}
VIOLATION_SIGNALS = ("outsider_in_cashier_area", "workplace_unattended_with_cash", "safe_open", "phone_to_ear",
                     "hands_hidden")


@pytest.mark.parametrize("scenario", scenario_names())
def test_bundle_passes_contract_checks(scenario):
    b = make_bundle(scenario, duration=12, sample_fps=5)
    assert cc.check_detections(b.detections) == []
    assert cc.check_tracks_zones((b.tracks, b.zones)) == []
    assert cc.check_pose_frames(b.poses) == []
    assert cc.check_scene_states(b.states, require_all=True) == []
    assert cc.check_events(b.events, duration=12, catalog=b.catalog) == []
    assert {t.role for t in b.tracks if t.label == "person"} <= {"cashier", "visitor", "collector"}
    assert {z.name for z in b.zones} == set(cc.ZONE_NAMES)


@pytest.mark.parametrize("scenario", list(KEY_SIGNAL))
@pytest.mark.parametrize("duration", [8.0, 20.0, 60.0])
def test_key_signal_fires_at_gt_second(scenario, duration):
    b = make_bundle(scenario, duration=duration, sample_fps=5)
    at_t = min(b.states, key=lambda s: abs(s.t - b.gt["t"]))
    assert at_t.signals[KEY_SIGNAL[scenario]] >= 0.99
    assert b.events and b.events[0].type_id == scenario and b.events[0].t_peak == b.gt["t"]


@pytest.mark.parametrize("scenario", ["clean", "clean_collector"])
def test_clean_scenarios_have_no_violation_signals(scenario):
    for s in make_scene_states(scenario, 20, 5):
        for k in VIOLATION_SIGNALS:
            assert s.signals[k] == 0.0, (s.t, k)


@pytest.mark.parametrize("view", VIEWS)
def test_views_and_noise_keep_detections_valid(view):
    dets = make_detections("V07", 10, 5, view=view, noise=0.02, dropout=0.2, seed=3)
    assert dets and cc.check_detections(dets) == []
    b = make_bundle("V01", 10, 5, view=view)
    assert cc.check_tracks_zones((b.tracks, b.zones)) == [] and cc.check_pose_frames(b.poses) == []


def test_noisy_states_stay_in_range():
    assert cc.check_scene_states(make_scene_states("V05", 10, 5, noise=0.3, seed=1), require_all=True) == []


@pytest.mark.parametrize("scenario", scenario_names())
def test_analysis_result_is_jury_valid(scenario):
    r = make_analysis_result(scenario, duration=10, with_images=scenario in ("V01", "clean"))
    body = result_to_jury_api(r)
    assert jury_response_errors(body, duration=10) == []
    assert body["verdict"] == ("violation" if scenario.startswith("V") else "no_violation")
    assert result_to_ui_report(r)["analysis_complete"] is True


def test_stub_catalog_has_no_norms():
    cat = make_catalog()
    assert cat["_testkit_stub"] and set(cat["types"]) >= {f"V0{i}" for i in range(1, 9)} | {"V99"}
    assert all(t["rules"] == [] for t in cat["types"].values())
