import pytest

from ml.track.tests import synth as S
from ml.track.tests._util import role_of, run


def test_clean_roles():
    res, ids, by_id, _, _ = run(S.scene_clean())
    assert role_of(ids, by_id, "cashier")[0] == ["cashier"]
    roles, confs = role_of(ids, by_id, "visitor")
    assert roles == ["visitor"] and confs[0] >= 0.6


def test_v01_outsider_is_not_cashier_and_flagged_honestly():
    res, ids, by_id, _, _ = run(S.scene_v01())
    assert role_of(ids, by_id, "cashier")[0] == ["cashier"]
    roles, confs = role_of(ids, by_id, "outsider")
    assert roles[0] in ("unknown", "visitor") and roles[0] != "cashier"
    reasons = res.info["roles"][sorted(ids["outsider"])[0]]["reasons"]
    assert "co_present_in_cashier_area" in reasons and "entered_mid_clip" in reasons


def test_v02_leave_and_return_with_appearance_is_same_track():
    res, ids, by_id, _, _ = run(S.scene_v02(), render=True)
    assert len(ids["cashier"]) == 1
    assert role_of(ids, by_id, "cashier")[0] == ["cashier"]
    assert any(e["kind"] == "long_gap_appearance" for e in res.info["stitch"])


def test_v02_return_without_appearance_is_not_silently_cashier():
    res, ids, by_id, _, _ = run(S.scene_v02())
    roles, _ = role_of(ids, by_id, "cashier")
    assert roles[0] == "cashier"
    if len(roles) > 1:  # без внешности: только позиция -> unknown с причиной, без самоподтверждения
        later = sorted(ids["cashier"])[1]
        assert by_id[later].role == "unknown"
        assert "position_only_no_appearance" in res.info["roles"][later]["reasons"]


def test_other_person_takes_cashier_seat_different_clothes():
    res, ids, by_id, _, _ = run(S.scene_v02_outsider(), render=True)
    assert ids["cashier"] != ids["other"]
    roles, _ = role_of(ids, by_id, "other")
    assert roles == ["unknown"]


def test_v07_collector_with_bag_and_entry():
    res, ids, by_id, _, _ = run(S.scene_v07())
    assert role_of(ids, by_id, "cashier")[0] == ["cashier"]
    roles, confs = role_of(ids, by_id, "collector")
    assert roles == ["collector"] and 0.5 <= confs[0] <= 0.75


def test_bag_alone_does_not_confirm_collector():
    # человек с самого начала рядом с сумкой (сумка тоже с начала) — не инкассатор «по сумке»
    sc = S.scene_clean()
    stayer = S.Person("stayer", [(0, 0.62, 0.40), (40, 0.62, 0.40)], [(0, 40)], color=(10, 10, 10))
    sc.persons = [sc.persons[0], stayer]
    sc.objects = sc.objects + [S.Obj("cash_bag", (0.70, 0.40, 0.76, 0.46), [(0.0, 40.0)], 0.8)]
    res, ids, by_id, _, _ = run(sc)
    roles, _ = role_of(ids, by_id, "stayer")
    assert roles == ["unknown"]
    assert "bag_contact_only" in res.info["roles"][sorted(ids["stayer"])[0]]["reasons"]


def test_two_staff_second_is_unknown_not_outsider():
    for seed in range(3):
        res, ids, by_id, _, _ = run(S.scene_two_staff(), seed=seed)
        assert role_of(ids, by_id, "cashier")[0] == ["cashier"]
        assert role_of(ids, by_id, "staff2")[0] == ["unknown"]
        assert by_id[sorted(ids["cashier"])[0]].role_conf < 0.8  # неоднозначность отражена в conf


def test_no_desk_fallback_is_low_confidence():
    res, ids, by_id, zones, _ = run(S.scene_no_desk())
    roles, confs = role_of(ids, by_id, "cashier")
    assert roles == ["cashier"] and confs[0] <= 0.45
    assert "fallback_no_desk_objects" in res.info["roles"][sorted(ids["cashier"])[0]]["reasons"]
    for name in ("cashier_area", "desk", "client_area"):
        if name in zones:
            assert zones[name].conf <= 0.3


@pytest.mark.parametrize("seed", [0, 1])
def test_roles_have_conf_in_range(seed):
    res, _, _, _, _ = run(S.scene_v07(), seed=seed, drop=0.1)
    for t in res.tracks:
        assert 0.0 <= t.role_conf <= 1.0
        if t.label == "person":
            assert t.track_id in res.info["roles"]
