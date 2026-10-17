from __future__ import annotations

import pytest

from testkit.scenarios import SCENARIOS, VIEWS, SceneScript, map_box, map_point, view_transform


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("duration", [6.0, 20.0, 145.0])
def test_ground_truth_shape(scenario, duration):
    s = SceneScript(scenario, duration)
    gt = s.gt
    assert gt.verdict in ("violation", "no_violation")
    if scenario.startswith("V"):
        assert gt.type_id == scenario and gt.verdict == "violation"
        assert 0 <= gt.t_start <= gt.t <= gt.t_end <= duration
    else:
        assert gt.type_id is None and gt.t is None
    for t in (0.0, duration / 3, duration - 0.01):
        snap = s.at(t)
        assert 0 <= snap.safe_open <= 1 and 0 <= snap.door_open <= 1
        for p in snap.persons:
            assert p.role in ("cashier", "visitor", "collector")


def test_unknown_scenario():
    with pytest.raises(ValueError):
        SceneScript("V42")


@pytest.mark.parametrize("view", VIEWS)
def test_views_keep_points_in_frame(view):
    for pt in [(0.0, 0.0), (1.0, 1.0), (0.3, 0.7), (0.5, 0.5)]:
        x, y = map_point(pt, view)
        assert -1e-6 <= x <= 1 + 1e-6 and -1e-6 <= y <= 1 + 1e-6
    b = map_box((0.1, 0.2, 0.3, 0.4), view)
    assert b[0] <= b[2] and b[1] <= b[3]


def test_view_geometry():
    assert map_point((0.25, 0.5), "default") == (0.25, 0.5)
    x, y = map_point((0.0, 0.5), "mirror")
    assert abs(x - 1.0) < 1e-9 and abs(y - 0.5) < 1e-9
    _, size = view_transform("landscape", (464, 848))
    assert size == (848, 464)
    x, y = map_point((0.0, 0.0), "landscape")  # левый верх -> правый верх при повороте по часовой
    assert abs(x - 1.0) < 1e-9 and abs(y) < 1e-9
    assert map_point((0.5, 0.5), "perspective") != (0.5, 0.5)
