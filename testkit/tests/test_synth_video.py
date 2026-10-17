from __future__ import annotations

import json

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from testkit.fake_clip import FakeClip, clip_from_video, sample_times  # noqa: E402
from testkit.synth_video import main as synth_main  # noqa: E402
from testkit.synth_video import make_video, render_frames  # noqa: E402

SMALL = (116, 212)


def test_make_video_roundtrip(tmp_path):
    gt = make_video(tmp_path / "v.mov", "V01", duration=3, fps=10, size=SMALL)
    assert gt["verdict"] == "violation" and gt["type_id"] == "V01" and gt["size"] == list(SMALL)
    clip = clip_from_video(gt["path"], sample_fps=5.0)
    assert (clip.width, clip.height) == SMALL
    assert abs(clip.duration - 3.0) < 0.15 and abs(clip.fps - 10) < 0.5
    assert len(clip.times) == len(sample_times(clip.duration, 5.0))
    assert clip.frame(0).shape == (SMALL[1], SMALL[0], 3) and clip.frame(0).dtype == np.uint8
    assert clip.frame_at(gt["t"]).shape == clip.frame(0).shape
    assert clip.frame_at(999).shape == clip.frame(0).shape  # за концом — последний кадр


def test_default_signature_matches_task(tmp_path):
    gt = make_video(str(tmp_path / "c.mov"), "clean", 2, 5)
    assert gt["size"] == [464, 848] and gt["n_frames"] == 10 and gt["verdict"] == "no_violation"


@pytest.mark.parametrize("view,size", [("landscape", (SMALL[1], SMALL[0])), ("perspective", SMALL),
                                       ("mirror", SMALL), ("rot180", SMALL)])
def test_views_change_frames(view, size):
    frames, gt = render_frames("V03", duration=2, fps=5, size=SMALL, view=view)
    base, _ = render_frames("V03", duration=2, fps=5, size=SMALL)
    assert gt["size"] == list(size) and frames[0].shape == (size[1], size[0], 3)
    if view != "landscape":
        assert np.abs(frames[3].astype(int) - base[3].astype(int)).mean() > 1.0


def test_safe_and_door_visibly_change():
    frames, gt = render_frames("V03", duration=20, fps=5, size=SMALL, noise_sigma=0)
    closed, opened = frames[0], frames[int(gt["t"] * 5)]
    assert np.abs(closed.astype(int) - opened.astype(int))[: SMALL[1] // 4].mean() > 1.0
    frames, gt = render_frames("V04", duration=20, fps=5, size=SMALL, noise_sigma=0)
    assert np.abs(frames[0].astype(int) - frames[int(gt["t"] * 5)].astype(int)).mean() > 0.3


def test_lazy_frames_and_fake_clip():
    clip = FakeClip.from_scenario("V05", duration=4, fps=10, size=SMALL)
    assert clip.meta["gt"]["type_id"] == "V05"
    assert clip.times[:3] == [0.0, 0.2, 0.4] and clip.times[-1] < 4
    assert clip.frame(len(clip) - 1).shape == (SMALL[1], SMALL[0], 3)
    assert clip.frame_at(-5).shape == clip.frame_at(100).shape
    frames, _ = render_frames("clean", duration=1, fps=10, size=SMALL)
    assert len(frames) == 10 and len(frames[2:5]) == 3 and frames[-1].shape == frames[0].shape
    with pytest.raises(IndexError):
        frames[10]


def test_fake_clip_from_arrays():
    arr = [np.full((8, 6, 3), i, np.uint8) for i in range(30)]
    clip = FakeClip(arr, fps=30, sample_fps=5)
    assert clip.duration == 1.0 and clip.times == [0.0, 0.2, 0.4, 0.6, 0.8]
    assert int(clip.frame(1)[0, 0, 0]) == 6 and int(clip.frame_at(0.5)[0, 0, 0]) == 15
    with pytest.raises(ValueError):
        FakeClip([], fps=30)


def test_cli(tmp_path):
    assert synth_main(["--out", str(tmp_path), "--scenario", "V02", "--duration", "2", "--fps", "5"]) == 0
    gts = json.loads((tmp_path / "gt.json").read_text(encoding="utf-8"))
    assert gts[0]["type_id"] == "V02" and (tmp_path / "V02.mov").exists()


def test_bad_view(tmp_path):
    with pytest.raises(ValueError):
        make_video(tmp_path / "x.mov", "clean", 1, 5, view="fisheye")
