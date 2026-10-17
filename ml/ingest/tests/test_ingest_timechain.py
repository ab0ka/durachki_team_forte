"""Цепочка времени: PTS -> сек -> сэмпл -> окно -> frame_at -> фактический PTS кадра-доказательства."""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from ml.ingest import FrameGapError, load_video
from ml.ingest.synthetic import read_index_code
from ml.ingest.video import find_gaps

from .conftest import FPS, ffmpeg, needs_ffmpeg

BACKENDS = ["pyav", "opencv"]


def _pyav_pts_seconds(path: str) -> list[float]:
    import av

    with av.open(path) as c:
        s = c.streams.video[0]
        pts = sorted(float(f.pts * s.time_base) for f in c.decode(s))
    return [p - pts[0] for p in pts]


def test_times_are_unrounded_pts(cfr_video: str) -> None:
    clip = load_video(cfr_video)
    exact = _pyav_pts_seconds(cfr_video)
    assert clip.source_times == exact  # бит-в-бит те же float, без округления
    assert set(clip.times) <= set(exact)
    assert all(q["t"] == t for q, t in zip(clip.quality, clip.times))


@pytest.mark.parametrize("backend", BACKENDS)
def test_full_chain_sample_window_exact(cfr_video: str, backend: str) -> None:
    clip = load_video(cfr_video, backend=backend)
    # кандидат: окно вокруг сэмпла 7
    i = 7
    t_sample = clip.times[i]
    win = clip.window_indices(t_sample - 0.25, t_sample + 0.25)
    assert i in win and all(abs(clip.times[k] - t_sample) <= 0.25 + 1e-9 for k in win)
    # точный кадр для «произвольного» момента внутри окна -> фактический PTS = PTS реального кадра
    t_req = t_sample + 0.013
    ref = clip.read_exact(t_req)
    assert ref.ok and ref.requested_sec == t_req
    assert ref.timestamp_sec == clip.snap_time(t_req)
    assert read_index_code(ref.image) == round(ref.timestamp_sec * FPS)
    # кадр-доказательство по фактическому PTS — тот же кадр (идемпотентность)
    again = clip.read_exact(ref.timestamp_sec)
    assert again.timestamp_sec == ref.timestamp_sec and np.array_equal(again.image, ref.image)


def test_distinct_frames_dedupe_same_frame(cfr_video: str) -> None:
    clip = load_video(cfr_video)
    refs = clip.frames_at([1.000, 1.004, 1.010, 1.5])  # три запроса попадают в один кадр 30
    assert len({r.timestamp_sec for r in refs}) == 2
    distinct = clip.distinct_frames_at([1.000, 1.004, 1.010, 1.5])
    assert [round(r.timestamp_sec * FPS) for r in distinct] == [30, 45]
    spaced = clip.distinct_frames_at([1.0, 1.1, 1.2, 1.5], min_separation_s=0.15)
    assert [round(r.timestamp_sec * FPS) for r in spaced] == [30, 36, 45]


@pytest.mark.parametrize("backend", BACKENDS)
def test_begin_and_end_of_video(cfr_video: str, backend: str) -> None:
    clip = load_video(cfr_video, backend=backend)
    first = clip.read_exact(0.0)
    assert first.ok and first.timestamp_sec == 0.0 and read_index_code(first.image) == 0
    last_pts = clip.source_times[-1]
    last = clip.read_exact(last_pts + 0.01)  # ещё внутри показа последнего кадра
    assert last.ok and last.timestamp_sec == last_pts and read_index_code(last.image) == 89
    after = clip.read_exact(clip.duration + 2.0)
    assert after.out_of_range and not after.ok and after.timestamp_sec == last_pts
    before = clip.read_exact(-1.0)
    assert before.out_of_range and before.timestamp_sec == 0.0
    with pytest.raises(FrameGapError) as e:
        clip.read_exact(clip.duration + 2.0, strict=True)
    assert e.value.code == "frame_gap"
    assert clip.times[0] == 0.0 and clip.times[-1] <= last_pts


@needs_ffmpeg
@pytest.mark.parametrize("backend", BACKENDS)
def test_start_offset_is_removed(tmp_path: Path, cfr_video: str, backend: str) -> None:
    out = tmp_path / "offset.mov"
    ffmpeg("-i", cfr_video, "-c", "copy", "-output_ts_offset", "2.5", str(out))
    clip = load_video(str(out), backend=backend)
    assert clip.meta["t0_pts_sec"] == pytest.approx(2.5, abs=0.05) or backend == "opencv"
    assert clip.times[0] == 0.0
    codes = [read_index_code(clip.frame(i)) for i in range(len(clip))]
    assert np.allclose(clip.times, np.array(codes) / FPS, atol=1e-6)
    img, ts = clip.frame_at_ts(1.0)
    assert read_index_code(img) == 30 and ts == pytest.approx(1.0, abs=1e-6)


@pytest.fixture(scope="module")
def holed_video(media_dir: Path, cfr_video: str) -> str:
    """VFR с дырой ~1 с (кадры 30..60 выкинуты), остальные PTS исходные."""
    out = media_dir / "holed.mov"
    if not out.exists():
        ffmpeg("-i", cfr_video, "-vf", r"select='not(between(n\,30\,60))'", "-fps_mode", "vfr",
               "-c:v", "mpeg4", "-q:v", "2", str(out))
    return str(out)


@needs_ffmpeg
@pytest.mark.parametrize("backend", BACKENDS)
def test_vfr_hole_is_reported_not_hidden(holed_video: str, backend: str) -> None:
    clip = load_video(holed_video, backend=backend)
    gaps = clip.coverage["gaps"]
    assert len(gaps) == 1
    a, b = gaps[0]
    assert a == pytest.approx(29 / FPS, abs=1e-3) and b == pytest.approx(61 / FPS, abs=1e-3)
    inside = clip.read_exact(1.5)
    assert inside.in_gap and not inside.ok
    assert read_index_code(inside.image) in (29, 61)  # ближайший реальный кадр, а не выдуманный
    with pytest.raises(FrameGapError):
        clip.read_exact(1.5, strict=True)
    near = clip.read_exact(2.04)
    assert near.ok and read_index_code(near.image) == 61
    assert any("пробелы покрытия" in w for w in clip.warnings)
    # сэмплы внутри дыры не выдумываются: между a и b нет ни одного сэмпла
    assert not [t for t in clip.times if a + 1e-6 < t < b - 1e-6]


def test_find_gaps_unit() -> None:
    st = [0.0, 0.1, 0.2, 1.2, 1.3]
    assert find_gaps(st, declared=1.4, dt=0.1) == [(0.2, 1.2)]
    assert find_gaps([0.5, 0.6], declared=0.7, dt=0.1) == [(0.0, 0.5)]
    assert find_gaps([0.0, 0.1], declared=3.0, dt=0.1) == [(0.2, 3.0)]
    assert find_gaps([], declared=2.0, dt=0.1) == [(0.0, 2.0)]


def test_pts_float_from_fraction_is_exact() -> None:
    tb = Fraction(1, 15360)
    assert float(512 * tb) == 512 / 15360  # time_base — Fraction, перевод без накопления ошибки
