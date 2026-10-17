"""SENTINEL_MOCK=1: детерминированные детекции на синтетике, формат Detection, гейтинг, every_n."""
from __future__ import annotations

import numpy as np
import pytest

from ml.detect import detect, last_stats
from ml.detect._contract_fallback import LABELS, Detection
from ml.detect.boxes import iou_matrix
from ml.detect.synth import ArrayClip, make_synth_video, synth_clip


def _check_format(dets):
    for d in dets:
        assert isinstance(d, Detection)
        assert d.label in LABELS
        assert len(d.bbox) == 4 and all(isinstance(v, float) for v in d.bbox)
        x1, y1, x2, y2 = d.bbox
        assert 0.0 <= x1 <= x2 <= 1.0 and 0.0 <= y1 <= y2 <= 1.0
        assert 0.0 <= d.score <= 1.0
        assert d.track_id is None
        assert isinstance(d.t, float) and d.source


def test_mock_detects_synth_objects_in_memory(mock_mode):
    clip, gts = synth_clip(duration_s=4.0, sample_fps=5.0)
    dets = detect(clip)
    _check_format(dets)
    st = last_stats()
    assert st["status"] == "ok" and st["mock"] is True
    by_t: dict[float, list[Detection]] = {}
    for d in dets:
        by_t.setdefault(d.t, []).append(d)
    assert sorted(by_t) == pytest.approx(clip.times)
    for t, gt in zip(clip.times, gts):
        got = by_t[t]
        for label, box in gt:
            cand = [d for d in got if d.label == label]
            assert cand, (t, label)
            best = max(iou_matrix(np.array([box]), np.array([d.bbox for d in cand]))[0])
            assert best > 0.85, (t, label, best)
    # телефон виден только на 1..3 с
    phone_t = sorted({d.t for d in dets if d.label == "cell_phone"})
    assert phone_t and min(phone_t) >= 1.0 and max(phone_t) < 3.0


def test_mock_is_deterministic(mock_mode):
    clip, _ = synth_clip(duration_s=2.0)
    a = [(d.t, d.label, d.bbox, d.score) for d in detect(clip)]
    b = [(d.t, d.label, d.bbox, d.score) for d in detect(clip)]
    assert a == b


def test_every_n_subsamples(mock_mode):
    clip, _ = synth_clip(duration_s=4.0)
    ts = sorted({d.t for d in detect(clip, every_n=3)})
    assert ts == pytest.approx(clip.times[::3])


def test_mock_on_written_video_file(mock_mode, tmp_path):
    p = tmp_path / "synth.mp4"
    make_synth_video(p, duration_s=2.0, fps=10.0)
    clip = ArrayClip.from_file(p, sample_fps=5.0)
    assert len(clip.times) == 10 and clip.width == 464 and clip.height == 848
    dets = detect(clip)
    _check_format(dets)
    labels = {d.label for d in dets}
    assert {"person", "safe", "laptop", "door", "money_counter"} <= labels  # пережили mp4-сжатие


def test_open_vocab_extra_runs_on_keyframes_only(mock_mode, monkeypatch):
    monkeypatch.setenv("SENTINEL_DETECT_EXTRAS", "open_vocab")
    clip, _ = synth_clip(duration_s=4.0, sample_fps=5.0)
    dets = detect(clip)
    st = last_stats()
    assert st["n_open"] == st["n_keyframes"]
    assert 4 <= st["n_keyframes"] < len(clip.times)          # ~1 fps + всплески движения
    assert st["n_open"] > 0 and dets


def test_extras_tiles_do_not_duplicate(mock_mode):
    clip, gts = synth_clip(duration_s=2.0)
    dets = detect(clip, config={"tiling": {"enabled": True, "on": "all"}})
    st = last_stats()
    assert st["n_tiles"] > 0
    for t in clip.times:
        safes = [d for d in dets if d.t == t and d.label == "safe"]
        assert len(safes) == 1


def test_motion_quality_from_clip_is_used(mock_mode):
    clip, _ = synth_clip(duration_s=4.0)
    clip.quality = {"motion_energy": [0.0] * len(clip.times)}
    detect(clip, config={"open_vocab": {"enabled": True}})
    st = last_stats()
    assert st["n_keyframes"] == 4  # только регулярная сетка 1 fps, всплесков нет


def test_empty_and_broken_frames(mock_mode):
    assert detect(ArrayClip(frames=[], times=[])) == []
    clip, _ = synth_clip(duration_s=1.0)
    clip.frames[2] = None  # битый кадр не должен ронять детекцию
    dets = detect(clip)
    assert clip.times[2] not in {d.t for d in dets}
    assert any("frame 2" in w for w in last_stats()["warnings"])


def test_budget_forces_stride(mock_mode, monkeypatch):
    clip, _ = synth_clip(duration_s=8.0, sample_fps=5.0)
    dets = detect(clip, config={"runtime": {"budget_s": 0.0, "hard_budget_s": 1e-9, "batch_size": 4}})
    st = last_stats()
    assert st["stride_final"] > 1 and st["status"] == "degraded"
    assert len({d.t for d in dets}) < len(clip.times)
