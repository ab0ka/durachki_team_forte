from __future__ import annotations

import json

import numpy as np
import pytest

from ml.temporal._video import SimpleClip
from ml.temporal.config import load_config
from ml.temporal.features import MockBackend, extract_clip

from .conftest import write_video

cv2 = pytest.importorskip("cv2")


def test_score_without_checkpoint_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTINEL_MOCK", "1")
    import ml.temporal as temporal

    clip = SimpleClip(str(write_video(tmp_path / "v.avi")), sample_fps=5.0)
    risk, events = temporal.score(clip)
    assert risk == [] and events == []                     # без обученной модели — никакого «случайного» score
    assert temporal.status()["status"] == "no_checkpoint"


def test_extract_clip_mock_shapes_and_focus(tmp_path):
    cfg = load_config()
    clip = SimpleClip(str(write_video(tmp_path / "v.avi", seconds=6.0)), sample_fps=4.0)
    fs = extract_clip(clip, MockBackend(32), cfg, fps=4.0, views=["full", "focus"])
    T = len(fs)
    assert T >= 20 and fs.emb.shape == (T, 2, 32) and fs.dyn.shape == (T, 4) and fs.boxes.shape == (T, 4)
    assert np.allclose(np.linalg.norm(fs.emb, axis=-1), 1.0, atol=1e-4)
    assert np.all(fs.boxes[:, 0] < fs.boxes[:, 2]) and np.all(fs.boxes[:, 1] < fs.boxes[:, 3])
    assert np.all((fs.boxes >= 0) & (fs.boxes <= 1))
    assert fs.meta["focus_mode"] == "motion"
    ev = (fs.times >= 3.2) & (fs.times <= 4.8)
    assert fs.dyn[ev, 0].mean() > fs.dyn[fs.times < 2.5, 0].mean()   # движение видно в окне события
    fs.save(tmp_path / "f.npz")
    from ml.temporal.features import FeatureSet

    fs2 = FeatureSet.load(tmp_path / "f.npz")
    assert fs2.views == fs.views and np.allclose(fs2.emb, fs.emb, atol=1e-3)


def test_extract_with_zone_and_deadline(tmp_path):
    from ml.temporal._contract_fallback import Zone

    cfg = load_config()
    clip = SimpleClip(str(write_video(tmp_path / "v.avi", seconds=6.0)), sample_fps=4.0)
    zone = [Zone("cashier_area", [(0.1, 0.5), (0.9, 0.5), (0.9, 0.95), (0.1, 0.95)])]
    fs = extract_clip(clip, MockBackend(16), cfg, zones=zone, fps=4.0, views=["full", "focus"])
    assert fs.meta["focus_mode"] == "zone"
    fs_cut = extract_clip(clip, MockBackend(16), cfg, fps=4.0, views=["full"], deadline=0.0)
    assert "truncated_at" in fs_cut.meta and len(fs_cut) >= 1


def test_end_to_end_extract_cv_train_score(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTINEL_MOCK", "1")
    vids = tmp_path / "raw"
    vids.mkdir()
    rows = ["file;label;second"]
    for i in range(1, 9):
        pos = i <= 4
        name = f"Х-{i}.avi" if i % 2 else f"X-{i}.avi"          # кириллица и латиница вперемешку
        write_video(vids / name, seconds=8.0, event=(3.0, 5.0) if pos else None, shade=70 + 7 * i)
        rows.append(f"{name};{'нарушение' if pos else 'без нарушения'};{4 if pos else ''}")
    (vids / "X-9.avi").write_bytes((vids / "X-6.avi").read_bytes())   # точный дубликат по md5
    rows.append("Х-9.avi;нарушение;5")
    (tmp_path / "labels.csv").write_text("\n".join(rows), encoding="utf-8")

    from ml.temporal import cv as cvmod
    from ml.temporal import extract, train

    assert extract.main(["--videos", str(vids), "--labels", str(tmp_path / "labels.csv"), "--n-aug", "1"]) == 0
    cfg = load_config()
    from ml.temporal.config import cache_dir

    idx = json.loads((cache_dir(cfg) / "index.json").read_text(encoding="utf-8"))
    assert idx["groups"]["X-6"] == ["X-6", "X-9"] and idx["backend"] == "mock"
    out = tmp_path / "cv.json"
    assert cvmod.main(["--kind", "baseline", "--outer", "4", "--inner", "2", "--out", str(out)]) == 0
    rep = json.loads(out.read_text(encoding="utf-8"))
    names = [v["name"] for v in rep["baseline"]["per_video"]]
    assert "X-9" not in names and "X-6" in names                       # дубликат свёрнут в группу
    x6 = next(v for v in rep["baseline"]["per_video"] if v["name"] == "X-6")
    # исходные метки обоих файлов сохранены, расхождение помечено (политика positive)
    assert x6["conflict"] and x6["label"] == 1
    assert {(s["name"], s["label"]) for s in x6["sources"]} == {("X-6", 0), ("X-9", 1)}
    assert train.main(["--kind", "baseline", "--cv-report", str(out)]) == 0

    import ml.temporal as temporal

    clip = SimpleClip(str(vids / "Х-1.avi"), sample_fps=5.0)
    risk, events = temporal.score(clip)
    st = temporal.status()
    assert st["status"] == "ok" and st["kind"] == "baseline"
    ts = [t for t, _ in risk]
    assert np.allclose(np.diff(ts), 0.5) and all(0.0 <= r <= 1.0 for _, r in risk)
    for e in events:
        assert e.source == "temporal" and e.t_start <= e.t_peak <= e.t_end
        assert e.type_id == "V99" and e.extra["type_source"] == "none"
        assert e.extra["zero_shot_hint"]["trained"] is False
        assert 0.0 <= e.confidence <= 1.0 and e.facts
