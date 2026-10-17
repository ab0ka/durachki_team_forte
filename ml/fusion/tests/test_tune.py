"""Тюнер порогов: разбор разметки, сплит с дублями в одной части, подбор только на train."""
from __future__ import annotations

import json

import yaml

from ml.fusion import fuse
from ml.fusion.tune import (
    Sample,
    evaluate,
    fit,
    main,
    normalize_id,
    parse_label,
    parse_seconds,
    read_labels,
    split_groups,
)

from .conftest import FakeClip, ev


def test_parsers():
    assert normalize_id("Х-3.mov") == "X-3" == normalize_id("x_3") == normalize_id("X–3")
    assert parse_label("нарушение") is True and parse_label("без нарушения") is False
    assert parse_label("1") is True and parse_label("no_violation") is False and parse_label("?") is None
    assert parse_seconds("22 секунда") == 22.0 and parse_seconds("1:05") == 65.0 and parse_seconds("") is None


def test_read_labels_semicolon_cyrillic(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text("Видео;Разметка;Секунда\nХ-4;нарушение;57\nХ-47;нарушение;68\nX-1;без нарушения;\n",
                 encoding="utf-8")
    labels = read_labels(p)
    assert labels["X-4"][0].positive and labels["X-4"][0].second == 57.0
    assert labels["X-1"][0].positive is False


def test_split_keeps_duplicates_together():
    samples = [Sample(f"V{i}", i % 2 == 0, [], [], f"g{i}") for i in range(20)]
    samples += [Sample("X-4", True, [57.0], [], "dup"), Sample("X-47", True, [68.0], [], "dup")]
    train, hold = split_groups(samples, 0.3, seed=1)
    names_t, names_h = {s.video for s in train}, {s.video for s in hold}
    assert not names_t & names_h
    assert ({"X-4", "X-47"} <= names_t) or ({"X-4", "X-47"} <= names_h)
    assert any(s.positive for s in hold) and any(not s.positive for s in hold)


def test_fit_reaches_target_recall_on_train():
    pos = [Sample(f"P{i}", True, [10.0], [("V01", 0.5 + i * 0.03, 10.0)], f"p{i}") for i in range(6)]
    neg = [Sample(f"N{i}", False, [], [("V01", 0.2 + i * 0.03, 5.0)], f"n{i}") for i in range(6)]
    thr = fit(pos + neg, {"V01": 0.9}, 0.5)
    m = evaluate(pos + neg, thr, 0.5)
    assert m["recall"] == 1.0 and m["fp"] == 0
    assert 0.35 < thr["V01"] <= 0.5


def test_cli_end_to_end(tmp_path, catalog, monkeypatch):
    monkeypatch.setenv("SENTINEL_FUSION_NO_FRAMES", "1")
    preds = tmp_path / "preds_raw"
    preds.mkdir()
    rows = ["video,label,second"]
    for i in range(12):
        positive = i % 2 == 0
        events = [ev(t=10.0, conf=0.55 + i * 0.02)] if positive else [ev(t=5.0, conf=0.2 + i * 0.01)]
        res = fuse(FakeClip(duration=30.0), events, [], catalog, f"job{i}")
        name = f"Х-{i}" if i % 3 == 0 else f"X-{i}"
        (preds / f"{name}.json").write_text(json.dumps({"meta": res.meta}, ensure_ascii=False), encoding="utf-8")
        rows.append(f"{name}.mov,{'нарушение' if positive else 'без нарушения'},{'10' if positive else ''}")
    labels = tmp_path / "labels.csv"
    labels.write_text("\n".join(rows) + "\n", encoding="utf-8")
    out = tmp_path / "thr.yaml"
    report = tmp_path / "report.json"
    assert main(["--preds", str(preds), "--labels", str(labels), "--out", str(out), "--report", str(report),
                 "--cv", "3"]) == 0
    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert doc["tuned"] is True and doc["tuned_on"]["mode"] == "train_split"
    assert doc["tuned_on"]["n_holdout"] > 0 and "V01" in doc["per_type"]
    rep = json.loads(report.read_text(encoding="utf-8"))
    assert rep["train"]["recall"] == 1.0 and rep["holdout"]["n"] > 0 and rep["cv"]["k"] == 3
    assert not set(rep["train_videos"]) & set(rep["holdout_videos"])
