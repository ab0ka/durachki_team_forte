"""replay CLI: кэш состояний (синтетика) + разметка → таблица и метрики."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.rules.cache_io import load_cache_dir, load_labels, normalize_video_id, parse_label, parse_timecode, save_cache
from ml.rules.replay import main
from ml.rules.synth import RULE_CASES, make_states


def _write_cache(tmp_path: Path) -> Path:
    cache = tmp_path / "cache"
    pos = RULE_CASES["V03_safe_left_open"]["pos"]
    save_cache(cache, "X-1", make_states(60.0, segments=pos["segments"]))
    save_cache(cache, "Х-3", make_states(60.0))  # кириллическая Х
    # плоский формат + jsonl
    flat = [{"t": s.t, **s.signals} for s in make_states(30.0)]
    (cache / "x-9.states.jsonl").write_text("\n".join(json.dumps(r) for r in flat), encoding="utf-8")
    return cache


def test_normalize_and_parsers() -> None:
    assert normalize_video_id("Х-3.MOV") == normalize_video_id("X_3") == "x3"
    assert normalize_video_id("cache/X-47.states.json") == "x47"
    assert parse_timecode("22 секунда") == 22.0
    assert parse_timecode("0:22") == 22.0 and parse_timecode("01:05") == 65.0
    assert parse_timecode("") is None
    assert parse_label("нарушение") is True and parse_label("без нарушения") is False
    assert parse_label("violation") is True and parse_label("no_violation") is False
    assert parse_label("???") is None


def test_load_cache_layouts(tmp_path: Path) -> None:
    cache = _write_cache(tmp_path)
    found = load_cache_dir(cache)
    assert set(found) == {"x1", "x3", "x9"}
    assert found["x9"]["states"][0]["t"] == 0.0


def test_labels_csv_semicolon_cp1251(tmp_path: Path) -> None:
    p = tmp_path / "labels.csv"
    p.write_bytes("Файл;Нарушение;Секунда\nX-1.mov;нарушение;15 секунда\nХ-3.mov;без нарушения;\n".encode("cp1251"))
    lab = load_labels(p)
    assert lab["x1"][0]["violation"] is True and lab["x1"][0]["t"] == 15.0
    assert lab["x3"][0]["violation"] is False and lab["x3"][0]["t"] is None


def test_replay_cli_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cache = _write_cache(tmp_path)
    labels = tmp_path / "labels.csv"
    labels.write_text("file,label,second\nX-1.mov,нарушение,15\nХ-3.mov,без нарушения,\nX-9.mov,без нарушения,\n",
                      encoding="utf-8")
    out_json = tmp_path / "res.json"
    out_csv = tmp_path / "res.csv"
    rc = main(["--cache", str(cache), "--labels", str(labels), "--assume-verified", "--json", str(out_json),
               "--csv", str(out_csv), "--events"])
    assert rc == 0
    text = capsys.readouterr().out
    assert "TP=1 FP=0 FN=0 TN=2" in text
    res = json.loads(out_json.read_text(encoding="utf-8"))
    row = next(v for v in res["videos"] if v["id"] == "x1")
    # синтетика: наличные на столе + кассир ушёл при открытом сейфе → V02 и V03 в один кадр
    assert row["pred"] is True and row["type_id"] in {"V02", "V03"} and abs(row["dt"]) <= 0.25 and row["hit"] is True
    assert {e["type_id"] for e in row["events"]} >= {"V02", "V03"}
    assert out_csv.read_text(encoding="utf-8").startswith("name,gt")


def test_replay_without_verified_norms_predicts_no_violation(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    import ml.rules.catalog as catalog_mod

    monkeypatch.setattr(catalog_mod, "_try_norms_catalog", lambda: None)
    cache = _write_cache(tmp_path)
    labels = tmp_path / "labels.csv"
    labels.write_text("file,label,second\nX-1.mov,нарушение,15\n", encoding="utf-8")
    assert main(["--cache", str(cache), "--labels", str(labels)]) == 0
    text = capsys.readouterr().out
    assert "TP=0 FP=0 FN=1" in text  # без проверенной нормы — только review_observation


def test_replay_sweep_and_overrides(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cache = _write_cache(tmp_path)
    labels = tmp_path / "labels.csv"
    labels.write_text("file,label,second\nX-1.mov,нарушение,15\nХ-3.mov,без нарушения,\n", encoding="utf-8")
    rc = main(["--cache", str(cache), "--labels", str(labels), "--assume-verified",
               "--set", "global.nms_window_s=3",
               "--sweep", "rules.V03_safe_left_open.min_duration_s=5,30"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "min_duration_s" in out and out.count("\n") >= 4


def test_rules_table_cli(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--rules-table"]) == 0
    out = capsys.readouterr().out
    assert "| `V01_outsider_in_cashier_area` | V01 | violation | да |" in out
    assert "| `V99_placeholder` | V99 | observation | нет |" in out


def test_replay_empty_cache(tmp_path: Path) -> None:
    assert main(["--cache", str(tmp_path / "nothing")]) == 2
