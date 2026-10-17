from __future__ import annotations

import json
from pathlib import Path

from eval.common import load_config
from eval.protocol import env_flags, inputs_fingerprint, same_procedure, sha256_json
from eval.runs import append, load, render_html


def _rec(i: int, fn: int, fp: int) -> dict:
    return {"run_id": f"r{i}", "label": f"iter {i}", "role": "single", "protocol": {"git_sha": "abc", "evaluation_kind":
            "exploratory"}, "metrics": {"fn": fn, "fp": fp, "hit_2": 0.5 + i / 10, "latency_p90": 40 + i,
                                        "vram_peak_mb": 4700}}


def test_runs_log_and_html(tmp_path: Path):
    log = tmp_path / "reports" / "runs.jsonl"
    for i, (fn, fp) in enumerate([(5, 3), (3, 4), (2, 2)]):
        append(log, _rec(i, fn, fp))
    runs = load(log)
    assert len(runs) == 3 and runs[-1]["metrics"]["fn"] == 2
    h = render_html(runs)
    assert h.startswith("<!doctype html>") and "FN (пропуски)" in h and "<title>" in h and "разведочные" in h
    assert "<script" not in h


def test_protocol_helpers(tmp_path: Path):
    v = tmp_path / "X-1.mov"
    v.write_bytes(b"abc")
    fp1 = inputs_fingerprint([("X-1", v)])
    v.write_bytes(b"abcd")
    assert inputs_fingerprint([("X-1", v)])["inputs_sha"] != fp1["inputs_sha"]
    assert env_flags({"SENTINEL_API_KEY": "secret", "SENTINEL_X": 1})["SENTINEL_API_KEY"] == "***"
    cfg = load_config()
    a = {"labels_sha256": "l", "mode": "inproc", "subset": "all", "eval_version": "e", "inputs": {"inputs_sha": "i"},
         "eval_config": cfg}
    assert same_procedure(a, json.loads(json.dumps(a))) == []
    b = dict(a, inputs={"inputs_sha": "other"})
    assert any("inputs_sha" in p for p in same_procedure(a, b))
    assert sha256_json({"a": 1, "b": 2}) == sha256_json({"b": 2, "a": 1})
