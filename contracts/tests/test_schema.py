from __future__ import annotations

import builtins
import copy
import json
from pathlib import Path

import pytest

from contracts import validate as V

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
JOB_EXAMPLES = ["job_done_tz", "job_done_full", "job_processing", "job_no_violation", "job_failed"]


def _load(name: str) -> dict:
    return json.loads((EXAMPLES / f"{name}.json").read_text(encoding="utf-8"))


def test_schema_is_valid_draft_2020_12():
    jsonschema = pytest.importorskip("jsonschema")
    schema = V.load_schema()
    assert schema["$schema"].endswith("2020-12/schema")
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema["required"] == ["job_id", "status", "verdict", "violations"]
    assert schema["additionalProperties"] is True


@pytest.mark.parametrize("name", JOB_EXAMPLES)
def test_examples_are_valid(name):
    assert V.job_response_errors(_load(name)) == []


def test_analyze_response():
    assert V.analyze_response_errors(_load("analyze_response")) == []
    assert V.analyze_response_errors({}) != []
    assert V.analyze_response_errors({"job_id": ""}) != []


def _broken():
    good = _load("job_done_tz")
    cases = {}
    for key in ("job_id", "status", "verdict", "violations"):
        d = copy.deepcopy(good)
        del d[key]
        cases[f"missing_{key}"] = d
    for key in ("type", "timecode_sec", "description", "evidence_frames"):
        d = copy.deepcopy(good)
        del d["violations"][0][key]
        cases[f"missing_violation_{key}"] = d
    d = copy.deepcopy(good)
    del d["violations"][0]["evidence_frames"][0]["image_base64"]
    cases["missing_frame_b64"] = d
    d = copy.deepcopy(good)
    d["status"] = "queued"
    cases["bad_status"] = d
    d = copy.deepcopy(good)
    d["verdict"] = None
    cases["done_without_verdict"] = d
    d = copy.deepcopy(_load("job_processing"))
    d["verdict"] = "violation"
    cases["processing_with_verdict"] = d
    d = copy.deepcopy(good)
    d["violations"][0]["timecode_sec"] = -1
    cases["negative_timecode"] = d
    d = copy.deepcopy(good)
    d["violations"][0]["timecode_sec"] = "22"
    cases["string_timecode"] = d
    return cases


@pytest.mark.parametrize("case", sorted(_broken()))
def test_broken_responses_rejected(case):
    assert V.job_response_errors(_broken()[case]) != []
    with pytest.raises(ValueError):
        V.validate_job_response(_broken()[case])


def test_extra_fields_allowed():
    d = _load("job_done_tz")
    d["whatever"] = {"a": 1}
    d["violations"][0]["extra_field"] = [1, 2]
    assert V.is_valid_job_response(d)


@pytest.mark.parametrize("case", sorted(_broken()))
def test_fallback_validator_without_jsonschema(case, monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "jsonschema" or name.startswith("jsonschema."):
            raise ImportError("no jsonschema")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    if case in ("bad_frame_pattern",):
        pytest.skip("pattern-проверка только в jsonschema")
    assert V.job_response_errors(_broken()[case]) != []
    for name in JOB_EXAMPLES:
        assert V.job_response_errors(_load(name)) == [], name


def test_consistency_warnings():
    assert V.check_consistency(_load("job_done_full")) == []
    d = _load("job_done_tz")
    warns = V.check_consistency(d)
    assert any("type_id" in w for w in warns) and any("rules" in w for w in warns)
    d["verdict"] = "no_violation"
    assert any("ложным" in w for w in V.check_consistency(d))
    d = _load("job_done_full")
    d["violations"][0]["evidence_frames"][0]["image_base64"] = "data:image/jpeg;base64,AAAA"
    assert any("data URI" in w for w in V.check_consistency(d))
    d["violations"][0]["evidence_frames"][0]["image_base64"] = "iVBORw0KGgo="  # PNG
    assert any("не JPEG" in w for w in V.check_consistency(d))
    assert any("длительности" in w for w in V.check_consistency(_load("job_done_full"), duration=10.0))
    assert V.check_consistency({"status": "done", "verdict": "violation", "violations": []})


def test_cli(tmp_path, capsys):
    assert V.main([str(EXAMPLES / "job_done_full.json")]) == 0
    assert V.main(["--analyze", str(EXAMPLES / "analyze_response.json")]) == 0
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"job_id": "x"}), encoding="utf-8")
    assert V.main([str(bad)]) == 1
    assert V.main(["--strict", str(EXAMPLES / "job_done_tz.json")]) == 1
    assert "FAIL" in capsys.readouterr().out
