"""Фаза 2: полнота (meta.coverage), неизвестно ≠ нет, реестр типов, флаги, воспроизводимость."""
from __future__ import annotations

import json

import pytest

from contracts.examples import build as ex
from contracts.extensions import (
    TYPE_REGISTRY,
    compute_coverage,
    coverage_problem,
    is_mock_result,
    signal_value,
    video_duration,
)
from contracts.serialize import (
    DEFAULT_TYPE_NAMES,
    api_to_result,
    canonical_json,
    fingerprint,
    load_config,
    result_to_jury_api,
    result_to_ui_report,
)
from contracts.validate import job_response_errors, jury_response_errors


def test_compute_coverage_subtracts_unknown_spans():
    cov = compute_coverage(20.0, 20.0, unknown_spans=[[5, 8], [7, 10], [30, 40], [3, 2]],
                           stages={"detect": {"status": "ok"}}, required=["detect"])
    assert cov["unknown_spans"] == [[5.0, 10.0]] and cov["analyzed_sec"] == 15.0 and cov["ratio"] == 0.75
    assert cov["complete"] is True
    assert "75%" in coverage_problem({"coverage": cov}, 0.98)
    short = compute_coverage(20.0, 12.0)
    assert short["ratio"] == 0.6 and short["complete"] is False
    bad = compute_coverage(20.0, 20.0, stages={"detect": "error"}, required=["detect"])
    assert bad["complete"] is False and coverage_problem({"coverage": bad}, 0.9)


def test_coverage_problem_contract():
    assert coverage_problem({"coverage": {"ratio": 1.0}}, 0.98) is None
    assert coverage_problem({}, 0.98) == "полнота анализа не подтверждена: нет meta.coverage"
    assert "не число" in coverage_problem({"coverage": "full"}, 0.98)
    mock_stage = {"coverage": {"ratio": 1.0, "required": ["detect"], "stages": {"detect": "mock"}}}
    assert coverage_problem(mock_stage, 0.98) and coverage_problem(mock_stage, 0.98, allow_mock=True) is None
    assert is_mock_result(mock_stage) and is_mock_result({"mock": True}) and not is_mock_result({"mock": False})


def test_unknown_is_not_false():
    assert signal_value({"phone_in_hand": 0.0}, "phone_in_hand") == 0.0      # наблюдали: нет
    assert signal_value({"cashier_present": 1.0}, "phone_in_hand") is None   # не наблюдали: неизвестно
    assert signal_value(None, "safe_open") is None
    assert signal_value({"safe_open": float("nan")}, "safe_open") is None


def test_type_registry_has_v09_and_serializer_uses_it():
    assert "V09" in TYPE_REGISTRY and DEFAULT_TYPE_NAMES["V09"] == TYPE_REGISTRY["V09"]
    from contracts.sentinel_contracts import AnalysisResult, Violation

    r = AnalysisResult(job_id="j", status="done", verdict="violation",
                       violations=[Violation(type="", timecode_sec=3.0, description="d", evidence_frames=[],
                                             type_id="V09")])
    assert result_to_jury_api(r)["violations"][0]["type"] == TYPE_REGISTRY["V09"]


def test_video_duration_sources():
    assert video_duration({"video": {"duration": 12.5}}) == 12.5          # формат A03
    assert video_duration({"coverage": {"duration_sec": 7.0}}) == 7.0
    assert video_duration({"duration_sec": "nan"}) is None


def test_config_flags_default_off_and_env_override(tmp_path, monkeypatch):
    cfg = load_config()
    assert cfg["jury"]["embed_rule_in_description"] is False   # гипотеза выключена до замера L2
    assert cfg["jury"]["max_frames"] == 3
    alt = tmp_path / "c.yaml"
    alt.write_text("jury:\n  max_frames: 1\n  embed_rule_in_description: true\n", encoding="utf-8")
    monkeypatch.setenv("SENTINEL_CONTRACTS_CONFIG", str(alt))
    assert load_config()["jury"] == {"max_frames": 1, "embed_rule_in_description": True}
    full = ex.build_examples()  # пример собирается и с другим конфигом
    assert len(full["jury_done_full"]["violations"][0]["evidence_frames"]) == 1


@pytest.mark.parametrize("name,obj", sorted(ex.build_examples().items()))
def test_examples_are_reproducible(name, obj):
    on_disk = (ex.HERE / f"{name}.json").read_text(encoding="utf-8")
    assert on_disk == ex.render(obj), f"{name}.json устарел: python -m contracts.examples.build"
    assert job_response_errors(obj) == []


def test_examples_semantics():
    e = ex.build_examples()
    assert e["job_no_violation"]["verdict"] == "no_violation"
    assert e["jury_unverified_no_violation"] == {"job_id": "3f2b6c1e-0000-4000-8000-000000000006",
                                                 "status": "failed", "verdict": None, "violations": []}
    for name in ("jury_done_full", "job_processing", "job_no_violation", "jury_unverified_no_violation",
                 "job_failed"):
        assert jury_response_errors(e[name]) == [], name


def test_serialization_is_reproducible_and_fingerprinted():
    e = ex.build_examples()
    a = canonical_json(e["job_done_full"])
    assert a == canonical_json(json.loads(json.dumps(e["job_done_full"])))
    assert fingerprint(e["jury_done_full"]) == fingerprint(result_to_jury_api(api_to_result(e["jury_done_full"])))
    assert fingerprint(e["jury_done_full"]) != fingerprint(e["job_no_violation"])
    ui = result_to_ui_report(api_to_result(e["job_done_full"]))
    assert canonical_json(ui["violations"]) == canonical_json(e["job_done_full"]["violations"])
