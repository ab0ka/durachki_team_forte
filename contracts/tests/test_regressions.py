"""Регрессии аудита интеграции (2055fc4): failed→no_violation, неверный PTS доказательства, нет meta.coverage.

Внешняя схема ответа жюри не меняется — проверяется только, что сериализатор и строгий валидатор не пропускают
эти случаи. Тесты не зависят от окружения: mock разрешается/запрещается явным ``allow_mock``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from contracts.extensions import compute_coverage
from contracts.sentinel_contracts import AnalysisResult, EvidenceFrame, Violation
from contracts.serialize import API_REQUIRED_FRAME, API_REQUIRED_TOP, API_REQUIRED_VIOLATION, result_to_jury_api
from contracts.serialize import result_to_ui_report
from contracts.validate import jury_response_errors

JPEG = json.loads((Path(__file__).resolve().parents[1] / "examples" / "job_done_tz.json").read_text(
    encoding="utf-8"))["violations"][0]["evidence_frames"][0]["image_base64"]
# Формат meta оркестратора A03: stages[имя].status, required_stages, complete, video.duration
A03_STAGES = {s: {"status": "ok"} for s in ("ingest", "detect", "track", "scene", "rules", "fusion")}
A03_REQUIRED = list(A03_STAGES)


def _cov(**kw) -> dict:
    return compute_coverage(20.0, 20.0, stages=A03_STAGES, required=A03_REQUIRED, **kw)


def _result(status="done", verdict="no_violation", violations=None, **meta) -> AnalysisResult:
    return AnalysisResult(job_id="r", status=status, verdict=verdict, violations=violations or [], meta=meta)


def _viol(ts=(10.0,), tc=10.0, t_start=9.5, t_end=12.0) -> Violation:
    return Violation(type="Кассир покинул рабочее место", timecode_sec=tc, description="наблюдаемые факты",
                     evidence_frames=[EvidenceFrame(t, JPEG) for t in ts], type_id="V02", confidence=0.9,
                     t_start=t_start, t_end=t_end)


# ----------------------------------------------------------------------------- failed -> no_violation

@pytest.mark.parametrize("case,result", [
    ("failed+no_violation", _result(status="failed", coverage=_cov(), complete=True)),
    ("processing+no_violation", _result(status="processing", coverage=_cov())),
    ("partial (неизвестный статус)", _result(status="partial", coverage=_cov())),
    ("meta.complete=false", _result(coverage=_cov(), complete=False)),
    ("нет meta.coverage", _result(stages=A03_STAGES, required_stages=A03_REQUIRED, complete=True)),
    ("coverage.ratio 0.5", _result(coverage=compute_coverage(20.0, 10.0, stages=A03_STAGES, required=A03_REQUIRED))),
    ("обязательная стадия error", _result(coverage=compute_coverage(
        20.0, 20.0, stages={**A03_STAGES, "detect": {"status": "error"}}, required=A03_REQUIRED))),
    ("тёмный участок (unknown span)", _result(coverage=_cov(unknown_spans=[[5.0, 15.0]]))),
], ids=lambda x: x if isinstance(x, str) else "")
def test_failed_or_incomplete_never_becomes_no_violation(case, result):
    for allow in (False, True):  # от режима mock не зависит
        body = result_to_jury_api(result, allow_mock=allow)
        assert body["verdict"] != "no_violation", case
        assert body["status"] in ("failed", "processing") and body["verdict"] is None, (case, body)
        assert jury_response_errors(body) == [], case


def test_confirmed_coverage_gives_no_violation_in_a03_format():
    r = _result(coverage=_cov(), stages=A03_STAGES, required_stages=A03_REQUIRED, complete=True,
                video={"duration": 20.0})
    assert result_to_jury_api(r, allow_mock=False) == {"job_id": "r", "status": "done", "verdict": "no_violation",
                                                       "violations": []}


@pytest.mark.parametrize("body", [
    {"job_id": "r", "status": "failed", "verdict": "no_violation", "violations": []},
    {"job_id": "r", "status": "processing", "verdict": "no_violation", "violations": []},
    {"job_id": "r", "status": "done", "verdict": None, "violations": []},
])
def test_strict_validator_rejects_fake_binary_verdict(body):
    assert jury_response_errors(body) != []


def test_missing_coverage_reason_is_reported():
    ui = result_to_ui_report(_result(), allow_mock=False)
    assert ui["status"] == "failed" and "meta.coverage" in ui["error"] and ui["analysis_complete"] is False


# ----------------------------------------------------------------------------- неверный PTS доказательства

def test_evidence_after_video_end_is_dropped_and_rejected():
    r = _result(verdict="violation", violations=[_viol(ts=(10.0, 25.0))], duration_sec=20.0)
    body = result_to_jury_api(r, allow_mock=False)
    assert [f["timestamp_sec"] for f in body["violations"][0]["evidence_frames"]] == [10.0]
    assert any("неверный PTS" in w for w in result_to_ui_report(r, allow_mock=False)["warnings"])
    bad = json.loads(json.dumps(body))
    bad["violations"][0]["evidence_frames"].append({"timestamp_sec": 25.0, "image_base64": JPEG})
    assert any("неверный PTS" in e for e in jury_response_errors(bad, duration=20.0))
    assert jury_response_errors(body, duration=20.0) == []


@pytest.mark.parametrize("ts", [float("nan"), float("inf"), -1.0, None])
def test_non_finite_or_negative_pts_never_reaches_jury(ts):
    r = _result(verdict="violation", violations=[_viol(ts=(10.0,))], duration_sec=20.0)
    r.violations[0].evidence_frames.insert(0, EvidenceFrame(ts, JPEG))  # type: ignore[arg-type]
    body = result_to_jury_api(r, allow_mock=False)
    assert [f["timestamp_sec"] for f in body["violations"][0]["evidence_frames"]] == [10.0]
    assert jury_response_errors(body, duration=20.0) == []


def test_evidence_outside_event_window_is_flagged_for_ui_only():
    r = _result(verdict="violation", violations=[_viol(ts=(1.0, 10.0))], duration_sec=20.0)
    body = result_to_jury_api(r, allow_mock=False)
    assert [f["timestamp_sec"] for f in body["violations"][0]["evidence_frames"]] == [1.0, 10.0]  # жюри как было
    assert any("вне окна события" in w for w in result_to_ui_report(r, allow_mock=False)["warnings"])


# ----------------------------------------------------------------------------- violation с валидными доказательствами

def test_violation_with_valid_evidence_is_strict_tz():
    r = _result(verdict="violation", violations=[_viol(ts=(9.6, 10.0, 10.4, 11.0))], duration_sec=20.0)
    body = result_to_jury_api(r, allow_mock=False)
    assert tuple(body) == API_REQUIRED_TOP and body["status"] == "done" and body["verdict"] == "violation"
    v = body["violations"][0]
    assert tuple(v) == API_REQUIRED_VIOLATION and 1 <= len(v["evidence_frames"]) <= 3
    assert all(tuple(f) == API_REQUIRED_FRAME for f in v["evidence_frames"])
    assert jury_response_errors(body, duration=20.0) == []


def test_violation_without_any_valid_evidence_keeps_verdict_but_no_frames():
    r = _result(verdict="violation", violations=[_viol(ts=(30.0,))], duration_sec=20.0)
    body = result_to_jury_api(r, allow_mock=False)
    assert body["verdict"] == "violation" and body["violations"][0]["evidence_frames"] == []
    assert any("нет валидных кадров" in w for w in result_to_ui_report(r, allow_mock=False)["warnings"])
