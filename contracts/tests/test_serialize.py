from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from contracts.sentinel_contracts import AnalysisResult, EvidenceFrame, RuleRef, Violation
from contracts.serialize import (
    API_REQUIRED_FRAME,
    API_REQUIRED_TOP,
    API_REQUIRED_VIOLATION,
    api_to_result,
    jpeg_b64_problem,
    minimal_api,
    resolve_outcome,
    result_to_api,
    result_to_jury_api,
    result_to_ui_report,
    to_jsonable,
)
from contracts.validate import check_consistency, job_response_errors, jury_response_errors

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
TZ_EXAMPLE = json.loads((EXAMPLES / "job_done_tz.json").read_text(encoding="utf-8"))
JPEG_B64 = TZ_EXAMPLE["violations"][0]["evidence_frames"][0]["image_base64"]
NAN = float("nan")


def _rule(i: int = 0) -> RuleRef:
    # Тестовая заглушка — НЕ норма. Реальные ссылки только из norms/catalog.yaml (A13).
    return RuleRef(document=f"ТЕСТ-ДОКУМЕНТ-{i}", clause=None if i else "п. TEST", url=f"https://example.invalid/{i}",
                   quote="тестовая цитата", verified=False)


def _violation(**kw) -> Violation:
    base = dict(type="Посторонний в кассовом помещении", timecode_sec=22.04,
                description="На кадре виден второй человек за кассовым столом.",
                evidence_frames=[EvidenceFrame(21.96, JPEG_B64), EvidenceFrame(23.449, JPEG_B64)],
                type_id="V01", confidence=0.8766, t_start=21.25, t_end=26.81, rules=[_rule(0), _rule(1)])
    base.update(kw)
    return Violation(**base)


def _result(**kw) -> AnalysisResult:
    base = dict(job_id="job-1", status="done", verdict="violation", violations=[_violation()],
                timeline=[(0.0, 0.1), (0.21, 0.5), (22.04, 0.93)],
                meta={"version": "0.1.0", "timings": {"total": 12.3456}, "progress": 1.0, "stage": "done"})
    base.update(kw)
    return AnalysisResult(**base)


# ----------------------------------------------------------------------------- jury (строгая схема ТЗ)

def test_jury_has_exactly_tz_fields():
    out = result_to_jury_api(_result())
    assert tuple(out) == API_REQUIRED_TOP
    v = out["violations"][0]
    assert tuple(v) == API_REQUIRED_VIOLATION
    assert all(tuple(f) == API_REQUIRED_FRAME for f in v["evidence_frames"])
    assert jury_response_errors(out) == []
    assert job_response_errors(out, strict=True) == []


def test_result_to_api_defaults_to_jury_profile():
    r = _result()
    assert result_to_api(r) == result_to_jury_api(r)
    assert result_to_api(r, profile="ui") == result_to_ui_report(r)
    with pytest.raises(ValueError):
        result_to_api(r, profile="xml")


def test_jury_embeds_first_rule_into_description_once():
    v = result_to_jury_api(_result())["violations"][0]
    assert v["description"].count("https://example.invalid/0") == 1
    assert "ТЕСТ-ДОКУМЕНТ-0, п. TEST" in v["description"]
    already = _result(violations=[_violation(description="Описание. https://example.invalid/0")])
    assert result_to_jury_api(already)["violations"][0]["description"] == "Описание. https://example.invalid/0"
    plain = result_to_jury_api(_result(), embed_rule_in_description=False)
    assert "Норма" not in plain["violations"][0]["description"]


def test_jury_caps_frames_at_three():
    frames = [EvidenceFrame(20.0 + i, JPEG_B64) for i in range(5)]
    out = result_to_jury_api(_result(violations=[_violation(evidence_frames=frames)]))
    assert len(out["violations"][0]["evidence_frames"]) == 3
    assert len(result_to_ui_report(_result(violations=[_violation(evidence_frames=frames)]))
               ["violations"][0]["evidence_frames"]) == 5


def test_rounding_to_tenth():
    out = result_to_ui_report(_result())
    v = out["violations"][0]
    assert v["timecode_sec"] == 22.0
    assert [f["timestamp_sec"] for f in v["evidence_frames"]] == [22.0, 23.4]
    assert (v["t_start"], v["t_end"]) == (21.2, 26.8)
    assert v["confidence"] == 0.877
    assert out["timeline"] == [[0.0, 0.1], [0.2, 0.5], [22.0, 0.93]]


# ----------------------------------------------------------------------------- политика вердикта

def test_processing_has_no_verdict():
    r = _result(status="processing", verdict="violation", meta={"progress": 0.4, "stage": "detect"})
    jury = result_to_jury_api(r)
    assert jury == {"job_id": "job-1", "status": "processing", "verdict": None, "violations": []}
    ui = result_to_ui_report(r)
    assert ui["verdict"] is None and ui["verdict_state"] == "pending"
    assert ui["progress"] == 0.4 and ui["stage"] == "detect"
    assert jury_response_errors(jury) == [] and job_response_errors(ui) == []


def test_failed_has_no_verdict_even_if_set():
    r = _result(status="failed", verdict="no_violation", meta={"error": "decode error"})
    jury = result_to_jury_api(r)
    assert jury["verdict"] is None and jury["violations"] == []
    ui = result_to_ui_report(r)
    assert ui["error"] == "decode error" and ui["verdict_state"] == "unknown"
    assert jury_response_errors(jury) == []


def test_unknown_status_becomes_failed():
    out = result_to_jury_api(AnalysisResult(job_id="j", status="weird"))  # type: ignore[arg-type]
    assert out["status"] == "failed" and out["verdict"] is None


def test_done_without_verdict_and_without_violations_is_not_no_violation():
    out = result_to_jury_api(_result(verdict=None, violations=[]))
    assert out["status"] == "failed" and out["verdict"] is None
    assert result_to_ui_report(_result(verdict=None, violations=[]))["verdict_state"] == "unknown"


def test_done_without_verdict_but_with_violations_is_violation():
    out = result_to_jury_api(_result(verdict=None))
    assert out["verdict"] == "violation" and len(out["violations"]) == 1


def test_violation_verdict_without_valid_violation_fails():
    assert result_to_jury_api(_result(violations=[]))["status"] == "failed"
    nan_v = _violation(timecode_sec=NAN, t_start=NAN, t_end=NAN, evidence_frames=[])
    assert result_to_jury_api(_result(violations=[nan_v]))["status"] == "failed"


def test_no_violation_requires_complete_analysis():
    ok = _result(verdict="no_violation", violations=[], meta={"completeness": {"complete": True}})
    assert result_to_jury_api(ok)["verdict"] == "no_violation"
    legacy = _result(verdict="no_violation", violations=[], meta={})  # модуль без completeness — совместимость
    assert result_to_jury_api(legacy)["verdict"] == "no_violation"
    partial = _result(verdict="no_violation", violations=[],
                      meta={"completeness": {"complete": False, "reason": "timeout"}})
    out = result_to_jury_api(partial)
    assert out["status"] == "failed" and out["verdict"] is None
    req_failed = _result(verdict="no_violation", violations=[],
                         meta={"completeness": {"stages": {"detect": "failed"}, "required_stages": ["detect"]}})
    assert result_to_jury_api(req_failed)["status"] == "failed"
    vlm_skipped = _result(verdict="no_violation", violations=[],
                          meta={"completeness": {"stages": {"vlm": "skipped"}, "required_stages": ["detect"]}})
    assert result_to_jury_api(vlm_skipped)["verdict"] == "no_violation"


def test_incomplete_analysis_still_reports_evidenced_violation():
    r = _result(meta={"completeness": {"complete": False}})
    assert result_to_jury_api(r)["verdict"] == "violation"


def test_no_violation_hides_candidates_from_jury():
    r = _result(verdict="no_violation")
    jury = result_to_jury_api(r)
    assert jury["verdict"] == "no_violation" and jury["violations"] == []
    ui = result_to_ui_report(r)
    assert ui["violations"] == [] and len(ui["candidates"]) == 1
    assert check_consistency(ui) == []


# ----------------------------------------------------------------------------- таймкоды и кадры

def test_nan_timecode_falls_back_to_frame_then_t_start():
    v = _violation(timecode_sec=NAN)
    oc = resolve_outcome(_result(violations=[v]))
    assert oc.violations[0]["timecode_sec"] == 22.0  # первый валидный кадр
    v2 = _violation(timecode_sec=float("inf"), evidence_frames=[], t_start=5.04)
    assert resolve_outcome(_result(violations=[v2])).violations[0]["timecode_sec"] == 5.0


def test_timecode_clamped_to_duration():
    r = _result(meta={"duration_sec": 20.0})
    v = result_to_ui_report(r)["violations"][0]
    assert v["timecode_sec"] == 20.0 and v["t_end"] <= 20.0
    assert any("длительности" in w for w in result_to_ui_report(r)["warnings"])


@pytest.mark.parametrize("bad", [NAN, float("inf"), -float("inf"), None, "abc"])
def test_no_nan_anywhere(bad):
    v = _violation(confidence=bad, t_start=bad, t_end=bad,
                   evidence_frames=[EvidenceFrame(bad, JPEG_B64), EvidenceFrame(22.0, JPEG_B64)])
    r = _result(violations=[v], timeline=[(bad, bad), (1.0, bad)], meta={"x": bad, "nested": [bad, {"y": bad}]})
    for out in (result_to_jury_api(r), result_to_ui_report(r)):
        json.dumps(out, allow_nan=False)  # не должно бросить
        assert out["verdict"] == "violation"
        assert len(out["violations"][0]["evidence_frames"]) == 1  # кадр без timestamp отброшен
    ui = result_to_ui_report(r)
    assert ui["violations"][0]["confidence"] == 0.0
    assert ui["timeline"] == [[1.0, 0.0]]


def test_invalid_jpeg_frames_dropped():
    png = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"0" * 20).decode()
    truncated = base64.b64encode(base64.b64decode(JPEG_B64)[:-10]).decode()
    frames = [EvidenceFrame(1.0, png), EvidenceFrame(2.0, "!!notbase64!!"), EvidenceFrame(3.0, truncated),
              EvidenceFrame(4.0, JPEG_B64)]
    ui = result_to_ui_report(_result(violations=[_violation(evidence_frames=frames, timecode_sec=4.0)]))
    assert [f["timestamp_sec"] for f in ui["violations"][0]["evidence_frames"]] == [4.0]
    assert sum("кадр" in w and "отброшен" in w for w in ui["warnings"]) == 3


def test_base64_normalisation():
    raw = base64.b64decode(JPEG_B64)
    frames = [EvidenceFrame(1.0, "data:image/jpeg;base64," + JPEG_B64),
              EvidenceFrame(2.0, raw),  # type: ignore[arg-type]  # сырые байты JPEG
              EvidenceFrame(3.0, JPEG_B64[:40] + "\n" + JPEG_B64[40:])]
    out = result_to_jury_api(_result(violations=[_violation(evidence_frames=frames, timecode_sec=1.0)]))
    assert [f["image_base64"] for f in out["violations"][0]["evidence_frames"]] == [JPEG_B64] * 3


def test_jpeg_b64_problem():
    assert jpeg_b64_problem(JPEG_B64) is None
    assert jpeg_b64_problem("") and jpeg_b64_problem(None)
    assert "data URI" in jpeg_b64_problem("data:image/jpeg;base64," + JPEG_B64)


# ----------------------------------------------------------------------------- UI-report и round-trip

def test_ui_report_fields_and_rules():
    out = result_to_ui_report(_result())
    assert tuple(out)[:4] == API_REQUIRED_TOP
    v = out["violations"][0]
    assert tuple(v)[:4] == API_REQUIRED_VIOLATION
    assert v["rule"] == v["rules"][0] and v["rules"][1]["clause"] is None
    assert out["verdict_state"] == "violation" and out["analysis_complete"] is None
    assert job_response_errors(out) == [] and job_response_errors(out, strict=True) != []
    no_rules = result_to_ui_report(_result(violations=[_violation(rules=[])]))
    assert no_rules["violations"][0]["rule"] is None


def test_ui_report_merges_violation_extra():
    r = _result(meta={"violation_extra": [{"provenance": {"sources": ["rules", "vlm"]}}]})
    assert result_to_ui_report(r)["violations"][0]["extra"] == {"provenance": {"sources": ["rules", "vlm"]}}
    assert "extra" not in result_to_jury_api(r)["violations"][0]


def test_round_trip_is_stable():
    for ser in (result_to_ui_report, result_to_jury_api):
        first = ser(_result())
        back = api_to_result(json.loads(json.dumps(first, ensure_ascii=False, allow_nan=False)))
        assert isinstance(back, AnalysisResult)
        if ser is result_to_ui_report:
            assert back.violations[0].rules[0].document == "ТЕСТ-ДОКУМЕНТ-0"
            assert ser(back) == first


def test_tz_example_round_trip_and_minimal_projection():
    res = api_to_result(TZ_EXAMPLE)
    assert res.verdict == "violation" and res.violations[0].timecode_sec == 22.0
    assert result_to_jury_api(res) == TZ_EXAMPLE
    assert minimal_api(result_to_ui_report(res)) == TZ_EXAMPLE
    assert minimal_api(res) == TZ_EXAMPLE
    assert jury_response_errors(TZ_EXAMPLE) == []


def test_accepts_plain_dict_violations():
    v = {"type": "x", "timecode_sec": 3.33, "description": "d",
         "evidence_frames": [{"timestamp_sec": 3.33, "image_base64": JPEG_B64}]}
    out = result_to_jury_api(_result(violations=[v]))  # type: ignore[list-item]
    assert out["violations"][0]["timecode_sec"] == 3.3


def test_to_jsonable_handles_numpy_and_misc():
    np = pytest.importorskip("numpy")
    obj = {"a": np.float32(1.5), "b": np.arange(3), "c": {1, 2}, "d": Path("x/y"), "e": (np.nan, 2),
           "f": EvidenceFrame(1.0, "AA==")}
    out = to_jsonable(obj)
    assert out["a"] == 1.5 and out["b"] == [0, 1, 2] and out["c"] == [1, 2]
    assert out["d"] in ("x/y", "x\\y") and out["e"] == [None, 2]
    assert out["f"] == {"timestamp_sec": 1.0, "image_base64": "AA=="}
    json.dumps(out, allow_nan=False)


def test_meta_with_numpy_serialises():
    np = pytest.importorskip("numpy")
    out = result_to_ui_report(_result(meta={"timings": {"detect": np.float64(1.23)}, "arr": np.zeros(2)}))
    assert out["meta"]["timings"]["detect"] == 1.23
    json.dumps(out, allow_nan=False)


def test_without_optional_sections():
    out = result_to_ui_report(_result(), include_timeline=False, include_meta=False)
    assert "timeline" not in out and "meta" not in out
