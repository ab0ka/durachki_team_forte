from __future__ import annotations

import base64
import json
import math
from pathlib import Path

import pytest

from contracts.sentinel_contracts import AnalysisResult, EvidenceFrame, RuleRef, Violation
from contracts.serialize import (
    API_REQUIRED_FRAME,
    API_REQUIRED_TOP,
    API_REQUIRED_VIOLATION,
    api_to_result,
    minimal_api,
    result_to_api,
    to_jsonable,
)
from contracts.validate import job_response_errors

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
JPEG_B64 = json.loads((EXAMPLES / "job_done_tz.json").read_text(encoding="utf-8"))["violations"][0][
    "evidence_frames"][0]["image_base64"]


def _rule(i: int = 0) -> RuleRef:
    # Тестовая заглушка — НЕ норма. Реальные ссылки только из norms/catalog.yaml (A13).
    return RuleRef(document=f"ТЕСТ-ДОКУМЕНТ-{i}", clause=None if i else "п. TEST", url="https://example.invalid/t",
                   quote="тестовая цитата", verified=False)


def _result(**kw) -> AnalysisResult:
    v = Violation(
        type="Посторонний в кассовом помещении",
        timecode_sec=22.04,
        description="На кадре виден второй человек за кассовым столом.",
        evidence_frames=[EvidenceFrame(21.96, JPEG_B64), EvidenceFrame(23.449, JPEG_B64)],
        type_id="V01",
        confidence=0.8766,
        t_start=21.25,
        t_end=26.81,
        rules=[_rule(0), _rule(1)],
    )
    base = dict(job_id="job-1", status="done", verdict="violation", violations=[v],
                timeline=[(0.0, 0.1), (0.21, 0.5), (22.04, 0.93)],
                meta={"version": "0.1.0", "timings": {"total": 12.3456}, "progress": 1.0, "stage": "done"})
    base.update(kw)
    return AnalysisResult(**base)


def test_required_fields_exact_order_and_types():
    out = result_to_api(_result())
    assert tuple(out)[:4] == API_REQUIRED_TOP
    v = out["violations"][0]
    assert tuple(v)[:4] == API_REQUIRED_VIOLATION
    assert tuple(v["evidence_frames"][0]) == API_REQUIRED_FRAME
    assert isinstance(v["timecode_sec"], float) and isinstance(v["evidence_frames"][0]["image_base64"], str)
    assert job_response_errors(out) == []


def test_rounding_to_tenth():
    out = result_to_api(_result())
    v = out["violations"][0]
    assert v["timecode_sec"] == 22.0
    assert [f["timestamp_sec"] for f in v["evidence_frames"]] == [22.0, 23.4]
    assert (v["t_start"], v["t_end"]) == (21.2, 26.8)
    assert v["confidence"] == 0.877
    assert out["timeline"] == [[0.0, 0.1], [0.2, 0.5], [22.0, 0.93]]


def test_rules_list_and_first_rule():
    v = result_to_api(_result())["violations"][0]
    assert len(v["rules"]) == 2
    assert v["rule"] == v["rules"][0]
    assert v["rule"]["clause"] == "п. TEST" and v["rules"][1]["clause"] is None
    no_rules = _result(violations=[Violation("x", 1.0, "d", [EvidenceFrame(1.0, JPEG_B64)])])
    assert result_to_api(no_rules)["violations"][0]["rule"] is None


def test_round_trip_is_stable():
    first = result_to_api(_result())
    text = json.dumps(first, ensure_ascii=False, allow_nan=False)
    back = api_to_result(json.loads(text))
    assert isinstance(back, AnalysisResult)
    assert back.violations[0].rules[0].document == "ТЕСТ-ДОКУМЕНТ-0"
    assert back.violations[0].evidence_frames[1].timestamp_sec == 23.4
    assert result_to_api(back) == first


def test_tz_example_round_trip_and_minimal_projection():
    tz = json.loads((EXAMPLES / "job_done_tz.json").read_text(encoding="utf-8"))
    res = api_to_result(tz)
    assert res.verdict == "violation" and res.violations[0].timecode_sec == 22.0
    out = result_to_api(res)
    assert minimal_api(out) == tz
    assert minimal_api(res) == tz
    assert job_response_errors(out) == []


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), None, "abc"])
def test_no_nan_anywhere(bad):
    v = Violation(type="", timecode_sec=bad, description="", evidence_frames=[EvidenceFrame(bad, JPEG_B64)],
                  type_id="V05", confidence=bad, t_start=bad, t_end=bad)
    r = AnalysisResult(job_id="j", status="done", verdict="violation", violations=[v],
                       timeline=[(bad, bad), (1.0, bad)], meta={"x": bad, "nested": [bad, {"y": bad}]})
    out = result_to_api(r)
    json.dumps(out, allow_nan=False)  # не должно бросить
    vv = out["violations"][0]
    assert vv["timecode_sec"] == 0.0 and vv["confidence"] == 0.0
    assert vv["type"]  # подставлено название по type_id
    assert vv["description"]  # подставлено нейтральное описание
    assert all(math.isfinite(p[0]) and math.isfinite(p[1]) for p in out["timeline"])
    assert job_response_errors(out) == []


def test_processing_hides_verdict_and_violations():
    r = _result(status="processing", verdict="violation", meta={"progress": 0.4, "stage": "detect"})
    out = result_to_api(r)
    assert out["verdict"] is None and out["violations"] == []
    assert out["progress"] == 0.4 and out["stage"] == "detect"
    assert job_response_errors(out) == []


def test_done_without_verdict_is_derived():
    assert result_to_api(_result(verdict=None))["verdict"] == "violation"
    assert result_to_api(_result(verdict=None, violations=[]))["verdict"] == "no_violation"


def test_failed_status_and_unknown_status():
    out = result_to_api(AnalysisResult(job_id="j", status="failed", meta={"error": "decode error"}))
    assert out["verdict"] is None and out["error"] == "decode error"
    weird = AnalysisResult(job_id="j", status="weird")  # type: ignore[arg-type]
    assert result_to_api(weird)["status"] == "failed"


def test_base64_normalisation():
    raw = base64.b64decode(JPEG_B64)
    frames = [EvidenceFrame(1.0, "data:image/jpeg;base64," + JPEG_B64),
              EvidenceFrame(2.0, raw),  # type: ignore[arg-type]  # сырые байты JPEG
              EvidenceFrame(3.0, JPEG_B64[:40] + "\n" + JPEG_B64[40:])]
    out = result_to_api(_result(violations=[Violation("t", 1.0, "d", frames)]))
    assert [f["image_base64"] for f in out["violations"][0]["evidence_frames"]] == [JPEG_B64] * 3


def test_accepts_plain_dict_violations():
    v = {"type": "x", "timecode_sec": 3.33, "description": "d",
         "evidence_frames": [{"timestamp_sec": 3.33, "image_base64": JPEG_B64}]}
    out = result_to_api(_result(violations=[v]))  # type: ignore[list-item]
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
    out = result_to_api(_result(meta={"timings": {"detect": np.float64(1.23)}, "arr": np.zeros(2)}))
    assert out["meta"]["timings"]["detect"] == 1.23
    json.dumps(out, allow_nan=False)


def test_without_optional_sections():
    out = result_to_api(_result(), include_timeline=False, include_meta=False)
    assert "timeline" not in out and "meta" not in out
