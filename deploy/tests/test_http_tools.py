import threading
from pathlib import Path

import pytest

from deploy import healthcheck, httpkit, sla_bench, stub_api, tunnel_probe


@pytest.fixture()
def stub():
    srv = stub_api.serve("127.0.0.1", 0, keys=["k1"], polls_until_done=1)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


@pytest.fixture()
def video(tmp_path):
    p = tmp_path / "v.mov"
    p.write_bytes(b"\x00" * 2048)
    return p


def test_live(stub):
    assert healthcheck.live(stub)["ok"]
    assert not healthcheck.live("http://127.0.0.1:9")["ok"]


def test_ready_requires_real_backend(stub, video):
    rep = healthcheck.ready(stub, "k1", video, limit_s=20)
    assert not rep["ok"] and rep["readyz"]["verdict"] == "bad" and "pipeline=stub" in rep["problems"][0]
    assert httpkit.request("GET", stub + "/__probe/stats").json()["n_analyze"] == 0  # задачу не слали
    rep = healthcheck.ready(stub, "k1", video, limit_s=20, allow_stub=True)
    assert rep["ok"], rep["problems"]
    assert rep["no_key_status"] == 401 and rep["verdict"] == "violation"
    assert not healthcheck.ready(stub, "wrong", video, limit_s=5, allow_stub=True)["ok"]


def test_validate_job_schema():
    good = {"job_id": "1", "status": "done", "verdict": "no_violation", "violations": []}
    assert httpkit.validate_job(good) == []
    bad = {"job_id": "1", "status": "done", "verdict": "violation",
           "violations": [{"type": "x", "timecode_sec": "22", "description": "d",
                           "evidence_frames": [{"timestamp_sec": 1, "image_base64": "aGVsbG8="}]}]}
    errs = httpkit.validate_job(bad)
    assert any("timecode_sec" in e for e in errs) and any("JPEG" in e for e in errs)
    assert httpkit.validate_job({"job_id": "1", "status": "done", "verdict": "violation", "violations": []})


def test_tunnel_probe_against_stub(stub):
    rep = tunnel_probe.probe_tunnel(stub, [0.5, 2], [0.2])
    assert rep["ok"] and rep["measured"]["max_upload_ok_mb"] == 2
    assert rep["measured"]["range_206"] and rep["measured"]["max_response_delay_ok_s"] == 0.2


def test_sla_bench_concurrent(stub, video):
    rep = sla_bench.run_bench(stub, "k1", [video] * 3, limit_s=30)
    assert rep["ok"] and rep["n"] == 3 and rep["all_done"]
    assert all(j["status"] == "done" for j in rep["jobs"])
    rep = sla_bench.run_bench(stub, "bad", [video], limit_s=5)
    assert not rep["ok"] and rep["jobs"][0]["status"].startswith("submit 401")


def test_ready_rejects_fake_pipeline_in_readyz(video):
    srv = stub_api.serve("127.0.0.1", 0, keys=["k1"], polls_until_done=0, pipeline="fake")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}"
        rep = healthcheck.ready(url, "k1", video, limit_s=10)
        assert not rep["ok"] and any("pipeline=fake" in p for p in rep["problems"])
        assert rep["readyz"]["pipeline"] == "fake"
    finally:
        srv.shutdown()


def _resp(status, body):
    import json as _json
    return httpkit.Response(status, {}, _json.dumps(body).encode() if body is not None else b"", 0.01)


@pytest.mark.parametrize("status,body,verdict", [
    (200, {"ready": True, "state": "ready", "pipeline": "real"}, "ready"),
    (200, {"ready": True, "state": "ready", "pipeline": "fake"}, "bad"),       # SENTINEL_FAKE_PIPELINE
    (503, {"ready": False, "state": "warming", "pipeline": "real"}, "waiting"),
    (503, {"ready": False, "state": "loading", "pipeline": "loading"}, "waiting"),
    (503, {"ready": False, "state": "warmup_failed", "pipeline": "real"}, "terminal"),
    (503, {"ready": False, "state": "unavailable", "pipeline": "unavailable"}, "terminal"),
    (503, {"ready": False, "state": "mock_not_allowed", "pipeline": "real"}, "terminal"),
    (503, {"ready": False, "state": "gpu_hung", "pipeline": "real"}, "terminal"),
    (404, {"detail": "not found"}, "bad"),
    (500, None, "bad"),
])
def test_readiness_matches_a02_states(monkeypatch, status, body, verdict):
    monkeypatch.setattr(healthcheck.httpkit, "request", lambda *a, **k: _resp(status, body))
    assert healthcheck.readiness("http://x")["verdict"] == verdict


def test_ready_never_submits_trial_when_warmup_failed(video):
    srv = stub_api.serve("127.0.0.1", 0, keys=["k1"], readyz_state="warmup_failed", pipeline="real")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}"
        rep = healthcheck.ready(url, "k1", video, limit_s=5, allow_stub=True)
        assert not rep["ok"] and rep["readyz"]["verdict"] == "terminal"
        assert httpkit.request("GET", url + "/__probe/stats").json()["n_analyze"] == 0
    finally:
        srv.shutdown()


def test_result_problems_mock_and_coverage():
    base = {"job_id": "1", "status": "done", "verdict": "no_violation", "violations": []}
    ok_meta = {"complete": True, "coverage": {"ratio": 1.0, "complete": True}, "api": {"pipeline": "real"}}
    assert healthcheck.result_problems({**base, "meta": ok_meta}) == []
    cases = {
        "provenance": {**ok_meta, "provenance": {"mock": True}},
        "stage": {**ok_meta, "stages": {"detect": {"status": "mock"}}},
        "api": {**ok_meta, "api": {"pipeline": "fake"}},
        "nocov": {"complete": True, "api": {"pipeline": "real"}},
        "lowcov": {**ok_meta, "coverage": {"ratio": 0.5, "complete": True}},
        "incomplete": {**ok_meta, "complete": False},
    }
    for name, meta in cases.items():
        assert healthcheck.result_problems({**base, "meta": meta}), name
    assert healthcheck.result_problems(base)                                   # нет meta → не подтверждено
    assert healthcheck.result_problems(base, jury_profile="strict") == []      # strict: meta вырезана A02
    viol = {"job_id": "1", "status": "done", "verdict": "violation", "meta": {"api": {"pipeline": "real"}},
            "violations": [{"type": "x", "timecode_sec": 1.0, "description": "d", "evidence_frames": [
                {"timestamp_sec": 1.0, "image_base64": stub_api.TINY_JPEG_B64}]}]}
    assert healthcheck.result_problems(viol) == []   # violation не требует полноты (её требует «нет нарушения»)
