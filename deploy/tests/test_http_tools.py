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
    assert not rep["ok"] and any("ЗАГЛУШКА" in p for p in rep["problems"])
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
