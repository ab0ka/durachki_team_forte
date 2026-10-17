"""jury_sim против локального фейкового API (http.server) — проверяет код симулятора, не сервис."""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from eval.common import load_config

from .conftest import make_pred, make_violation

KEY = "test-key"


def _server(enforce_auth: bool = True, max_bytes: int = 1 << 20):
    jobs: dict[str, float] = {}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # тишина
            pass

        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authed(self) -> bool:
            return not enforce_auth or self.headers.get("X-API-Key") == KEY

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            if n > max_bytes:
                self.rfile.read(n)
                return self._send(413, {"detail": "too large"})
            body = self.rfile.read(n)
            if not self._authed():
                return self._send(401, {"detail": "bad key"})
            m = re.search(rb'filename="([^"]+)"', body)
            if not m or not m.group(1).lower().endswith((b".mov", b".mp4")):
                return self._send(415, {"detail": "unsupported"})
            jid = uuid.uuid4().hex
            jobs[jid] = time.time()
            self._send(200, {"job_id": jid})

        def do_GET(self):
            if not self._authed():
                return self._send(401, {"detail": "bad key"})
            jid = self.path.rstrip("/").split("/")[-1]
            if jid not in jobs:
                return self._send(404, {"detail": "nf"})
            if time.time() - jobs[jid] < 0.2:
                return self._send(200, {"job_id": jid, "status": "processing"})
            self._send(200, make_pred(jid, [make_violation(3.0)]) | {"job_id": jid})

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.fixture
def videos(tmp_path: Path) -> list[Path]:
    out = []
    for i in range(3):
        p = tmp_path / f"X-{i + 1}.mov"
        p.write_bytes(b"\x00" * 1000)
        out.append(p)
    return out


def _cfg():
    return load_config(overrides={"jury_sim": {"poll_s": 0.05, "total_limit_s": 5, "big_file_mb": 2}})


def _session():
    import requests

    s = requests.Session()
    s.trust_env = False          # локальный сервер — мимо прокси окружения
    return s


def test_good_server_all_pass(videos):
    pytest.importorskip("requests")
    from eval.jury_sim import format_table, run_jury_sim

    srv = _server()
    try:
        res = run_jury_sim(f"http://127.0.0.1:{srv.server_port}", KEY, videos, _cfg(), concurrency=2,
                           session=_session())
    finally:
        srv.shutdown()
    st = {c["name"]: c["status"] for c in res["checks"]}
    assert res["summary"]["n_fail"] == 0, format_table(res)
    assert any(n.startswith("sequential") for n in st) and any(n.startswith("bounded stress") for n in st)
    seq = next(c for c in res["checks"] if c["name"].startswith("sequential"))
    assert all(j["status"] == "done" and j["accept_to_done_s"] < 5 for j in seq["data"]["jobs"])
    assert "image_base64" not in json.dumps(res)            # в отчёте только статусы и тайминги


def test_broken_auth_detected(videos):
    pytest.importorskip("requests")
    from eval.jury_sim import run_jury_sim

    srv = _server(enforce_auth=False)
    try:
        res = run_jury_sim(f"http://127.0.0.1:{srv.server_port}", KEY, videos[:1], _cfg(), skip_stress=True, big_mb=0,
                           session=_session())
    finally:
        srv.shutdown()
    fails = [c["name"] for c in res["checks"] if c["status"] == "FAIL"]
    assert any("401" in n for n in fails)
