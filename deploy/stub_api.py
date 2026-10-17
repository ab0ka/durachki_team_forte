#!/usr/bin/env python3
"""ЗАГЛУШКА API ТЗ (stdlib) — только для тестов deploy/ и замера туннеля ДО интеграции. НЕ ПРОД.

Каждый ответ помечен заголовком X-Sentinel-Stub: 1 и полем "stub": true — healthcheck ready
отказывается считать заглушку готовым сервисом.
Эндпоинты ТЗ: POST /api/v1/analyze, GET /api/v1/jobs/{id} (X-API-Key из SENTINEL_API_KEYS), /healthz, /readyz.
Замеры туннеля: POST /__probe/echo, GET /__probe/slow?seconds=N, GET /__probe/range (Range → 206).

  SENTINEL_API_KEYS=test python deploy/stub_api.py --port 8000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# 1×1 JPEG (валидный, сгенерирован заранее; не кадр из данных)
TINY_JPEG_B64 = ("/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwc"
                 "KDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAABAAEBAREA/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAA"
                 "AAAAAAD/2gAIAQEAAD8AKp//2Q==")
RANGE_PAYLOAD = bytes(range(256)) * 4096  # 1 MiB детерминированных байт


class StubState:
    def __init__(self, keys: list[str], polls_until_done: int = 2) -> None:
        self.keys, self.polls_until_done = keys, polls_until_done
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()


def make_handler(state: StubState):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # тихо
            pass

        def _send(self, code: int, payload, ctype: str = "application/json", extra: dict | None = None) -> None:
            body = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Sentinel-Stub", "1")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _authed(self) -> bool:
            return bool(state.keys) and self.headers.get("X-API-Key", "") in state.keys

        def _read_body(self) -> bytes:
            n = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(n) if n else b""

        def do_GET(self):  # noqa: N802
            u = urllib.parse.urlsplit(self.path)
            q = urllib.parse.parse_qs(u.query)
            if u.path in ("/healthz", "/readyz"):
                return self._send(200, {"status": "ok", "stub": True})
            if u.path == "/__probe/slow":
                secs = min(float(q.get("seconds", ["1"])[0]), 600.0)
                time.sleep(secs)
                return self._send(200, {"slept_s": secs, "stub": True})
            if u.path == "/__probe/range":
                rng = re.match(r"bytes=(\d+)-(\d*)", self.headers.get("Range", ""))
                if rng:
                    a = int(rng.group(1))
                    b = int(rng.group(2)) if rng.group(2) else len(RANGE_PAYLOAD) - 1
                    return self._send(206, RANGE_PAYLOAD[a:b + 1], "application/octet-stream",
                                      {"Content-Range": f"bytes {a}-{b}/{len(RANGE_PAYLOAD)}", "Accept-Ranges": "bytes"})
                return self._send(200, RANGE_PAYLOAD, "application/octet-stream", {"Accept-Ranges": "bytes"})
            m = re.fullmatch(r"/api/v1/jobs/([\w-]+)", u.path)
            if m:
                if not self._authed():
                    return self._send(401, {"detail": "invalid api key"})
                with state.lock:
                    job = state.jobs.get(m.group(1))
                    if job is None:
                        return self._send(404, {"detail": "not found"})
                    job["polls"] += 1
                    done = job["polls"] > state.polls_until_done
                if not done:
                    return self._send(200, {"job_id": m.group(1), "status": "processing", "stub": True})
                return self._send(200, {
                    "job_id": m.group(1), "status": "done", "verdict": "violation", "stub": True,
                    "violations": [{"type": "ЗАГЛУШКА (stub)", "timecode_sec": 1.0, "type_id": "V99",
                                    "description": "Ответ заглушки deploy/stub_api.py — не результат анализа.",
                                    "evidence_frames": [{"timestamp_sec": 1.0, "image_base64": TINY_JPEG_B64}]}]})
            return self._send(404, {"detail": "not found", "stub": True})

        def do_POST(self):  # noqa: N802
            u = urllib.parse.urlsplit(self.path)
            body = self._read_body()
            if u.path == "/__probe/echo":
                return self._send(200, {"received_bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(),
                                        "stub": True})
            if u.path == "/api/v1/analyze":
                if not self._authed():
                    return self._send(401, {"detail": "invalid api key"})
                if b'name="file"' not in body[:4096]:
                    return self._send(422, {"detail": "multipart field 'file' required"})
                jid = uuid.uuid4().hex
                with state.lock:
                    state.jobs[jid] = {"polls": 0, "bytes": len(body)}
                return self._send(200, {"job_id": jid})
            return self._send(404, {"detail": "not found", "stub": True})

    return H


def serve(host: str = "127.0.0.1", port: int = 8000, keys: list[str] | None = None,
          polls_until_done: int = 2) -> ThreadingHTTPServer:
    keys = keys if keys is not None else [k.strip() for k in (os.environ.get("SENTINEL_API_KEYS") or "").split(",")
                                          if k.strip()]
    srv = ThreadingHTTPServer((host, port), make_handler(StubState(keys, polls_until_done)))
    srv.daemon_threads = True
    return srv


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)
    srv = serve(a.host, a.port)
    print(f"!!! STUB API (не прод) на http://{a.host}:{srv.server_address[1]} — ответы фиктивные", file=sys.stderr)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
