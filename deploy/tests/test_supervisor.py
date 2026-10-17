import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from deploy import supervisor as sv

ROOT = Path(__file__).resolve().parents[2]
SUP = [sys.executable, str(ROOT / "deploy" / "supervisor.py")]
FAKE_TUNNEL = [sys.executable, "-c", "import sys,time; print('INF |  https://abc-def-1.trycloudflare.com  |', "
                                     "file=sys.stderr, flush=True); time.sleep(600)"]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _env(tmp_path: Path) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SENTINEL_", "TUNNEL_"))}
    logs = tmp_path / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "trial_synth.mov").write_bytes(b"\x00" * 1024)  # заглушке декодировать не нужно
    env.update({"SENTINEL_LOG_DIR": str(logs), "SENTINEL_API_KEYS": "k1-test-key-0123456789",
                "SENTINEL_DATA_DIR": str(tmp_path / "data")})
    return env


def _wait(pred, timeout=40.0, step=0.2):
    end = time.time() + timeout
    while time.time() < end:
        v = pred()
        if v:
            return v
        time.sleep(step)
    return None


def _state(tmp_path):
    try:
        return json.loads((tmp_path / "logs/state/state.json").read_text())
    except (OSError, ValueError):
        return {}


@pytest.mark.parametrize("hide_psutil", [False, True])
def test_full_cycle_single_instance_and_stop(tmp_path, hide_psutil):
    port, env = _free_port(), _env(tmp_path)
    if hide_psutil:  # путь без psutil: /proc + killpg (на Windows — tasklist/taskkill)
        fake = tmp_path / "nopsutil" / "psutil"
        fake.mkdir(parents=True)
        (fake / "__init__.py").write_text("raise ImportError('hidden for test')\n")
        env["PYTHONPATH"] = str(fake.parent)
    api = json.dumps([sys.executable, str(ROOT / "deploy/stub_api.py"), "--port", str(port)])
    args = ["run", "--port", str(port), "--interval", "0.2", "--allow-stub", "--api-cmd", api,
            "--tunnel-cmd", json.dumps(FAKE_TUNNEL)]
    p = subprocess.Popen(SUP + args, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        st = _wait(lambda: (s := _state(tmp_path)) and (s.get("tunnel") or {}).get("url") and s)
        assert st, (p.poll(), (tmp_path / "logs/supervisor.log").read_text() if (tmp_path / "logs/supervisor.log").exists() else "")
        assert st["phase"] == "ready" and st["tunnel"]["url"] == "https://abc-def-1.trycloudflare.com"
        assert (tmp_path / "logs/public_url.txt").read_text().strip() == st["tunnel"]["url"]
        ready = json.loads((tmp_path / "logs/ready.json").read_text())
        assert ready["ok"] and ready["no_key_status"] == 401
        assert "k1-test-key" not in (tmp_path / "logs/supervisor.log").read_text()  # ключ не логируется
        r = subprocess.run(SUP + ["status", "--json"], env=env, capture_output=True, text=True, timeout=30)
        assert r.returncode == 0 and json.loads(r.stdout)["ok"]
        second = subprocess.run(SUP + args, env=env, capture_output=True, text=True, timeout=30)
        assert second.returncode == 3  # второй экземпляр не стартует
        api_pid, tun_pid = st["api"]["pid"], st["tunnel"]["pid"]
        assert sv.pid_alive(api_pid) and sv.pid_alive(tun_pid)
        r = subprocess.run(SUP + ["stop"], env=env, capture_output=True, text=True, timeout=90)
        assert r.returncode == 0
        assert p.wait(timeout=30) == 0
        assert _wait(lambda: not sv.pid_alive(api_pid) and not sv.pid_alive(tun_pid), timeout=15)
        assert _state(tmp_path)["phase"] == "stopped"
    finally:
        if p.poll() is None:
            p.kill()


def test_crash_loop_stops_supervisor(tmp_path):
    port, env = _free_port(), _env(tmp_path)
    api = json.dumps([sys.executable, "-c", "import sys; print('CUDA out of memory', file=sys.stderr); sys.exit(3)"])
    r = subprocess.run(SUP + ["run", "--port", str(port), "--interval", "0.05", "--no-tunnel", "--api-cmd", api],
                       env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 4, r.stdout + r.stderr
    st = _state(tmp_path)
    assert st["phase"] == "crash_loop" and "GPU OOM" in st["last_error"]


def test_foreign_process_on_port_is_not_touched(tmp_path):
    port, env = _free_port(), _env(tmp_path)
    foreign = socket.socket()
    foreign.bind(("127.0.0.1", port))
    foreign.listen(1)
    try:
        r = subprocess.run(SUP + ["run", "--port", str(port), "--interval", "0.1", "--no-tunnel",
                                  "--api-cmd", json.dumps([sys.executable, "-c", "pass"])],
                           env=env, capture_output=True, text=True, timeout=60)
        assert r.returncode == 5
        assert "занят" in _state(tmp_path)["last_error"]
        foreign.getsockname()  # наш «чужой» сокет жив
    finally:
        foreign.close()


def test_is_ours_rejects_reused_pid():
    me = {"pid": os.getpid(), "create_time": sv.create_time(os.getpid())}
    assert sv.is_ours(me)
    if me["create_time"] is not None:
        assert not sv.is_ours({**me, "create_time": me["create_time"] - 1000})
    assert not sv.is_ours({"pid": 0}) and not sv.is_ours(None)


def test_status_without_state(tmp_path):
    assert sv.status(sv.Store(tmp_path / "none")) == 1


def test_api_key_sources(tmp_path):
    assert sv.api_key({"SENTINEL_API_KEYS": " a1 , b2 "}, tmp_path) == "a1"
    (tmp_path / "api_keys.txt").write_text("# comment\nfilekey123\n")
    assert sv.api_key({}, tmp_path) == "filekey123"
