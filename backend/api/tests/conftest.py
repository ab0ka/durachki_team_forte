"""Фикстуры: приложение на tmp_path, синтетические видео, тестовые пайплайны. Без GPU и весов."""
from __future__ import annotations

import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from backend.api.config import Settings  # noqa: E402
from backend.api.fake_pipeline import _PLACEHOLDER_JPEG  # noqa: E402
from backend.api.worker import PipelineSpec  # noqa: E402

try:
    from contracts.sentinel_contracts import AnalysisResult, EvidenceFrame, Violation  # noqa: E402
except ImportError:
    from backend.api._contract_fallback import AnalysisResult, EvidenceFrame, Violation  # noqa: E402

KEY = "test-key-123"
H = {"X-API-Key": KEY}
# минимальный ISO-BMFF: проходит сигнатуру, но НЕ декодируется
FTYP_ONLY = (b"\x00\x00\x00\x14ftypqt  \x00\x00\x02\x00qt  " + b"\x00\x00\x00\x08free"
             + b"\x00" * 64)


def pytest_configure(config: Any) -> None:
    config.addinivalue_line("markers", "gpu: требует GPU")
    config.addinivalue_line("markers", "slow: медленный тест")


def make_result(job_id: str, t: float = 4.2) -> Any:
    v = Violation(type="Посторонний в кассовом помещении", timecode_sec=t,
                  description="Признаки возможного нарушения: тест.",
                  evidence_frames=[EvidenceFrame(timestamp_sec=t, image_base64=_PLACEHOLDER_JPEG)],
                  type_id="V01", confidence=0.8, t_start=t - 1, t_end=t + 1)
    return AnalysisResult(job_id=job_id, status="done", verdict="violation", violations=[v],
                          timeline=[(0.0, 0.1), (t, 0.9)], meta={"version": "test"})


def spec(run: Callable[..., Any], warmup: Callable[[], None] | None = None) -> PipelineSpec:
    return PipelineSpec(run=run, warmup=warmup, mode="real")


def ok_run(video_path: str, job_id: str, progress: Callable[[str, float], None],
           deadline_s: float = 240) -> Any:
    progress("detect", 0.5)
    return make_result(job_id)


@pytest.fixture
def settings_factory(tmp_path: Path) -> Callable[..., Settings]:
    def make(**kw: Any) -> Settings:
        base: dict[str, Any] = dict(data_dir=tmp_path / "data", api_keys=[KEY],
                                    frontend_dir=tmp_path / "no-frontend", video_validation="off",
                                    log_json=True, job_ttl_h=0, rl_auth_fail="10000/60",
                                    dedup=False)  # тесты грузят одни и те же байты; dedup — свой тест
        base.update(kw)
        return Settings.from_env(env={}, **base)
    return make


@pytest.fixture
def client_factory(settings_factory: Callable[..., Settings]) -> Iterator[Callable[..., Any]]:
    from fastapi.testclient import TestClient
    from backend.api.main import create_app
    stack: list[Any] = []

    @contextmanager
    def make(pipe: Any = None, **kw: Any) -> Iterator[Any]:
        app = create_app(settings_factory(**kw), pipeline=pipe)
        with TestClient(app) as c:
            stack.append(c)
            yield c

    yield make


def wait_job(c: Any, job_id: str, headers: dict[str, str] | None = H, prefix: str = "/api/v1",
             timeout: float = 15.0) -> dict[str, Any]:
    end = time.time() + timeout
    while time.time() < end:
        r = c.get(f"{prefix}/jobs/{job_id}", headers=headers)
        assert r.status_code == 200, r.text
        d = r.json()
        if d["status"] != "processing":
            return d
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} still processing after {timeout}s")


def wait_ready(c: Any, timeout: float = 5.0) -> Any:
    end = time.time() + timeout
    r = c.get("/readyz")
    while time.time() < end and r.json().get("state") in ("loading", "warming", "starting"):
        time.sleep(0.02)
        r = c.get("/readyz")
    return r


def upload(c: Any, data: bytes, name: str = "X-1.mov", headers: dict[str, str] | None = H,
           prefix: str = "/api/v1") -> Any:
    return c.post(f"{prefix}/analyze", headers=headers,
                  files={"file": (name, data, "video/quicktime")})


def write_mov(path: Path, seconds: float = 2.0, fps: int = 30, size: tuple[int, int] = (232, 424)) -> Path:
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not w.isOpened():
        pytest.skip("cv2 VideoWriter cannot write mp4v/.mov here")
    for i in range(int(seconds * fps)):
        f = np.full((size[1], size[0], 3), (i * 5) % 255, np.uint8)
        cv2.rectangle(f, (10 + i, 40), (80 + i, 140), (0, 200, 0), -1)
        w.write(f)
    w.release()
    return path


@pytest.fixture
def mov_bytes(tmp_path: Path) -> bytes:
    return write_mov(tmp_path / "syn.mov").read_bytes()


class Gate:
    """Пайплайн, который ждёт сигнала (для проверки очереди/рестарта)."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.started = threading.Event()
        self.calls: list[tuple[str, float]] = []

    def run(self, video_path: str, job_id: str, progress: Callable[[str, float], None],
            deadline_s: float = 240) -> Any:
        self.calls.append((job_id, deadline_s))
        self.started.set()
        self.release.wait(10)
        return make_result(job_id)
