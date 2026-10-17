from __future__ import annotations

import socket

import pytest


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """Каждый тест: пустой SENTINEL_WEIGHTS, без mock/extras по умолчанию, чистый кэш бэкендов."""
    from ml.detect.backends import clear_cache

    monkeypatch.setenv("SENTINEL_WEIGHTS", str(tmp_path / "weights"))
    for k in ("SENTINEL_MOCK", "SENTINEL_DETECT_EXTRAS", "SENTINEL_DETECT_CONFIG", "SENTINEL_DETECT_BUDGET_S"):
        monkeypatch.delenv(k, raising=False)
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def mock_mode(monkeypatch):
    monkeypatch.setenv("SENTINEL_MOCK", "1")
    yield


@pytest.fixture
def no_network(monkeypatch):
    """Любая попытка сетевого соединения в тесте — ошибка."""
    def guard(*a, **k):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket, "create_connection", guard)
    yield


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: тяжёлый тест (крошечные модели transformers на CPU)")
    config.addinivalue_line("markers", "gpu: требует CUDA")
