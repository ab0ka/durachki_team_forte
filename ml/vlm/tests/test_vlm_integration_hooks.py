import sys
import types

import ml.vlm as vlm
from ml.vlm import runtime
from ml.vlm.backends.base import VLMBackend


class _Fake(VLMBackend):
    name = "fake"

    def __init__(self):
        self._loaded = False
        self.loads = 0
        self.unloads = 0

    @property
    def loaded(self):
        return self._loaded

    def load(self):
        self.loads += 1
        self._loaded = True

    def unload(self):
        self.unloads += 1
        self._loaded = False


def _install_registry(monkeypatch):
    calls = []

    class Reg:
        def __init__(self):
            self.models = {}

        def get_or_load(self, name, loader, *, vram_gb=0.0, keep_warm=True, unload=None):
            calls.append((name, vram_gb))
            if name not in self.models:
                self.models[name] = (loader(), unload)
            return self.models[name][0]

        def release(self, name):
            obj, unload = self.models.pop(name)
            unload(obj)
            return True

    reg = Reg()
    mod = types.ModuleType("backend.pipeline.registry")
    mod.ModelRegistry = types.SimpleNamespace(peek=lambda: reg)
    monkeypatch.setitem(sys.modules, "backend", types.ModuleType("backend"))
    monkeypatch.setitem(sys.modules, "backend.pipeline", types.ModuleType("backend.pipeline"))
    monkeypatch.setitem(sys.modules, "backend.pipeline.registry", mod)
    return reg, calls


def test_load_goes_through_a03_registry(monkeypatch):
    reg, calls = _install_registry(monkeypatch)
    be = _Fake()
    ok, _ = runtime.ensure_loaded(be)
    assert ok and be.loaded and calls == [("ml.vlm:qwen", 3.5)]
    reg.release("ml.vlm:qwen")  # A03 вытесняет модель под детектор
    assert be.unloads == 1 and not be.loaded
    ok, _ = runtime.ensure_loaded(be)  # следующая проверка — перезагрузка через реестр
    assert ok and be.loaded and be.loads == 2


def test_no_registry_outside_a03(monkeypatch):
    monkeypatch.setitem(sys.modules, "backend.pipeline.registry", None)  # импорт падает
    be = _Fake()
    assert runtime.ensure_loaded(be)[0] and be.loads == 1


def test_model_info_for_cache_key():
    info = vlm.model_info()
    assert info["checkpoint"] == "Qwen/Qwen3-VL-4B-Instruct" and info["prompt_version"].startswith("v1+")
    assert info["verify_mode"] in ("json", "qa") and info["version"] == vlm.__version__


def test_verify_accepts_zones_kwarg(clip):
    from ml.vlm._contract import Event, Zone
    from ml.vlm.backends.mock import MockBackend
    runtime.set_backend(MockBackend(mode="confirm"))
    out = vlm.verify(clip, [Event("V05", 1, 2, 1.5, 0.5, "rules")], 30.0,
                     zones=[Zone("desk", [(0.1, 0.3), (0.9, 0.3), (0.9, 0.6)])])
    assert out[0].extra["vlm"]["status"] == "ok"
