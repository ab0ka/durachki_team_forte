"""ModelRegistry: тёплые модели между задачами, GPU-шлюз, учёт VRAM (8 ГБ), прогрев."""
from __future__ import annotations

import contextlib
import gc
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping, Optional

from .config import cfg_get

log = logging.getLogger("sentinel.pipeline")


class GpuGate:
    """Шлюз GPU-стадий (общий для всех задач процесса).

    Обычные стадии идут параллельно до ``concurrency`` штук (detect ∥ temporal), эксклюзивные (VLM)
    ждут, пока GPU освободится полностью, и на время работы блокируют остальных.
    """

    def __init__(self, concurrency: int = 2) -> None:
        self.concurrency = max(1, int(concurrency))
        self._cond = threading.Condition()
        self._active = 0
        self._exclusive = False
        self._waiting_exclusive = 0

    def acquire(self, exclusive: bool = False, timeout: Optional[float] = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        with self._cond:
            if exclusive:
                self._waiting_exclusive += 1
            try:
                while True:
                    if exclusive:
                        ok = not self._exclusive and self._active == 0
                    else:  # обычные не обгоняют ждущую эксклюзивную стадию
                        ok = (not self._exclusive and self._active < self.concurrency
                              and self._waiting_exclusive == 0)
                    if ok:
                        if exclusive:
                            self._exclusive = True
                        else:
                            self._active += 1
                        return True
                    left = None if deadline is None else deadline - time.monotonic()
                    if left is not None and left <= 0:
                        return False
                    self._cond.wait(timeout=left if left is not None else 1.0)
            finally:
                if exclusive:
                    self._waiting_exclusive -= 1

    def release(self, exclusive: bool = False) -> None:
        with self._cond:
            if exclusive:
                self._exclusive = False
            else:
                self._active = max(0, self._active - 1)
            self._cond.notify_all()

    @contextlib.contextmanager
    def hold(self, exclusive: bool = False, timeout: Optional[float] = None) -> Iterator[bool]:
        ok = self.acquire(exclusive=exclusive, timeout=timeout)
        try:
            yield ok
        finally:
            if ok:
                self.release(exclusive=exclusive)

    def state(self) -> dict:
        with self._cond:
            return {"active": self._active, "exclusive": self._exclusive,
                    "waiting_exclusive": self._waiting_exclusive, "concurrency": self.concurrency}


@dataclass
class _Entry:
    name: str
    obj: Any
    vram_gb: float
    keep_warm: bool
    unload: Optional[Callable[[Any], None]]
    last_used: float = field(default_factory=time.monotonic)


class ModelRegistry:
    """Синглтон процесса. Модули могут (необязательно) держать модели здесь через ``get_or_load``;
    иначе оркестратор управляет памятью через необязательные хуки модулей ``warmup()``/``unload()``.
    """

    _instance: Optional["ModelRegistry"] = None
    _instance_lock = threading.Lock()

    def __init__(self, cfg: Optional[Mapping] = None) -> None:
        self._lock = threading.RLock()
        self._models: dict[str, _Entry] = {}
        self._device: Optional[str] = None
        self.cfg: Mapping = cfg or {}
        self.gate = GpuGate(int(cfg_get(self.cfg, "gpu.concurrency", 2)))
        self.warm = False
        self.warmup_report: dict[str, Any] = {}
        self.vram_measured_mb: dict[str, float] = {}

    # ---------------------------------------------------------------- синглтон
    @classmethod
    def instance(cls, cfg: Optional[Mapping] = None) -> "ModelRegistry":
        with cls._instance_lock:
            if cls._instance is None:
                if cfg is None:
                    from .config import load_config
                    cfg = load_config()
                cls._instance = cls(cfg)
            elif cfg is not None:
                cls._instance.cfg = cfg
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Только для тестов: забыть синглтон (модели освобождаются)."""
        with cls._instance_lock:
            if cls._instance is not None:
                cls._instance.release_all()
            cls._instance = None

    # ---------------------------------------------------------------- устройство
    def device(self) -> str:
        if self._device is not None:
            return self._device
        want = str(cfg_get(self.cfg, "gpu.device", "auto") or "auto").lower()
        if want == "cpu":
            self._device = "cpu"
        else:
            self._device = "cuda" if _cuda_available() else "cpu"
            if want == "cuda" and self._device != "cuda":
                log.warning("gpu.device=cuda, но CUDA недоступна — работаем на CPU")
        return self._device

    def set_device(self, device: Optional[str]) -> None:
        self._device = device

    # ---------------------------------------------------------------- модели
    def get_or_load(self, name: str, loader: Callable[[], Any], *, vram_gb: float = 0.0,
                    keep_warm: bool = True, unload: Optional[Callable[[Any], None]] = None) -> Any:
        with self._lock:
            entry = self._models.get(name)
            if entry is not None:
                entry.last_used = time.monotonic()
                return entry.obj
            if vram_gb > 0 and self.device() == "cuda":
                self.ensure_vram(vram_gb, exclude=(name,))
            obj = loader()
            self._models[name] = _Entry(name, obj, float(vram_gb), keep_warm, unload)
            return obj

    def loaded(self) -> dict[str, float]:
        with self._lock:
            return {n: e.vram_gb for n, e in self._models.items()}

    def release(self, name: str) -> bool:
        with self._lock:
            entry = self._models.pop(name, None)
        if entry is None:
            return False
        if entry.unload is not None:
            try:
                entry.unload(entry.obj)
            except Exception as exc:  # noqa: BLE001
                log.warning("unload(%s) упал: %s", name, exc)
        del entry
        self.empty_cache()
        return True

    def release_all(self, keep: tuple[str, ...] = ()) -> None:
        for name in list(self._models):
            if name not in keep:
                self.release(name)

    def ensure_vram(self, need_gb: float, exclude: tuple[str, ...] = ()) -> bool:
        """Освобождает LRU-модели реестра, пока не освободится need_gb (или нечего освобождать)."""
        budget = float(cfg_get(self.cfg, "gpu.vram_total_gb", 8.0)) - float(cfg_get(self.cfg, "gpu.vram_reserve_gb", 0.7))
        while True:
            free = self.free_vram_gb()
            used = sum(e.vram_gb for e in self._models.values())
            fits = (free is None or free >= need_gb) and used + need_gb <= budget
            if fits:
                return True
            victims = sorted((e for e in self._models.values() if e.name not in exclude),
                             key=lambda e: (e.keep_warm, e.last_used))
            if not victims:
                return free is None or free >= need_gb
            self.release(victims[0].name)

    # ---------------------------------------------------------------- VRAM-утилиты
    @staticmethod
    def free_vram_gb() -> Optional[float]:
        torch = sys.modules.get("torch")
        if torch is None:
            return None
        try:
            if not torch.cuda.is_available():
                return None
            free, _total = torch.cuda.mem_get_info()
            return free / 1024 ** 3
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def empty_cache() -> None:
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is None:
            return
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def stats(self) -> dict:
        return {"device": self.device(), "loaded": self.loaded(), "gate": self.gate.state(),
                "free_vram_gb": _round(self.free_vram_gb()), "warm": self.warm,
                "vram_measured_mb": dict(self.vram_measured_mb)}

    # ---------------------------------------------------------------- политика VLM
    def vlm_policy(self) -> str:
        policy = str(cfg_get(self.cfg, "gpu.vlm_policy", "auto") or "auto").lower()
        if policy not in ("auto", "keep", "on_demand", "off"):
            policy = "auto"
        if policy == "auto":
            if self.device() != "cuda":
                return "on_demand"
            need = float(cfg_get(self.cfg, "gpu.vram_gb.vlm_verify", 5.0))
            others = sum(float(v) for k, v in (cfg_get(self.cfg, "gpu.vram_gb", {}) or {}).items()
                         if not str(k).startswith("vlm"))
            total = float(cfg_get(self.cfg, "gpu.vram_total_gb", 8.0)) - float(cfg_get(self.cfg, "gpu.vram_reserve_gb", 0.7))
            return "keep" if need + others <= total else "on_demand"
        return policy

    def prepare_for_exclusive(self, stage: str, modules: Mapping[str, Any]) -> list[str]:
        """Перед VLM: если VRAM не хватает — выгрузить другие модели (хуки ``unload()`` модулей)."""
        notes: list[str] = []
        if self.device() != "cuda":
            return notes
        need = float(cfg_get(self.cfg, f"gpu.vram_gb.{stage}", 5.0))
        free = self.free_vram_gb()
        if free is not None and free >= need + 0.3:
            return notes
        self.ensure_vram(need)
        for mod_name, mod in modules.items():
            hook = getattr(mod, "unload", None)
            if callable(hook):
                try:
                    hook()
                    notes.append(f"{mod_name}.unload() перед {stage}")
                except Exception as exc:  # noqa: BLE001
                    notes.append(f"{mod_name}.unload() упал: {exc}")
        self.empty_cache()
        return notes


def _round(v: Optional[float]) -> Optional[float]:
    return None if v is None else round(v, 3)


def _cuda_available() -> bool:
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
        return False
    try:
        import torch  # noqa: WPS433 — лениво, только при первом вопросе об устройстве
    except Exception:  # noqa: BLE001
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False
