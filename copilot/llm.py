"""LLM-бэкенды Sentinel Copilot. Всё локально: никаких облачных API (правило 1).

* ``mock`` (SENTINEL_MOCK=1) / ``extractive`` (только явно в конфиге) — без модели: детерминированный
  ответ из фактов результата и норм. НЕ подставляется автоматически вместо LLM: если модели нет,
  ассистент честно сообщает «недоступен» (HTTP 501).
* ``llama_cpp`` — llama-cpp-python (MIT) в процессе, GGUF Q4_K_M, загрузка по требованию,
  частичный offload на GPU по свободной VRAM, выгрузка по простою (VRAM нужна VLM).
* ``llama_server`` — локальный llama.cpp server (OpenAI-совместимый /v1/chat/completions),
  разрешены только loopback/частные адреса.
* ``scripted`` — заранее заданные ответы (для тестов и оценки пост-проверки).
"""
from __future__ import annotations

import gc
import importlib.util
import ipaddress
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence
from urllib.parse import urlparse

from .config import default_config, is_mock, resolve_path

log = logging.getLogger(__name__)

Message = dict[str, str]


class LLMTimeout(RuntimeError):
    """Генерация не уложилась в бюджет."""


class LLMBusy(RuntimeError):
    """Модель/GPU заняты и не освободились до дедлайна."""


# ----------------------------------------------------------------------------- think-фильтр

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.S)


def strip_think(text: str) -> str:
    """Убрать рассуждения Qwen3 (<think>…</think>), в т.ч. незакрытый блок."""
    text = _THINK_RE.sub("", text or "")
    if "<think>" in text:
        text = text.split("<think>", 1)[0]
    return text.strip()


# ----------------------------------------------------------------------------- VRAM

def free_vram_mb() -> Optional[float]:
    """Свободная VRAM (МБ) без обязательного импорта torch; None — GPU не найден."""
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available():
                free, _total = torch.cuda.mem_get_info()
                return free / 2**20
        except Exception:
            pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3, check=False,
        )
        vals = [float(x) for x in out.stdout.split() if x.strip().replace(".", "", 1).isdigit()]
        return max(vals) if vals else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def plan_gpu_layers(free_mb: Optional[float], model_mb: float, n_layers: int,
                    reserve_mb: float = 1500, kv_mb: float = 400) -> int:
    """Сколько слоёв отдать на GPU: -1 — все, 0 — CPU, N — частичный offload."""
    if free_mb is None or model_mb <= 0 or n_layers <= 0:
        return 0
    budget = free_mb - reserve_mb - kv_mb
    if budget <= 0:
        return 0
    if budget >= model_mb:
        return -1
    per_layer = model_mb / n_layers
    return max(0, min(n_layers, int(budget // per_layer)))


# ----------------------------------------------------------------------------- бэкенды

class ExtractiveBackend:
    """Ответ собирает copilot.extractive (без генерации); stream_chat не используется."""

    name = "extractive"
    is_generative = False

    def stream_chat(self, messages: Sequence[Message], **kw: Any) -> Iterator[str]:
        raise NotImplementedError("extractive backend composes answers in copilot.extractive")

    def release(self) -> None:
        return None

    def info(self) -> dict[str, Any]:
        return {"backend": self.name}


class ScriptedLLM:
    """Тестовый «LLM»: возвращает заданные ответы по кругу (или функцию от messages)."""

    name = "scripted"
    is_generative = True

    def __init__(self, responses: Iterable[str] | Any, delay_s: float = 0.0) -> None:
        self._fn = responses if callable(responses) else None
        self._responses = [] if callable(responses) else list(responses)
        self._i = 0
        self.delay_s = delay_s
        self.calls: list[list[Message]] = []

    def stream_chat(self, messages: Sequence[Message], max_tokens: int = 512, temperature: float = 0.2,
                    deadline: Optional[float] = None, **kw: Any) -> Iterator[str]:
        self.calls.append(list(messages))
        if self._fn is not None:
            text = str(self._fn(list(messages)))
        else:
            text = self._responses[self._i % len(self._responses)] if self._responses else ""
            self._i += 1
        for piece in re.findall(r"\S+\s*", text):
            if self.delay_s:
                time.sleep(self.delay_s)
            if deadline is not None and time.monotonic() > deadline:
                raise LLMTimeout("scripted timeout")
            yield piece

    def release(self) -> None:
        return None

    def info(self) -> dict[str, Any]:
        return {"backend": self.name}


class LlamaCppBackend:
    """llama-cpp-python: ленивое создание Llama, один экземпляр, генерации сериализованы."""

    name = "llama_cpp"
    is_generative = True

    def __init__(self, model_path: str | os.PathLike[str], cfg: Optional[dict[str, Any]] = None) -> None:
        self.cfg = (cfg or default_config()).get("llm", {})
        self.model_path = str(model_path)
        self._llm: Any = None
        # обычный Lock (не RLock): допускает освобождение из другого потока
        self._lock = threading.Lock()
        self._last_used = 0.0
        self._timer: Optional[threading.Timer] = None
        self.n_gpu_layers: Optional[int] = None

    def _resolve_gpu_layers(self) -> int:
        raw = str(self.cfg.get("gpu_layers", "auto")).strip().lower()
        if raw not in {"auto", ""}:
            try:
                return int(raw)
            except ValueError:
                return 0
        try:
            model_mb = Path(self.model_path).stat().st_size / 2**20
        except OSError:
            model_mb = 0.0
        return plan_gpu_layers(free_vram_mb(), model_mb, int(self.cfg.get("model_n_layers", 36)),
                               float(self.cfg.get("vram_reserve_mb", 1500)))

    def _ensure_loaded(self) -> Any:
        if self._llm is not None:
            return self._llm
        from llama_cpp import Llama  # type: ignore  # ленивый импорт

        self.n_gpu_layers = self._resolve_gpu_layers()
        kwargs: dict[str, Any] = {
            "model_path": self.model_path,
            "n_ctx": int(self.cfg.get("n_ctx", 4096)),
            "n_gpu_layers": self.n_gpu_layers,
            "verbose": False,
        }
        if self.cfg.get("n_threads"):
            kwargs["n_threads"] = int(self.cfg["n_threads"])
        lora = resolve_path(self.cfg.get("lora_path"))
        if lora and lora.is_file():
            kwargs["lora_path"] = str(lora)
        t0 = time.monotonic()
        try:
            self._llm = Llama(**kwargs)
        except Exception:
            if self.n_gpu_layers != 0:  # нет VRAM / CUDA — пробуем CPU
                log.warning("llama.cpp GPU load failed, retry on CPU", exc_info=True)
                kwargs["n_gpu_layers"] = self.n_gpu_layers = 0
                self._llm = Llama(**kwargs)
            else:
                raise
        log.info("copilot LLM loaded %s (gpu_layers=%s) in %.1fs", Path(self.model_path).name,
                 self.n_gpu_layers, time.monotonic() - t0)
        return self._llm

    def _schedule_unload(self) -> None:
        idle = float(self.cfg.get("idle_unload_s", 300) or 0)
        if idle <= 0:
            return
        if self._timer is not None:
            self._timer.cancel()
        self._timer = threading.Timer(idle, self._unload_if_idle, args=(idle,))
        self._timer.daemon = True
        self._timer.start()

    def _unload_if_idle(self, idle: float) -> None:
        if time.monotonic() - self._last_used >= idle * 0.99:
            self.release()

    def release(self) -> None:
        """Выгрузить модель (освободить VRAM для VLM/детектора)."""
        acquired = self._lock.acquire(timeout=5)
        if not acquired:  # идёт генерация — не освобождаем модель из-под неё, повторим по таймеру
            log.warning("copilot LLM busy, unload postponed")
            self._schedule_unload()
            return
        try:
            if self._llm is not None:
                try:
                    close = getattr(self._llm, "close", None)
                    if callable(close):
                        close()
                finally:
                    self._llm = None
                    gc.collect()
        finally:
            self._lock.release()

    def warmup(self) -> None:
        with self._lock:
            self._ensure_loaded()
            self._last_used = time.monotonic()
        self._schedule_unload()

    def stream_chat(self, messages: Sequence[Message], max_tokens: int = 512, temperature: float = 0.2,
                    deadline: Optional[float] = None, **kw: Any) -> Iterator[str]:
        # ждём модель не дольше дедлайна: зависшие запросы не должны занимать потоки API
        wait = -1 if deadline is None else max(0.0, deadline - time.monotonic())
        if not self._lock.acquire(timeout=wait):
            raise LLMBusy("copilot model is busy with another request")
        try:
            llm = self._ensure_loaded()
            try:
                stream = llm.create_chat_completion(
                    messages=list(messages), stream=True, max_tokens=int(max_tokens),
                    temperature=float(temperature), top_p=float(self.cfg.get("top_p", 0.9)),
                    repeat_penalty=float(self.cfg.get("repeat_penalty", 1.05)),
                )
                for chunk in stream:
                    try:
                        delta = chunk["choices"][0].get("delta", {}).get("content")
                    except (KeyError, IndexError, AttributeError):
                        delta = None
                    if delta:
                        yield delta
                    if deadline is not None and time.monotonic() > deadline:
                        raise LLMTimeout("llama.cpp generation exceeded budget")
            finally:
                self._last_used = time.monotonic()
                self._schedule_unload()
        finally:
            self._lock.release()

    def info(self) -> dict[str, Any]:
        return {"backend": self.name, "model": Path(self.model_path).name, "gpu_layers": self.n_gpu_layers,
                "loaded": self._llm is not None}


def check_local_url(url: str, allow_private: bool = True) -> str:
    """Разрешаем только локальный инференс: loopback, (опц.) частные сети и docker-имена без точки."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"bad LLM server url: {url!r}")
    host = parsed.hostname.lower()
    if host == "localhost" or host.endswith(".localhost"):
        return url.rstrip("/")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        if allow_private and "." not in host:  # имя сервиса docker-compose
            return url.rstrip("/")
        raise ValueError(f"LLM server must be local (rule 1: no cloud APIs), got host {host!r}") from None
    if ip.is_loopback or (allow_private and (ip.is_private or ip.is_link_local)):
        return url.rstrip("/")
    raise ValueError(f"LLM server must be local (rule 1: no cloud APIs), got {host!r}")


class LlamaServerBackend:
    """Локальный llama.cpp server (`llama-server -m model.gguf --port 8081`)."""

    name = "llama_server"
    is_generative = True

    def __init__(self, url: str, cfg: Optional[dict[str, Any]] = None) -> None:
        self.cfg = (cfg or default_config()).get("llm", {})
        self.url = check_local_url(url, bool(self.cfg.get("allow_private_hosts", True)))

    def _opener(self) -> Any:
        import urllib.request

        return urllib.request.build_opener(urllib.request.ProxyHandler({}))  # никаких прокси наружу

    def healthy(self, timeout: float = 1.5) -> bool:
        try:
            with self._opener().open(f"{self.url}/health", timeout=timeout) as resp:
                return 200 <= resp.status < 300
        except Exception:
            return False

    def stream_chat(self, messages: Sequence[Message], max_tokens: int = 512, temperature: float = 0.2,
                    deadline: Optional[float] = None, **kw: Any) -> Iterator[str]:
        import urllib.request

        body = json.dumps({
            "messages": list(messages), "stream": True, "max_tokens": int(max_tokens),
            "temperature": float(temperature), "top_p": float(self.cfg.get("top_p", 0.9)),
        }).encode("utf-8")
        req = urllib.request.Request(f"{self.url}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
        timeout = max(1.0, (deadline - time.monotonic()) if deadline else float(self.cfg.get("timeout_s", 45)))
        with self._opener().open(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    delta = json.loads(payload)["choices"][0].get("delta", {}).get("content")
                except (ValueError, KeyError, IndexError):
                    delta = None
                if delta:
                    yield delta
                if deadline is not None and time.monotonic() > deadline:
                    raise LLMTimeout("llama server generation exceeded budget")

    def release(self) -> None:
        return None

    def info(self) -> dict[str, Any]:
        return {"backend": self.name, "url": self.url}


# ----------------------------------------------------------------------------- выбор

def find_model(cfg: dict[str, Any]) -> Optional[Path]:
    for cand in cfg.get("llm", {}).get("model_candidates") or []:
        p = resolve_path(cand)
        if p and p.is_file():
            return p
    return None


def make_llm(cfg: Optional[dict[str, Any]] = None, warnings: Optional[list[str]] = None) -> Optional[Any]:
    """Выбор бэкенда. auto: llama_server (если задан и жив) -> llama_cpp (если есть GGUF и библиотека).

    Если локальной модели нет — возвращает None (честное «недоступно», без подмены ответа и без
    облачных API). ``mock`` — только при SENTINEL_MOCK=1 (auto) или явно; ``extractive`` — только явно.
    Модель здесь НЕ загружается (загрузка — при первом вопросе или warmup()).
    """
    cfg = cfg or default_config()
    lcfg = cfg.get("llm", {})
    backend = str(lcfg.get("backend", "auto")).lower()
    warn = warnings if warnings is not None else []
    if backend == "extractive":
        return ExtractiveBackend()
    if backend == "mock" or (is_mock() and backend == "auto"):
        mock = ExtractiveBackend()
        mock.name = "mock"
        return mock
    if backend in {"auto", "llama_server"} and lcfg.get("server_url"):
        try:
            srv = LlamaServerBackend(str(lcfg["server_url"]), cfg)
            if srv.healthy():
                return srv
            warn.append("llama.cpp server не отвечает")
        except ValueError as exc:
            warn.append(str(exc))
    if backend in {"auto", "llama_cpp"}:
        path = find_model(cfg)
        if path is None:
            warn.append("GGUF-модель не найдена (llm.model_candidates)")
        elif importlib.util.find_spec("llama_cpp") is None:
            warn.append("llama-cpp-python не установлен")
        else:
            return LlamaCppBackend(path, cfg)
    if backend not in {"auto", "llama_cpp", "llama_server"}:
        warn.append(f"неизвестный llm.backend: {backend!r}")
    return None
