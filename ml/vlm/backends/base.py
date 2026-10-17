"""Интерфейс VLM-бэкенда. Все бэкенды — локальные (правило 1: никаких облачных API)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class GenResult:
    text: str
    latency_s: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    timed_out: bool = False
    error: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


class BackendUnavailable(RuntimeError):
    """Бэкенд нельзя использовать (нет весов/CUDA/сервера) — стадию VLM пропускаем."""


class VLMBackend:
    name: str = "base"
    model_id: str = ""
    supports_multi_image: bool = True

    def available(self) -> tuple[bool, str]:
        """Дешёвая проверка без загрузки весов: (можно_ли, причина)."""
        return True, ""

    def load(self) -> None:  # лениво, один раз; держим «тёплым»
        return None

    def unload(self) -> None:
        return None

    @property
    def loaded(self) -> bool:
        return True

    def generate(self, images: list[Any], system: str, prompt: str, max_new_tokens: int = 256,
                 timeout_s: float = 10.0, json_schema: Optional[dict[str, Any]] = None,
                 temperature: float = 0.0) -> GenResult:
        raise NotImplementedError

    def info(self) -> dict[str, Any]:
        return {"backend": self.name, "model": self.model_id}
