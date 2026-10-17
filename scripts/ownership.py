"""Таблица владения файлами монорепо (владелец A01). Используется check_ownership.py и CI."""
from __future__ import annotations

import re
from typing import Optional

# Агент -> префиксы путей (папки с "/" на конце или точные файлы).
OWNERS: dict[str, tuple[str, ...]] = {
    "A01": ("contracts/", "testkit/", ".github/", "scripts/", "pyproject.toml", ".gitignore", ".gitattributes",
            "Makefile", "README.md"),
    "A02": ("backend/api/",),
    "A03": ("backend/pipeline/",),
    "A04": ("ml/ingest/",),
    "A05": ("ml/detect/",),
    "A06": ("ml/track/",),
    "A07": ("ml/pose/",),
    "A08": ("ml/scene/",),
    "A09": ("ml/rules/",),
    "A10": ("ml/temporal/",),
    "A11": ("ml/vlm/",),
    "A12": ("ml/fusion/",),
    "A13": ("norms/",),
    "A14": ("copilot/",),
    "A15": ("frontend/",),
    "A16": ("eval/",),
    "A17": ("deploy/",),
    "A18": ("docs/", "pitch/"),
}

MODULES: dict[str, str] = {
    "A02": "FastAPI, API-ключ, очередь задач", "A03": "Оркестратор, бюджет времени, кэш, CLI",
    "A04": "Декодирование видео, сэмплинг", "A05": "Детекция объектов", "A06": "Трекинг, роли, авто-зоны",
    "A07": "Поза и жесты рук", "A08": "Состояния сцены", "A09": "Движок правил", "A10": "Временная модель",
    "A11": "Локальная VLM", "A12": "Слияние, вердикт, доказательства", "A13": "Нормативка РК",
    "A14": "LLM-ассистент проверяющего", "A15": "Веб-интерфейс", "A16": "Метрики, LOO, симулятор жюри",
    "A17": "Docker, офлайн-веса, Windows, Tunnel", "A18": "Документы, питч",
    "A01": "Контракты, скелет, тест-кит, CI",
}

# Пустые файлы, которые разрешено создавать всем (у всех идентичны — конфликта нет).
SHARED_FILES: frozenset[str] = frozenset({"ml/__init__.py", "backend/__init__.py"})

# Типы файлов, которые нельзя коммитить никому (правило 6).
FORBIDDEN_RE = re.compile(
    r"\.(mov|mp4|avi|mkv|m4v|webm|onnx|pt|pth|ckpt|safetensors|gguf|ggml|engine|trt|plan|tflite)$", re.I)
JPEG_RE = re.compile(r"\.(jpe?g)$", re.I)
JPEG_ALLOWED_PREFIXES = ("docs/assets/", "pitch/assets/", "frontend/public/")

_TITLE_RE = re.compile(r"\[\s*(A\d{2})\s*\]", re.I)


def owner_of(path: str) -> Optional[str]:
    """Агент-владелец пути (самый длинный совпавший префикс); None — ничей."""
    path = path.replace("\\", "/")
    if path.startswith("./"):
        path = path[2:]
    best: tuple[int, Optional[str]] = (0, None)
    for agent, prefixes in OWNERS.items():
        for p in prefixes:
            if (path == p or (p.endswith("/") and path.startswith(p))) and len(p) > best[0]:
                best = (len(p), agent)
    return best[1]


def is_shared(path: str) -> bool:
    return path.replace("\\", "/") in SHARED_FILES


def forbidden_reason(path: str) -> Optional[str]:
    p = path.replace("\\", "/")
    if FORBIDDEN_RE.search(p):
        return "видео/веса в git запрещены (правило 6)"
    if JPEG_RE.search(p) and not p.startswith(JPEG_ALLOWED_PREFIXES):
        return "JPEG вне docs/assets, pitch/assets, frontend/public (кадры датасета в git запрещены)"
    return None


def agent_from_title(title: str | None) -> Optional[str]:
    if not title:
        return None
    m = _TITLE_RE.search(title)
    return m.group(1).upper() if m else None
