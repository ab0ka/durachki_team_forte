#!/usr/bin/env python3
"""CI-лёгкие зависимости модуля: печатает requirements без torch/CUDA/компилируемых пакетов (владелец A01).

    python scripts/ci_requirements.py ml/detect      # <модуль>/requirements-ci.txt или отфильтрованный requirements.txt
    python scripts/ci_requirements.py --all          # объединение по всем модулям

Тесты модулей обязаны проходить без GPU и без весов (моки), поэтому тяжёлое в CI не ставится.
Если модулю нужно что-то особенное — положите ``<модуль>/requirements-ci.txt`` (используется как есть).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_DIRS = ["backend/api", "backend/pipeline", "ml/ingest", "ml/detect", "ml/track", "ml/pose", "ml/scene",
               "ml/rules", "ml/temporal", "ml/vlm", "ml/fusion", "norms", "copilot", "eval", "deploy"]

HEAVY = {
    "torch", "torchvision", "torchaudio", "xformers", "flash-attn", "bitsandbytes", "vllm", "llama-cpp-python",
    "onnxruntime-gpu", "tensorrt", "triton", "deepspeed", "auto-gptq", "autoawq", "tensorflow", "tensorflow-gpu",
    "jax", "jaxlib", "paddlepaddle", "paddlepaddle-gpu", "mmcv", "mmcv-full", "mmdet", "mmpose", "mmengine",
    "open-clip-torch", "timm", "accelerate", "peft", "sentence-transformers", "ultralytics", "cupy", "decord",
    "pytorch-lightning", "lightning", "optimum", "einops-exts", "unsloth", "trl",
}
HEAVY_PREFIXES = ("nvidia-", "cupy-", "torch-")
RENAME = {"opencv-python": "opencv-python-headless", "opencv-contrib-python": "opencv-contrib-python-headless"}
_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def filter_lines(lines: list[str]) -> list[str]:
    out = []
    for raw in lines:
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):  # -r, -e, --extra-index-url и т.п. — не переносим
            continue
        m = _NAME_RE.match(line)
        if not m:
            continue
        name = _norm(m.group(1).split("[", 1)[0])
        if name in HEAVY or name.startswith(HEAVY_PREFIXES):
            continue
        if name in RENAME:
            line = RENAME[name] + line[m.end():]
        out.append(line)
    return out


def module_requirements(module: str, root: Path = ROOT) -> list[str]:
    d = root / module
    ci = d / "requirements-ci.txt"
    if ci.is_file():
        return [ln.strip() for ln in ci.read_text(encoding="utf-8").splitlines()
                if ln.strip() and not ln.strip().startswith("#")]
    req = d / "requirements.txt"
    if not req.is_file():
        return []
    return filter_lines(req.read_text(encoding="utf-8").splitlines())


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    modules = MODULE_DIRS if argv[0] == "--all" else argv
    seen: dict[str, str] = {}
    for m in modules:
        for line in module_requirements(m):
            key = _norm(_NAME_RE.match(line).group(1).split("[", 1)[0]) if _NAME_RE.match(line) else line
            seen.setdefault(key, line)  # при --all берём первую встреченную версию
    print("\n".join(seen.values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
