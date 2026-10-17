#!/usr/bin/env python3
"""Запрет коммита медиа, весов и крупных файлов (владелец A01). Репозиторий публичный.

    python scripts/guard_media.py file1 file2 ...   # так вызывает pre-commit (.pre-commit-config.yaml)
    python scripts/guard_media.py --staged          # файлы из индекса git
    python scripts/guard_media.py --all             # все отслеживаемые файлы (CI)

Ошибка (код 1):
* видео и веса по расширению: .mov .mp4 .avi .mkv .m4v .webm .onnx .pt .pth .ckpt .safetensors .gguf .ggml
  .engine .trt .plan .tflite;
* JPEG (кадры датасета и доказательства) вне ``docs/``, ``pitch/``, ``frontend/``;
* содержимое видео/весов/JPEG под другим расширением (сигнатуры ``ftyp``, ``GGUF``, ``FF D8 FF``);
* любой файл больше ``--max-mb`` (по умолчанию 5 МБ).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

MAX_MB_DEFAULT = 5.0
MEDIA_EXT_RE = re.compile(
    r"\.(mov|mp4|avi|mkv|m4v|webm|onnx|pt|pth|ckpt|safetensors|gguf|ggml|engine|trt|plan|tflite)$", re.I)
JPEG_EXT_RE = re.compile(r"\.(jpe?g)$", re.I)
ASSET_PREFIXES = ("docs/", "pitch/", "frontend/")


def _norm(path: str) -> str:
    p = path.replace("\\", "/")
    return p[2:] if p.startswith("./") else p


def name_problem(path: str) -> Optional[str]:
    """Проблема по одному имени файла (без чтения содержимого)."""
    p = _norm(path)
    if MEDIA_EXT_RE.search(p):
        return "видео/веса в git запрещены (правило 6)"
    if JPEG_EXT_RE.search(p) and not p.startswith(ASSET_PREFIXES):
        return "JPEG вне docs/, pitch/, frontend/ (кадры датасета и доказательства в git запрещены)"
    return None


def content_problem(head: bytes, path: str) -> Optional[str]:
    """Сигнатуры медиа/весов под чужим расширением."""
    p = _norm(path)
    if len(head) >= 8 and head[4:8] == b"ftyp":
        return "содержимое — видео-контейнер (ftyp)"
    if head[:4] == b"GGUF":
        return "содержимое — веса GGUF"
    if head[:3] == b"\xff\xd8\xff" and not p.startswith(ASSET_PREFIXES):
        return "содержимое — JPEG вне docs/, pitch/, frontend/"
    return None


def check_file(path: str, root: Path, max_mb: float = MAX_MB_DEFAULT) -> list[str]:
    problems = []
    why = name_problem(path)
    if why:
        problems.append(why)
    f = root / path
    if f.is_file():
        size = f.stat().st_size
        if size > max_mb * 1024 * 1024:
            problems.append(f"файл {size / 1024 / 1024:.1f} МБ > {max_mb:g} МБ")
        if not why:
            with f.open("rb") as fh:
                why2 = content_problem(fh.read(16), path)
            if why2:
                problems.append(why2)
    return problems


def _git_files(root: Path, *args: str) -> list[str]:
    out = subprocess.run(["git", *args, "-z"], cwd=root, capture_output=True, check=True).stdout
    return [x for x in out.decode("utf-8", "replace").split("\0") if x]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Запрет коммита медиа, весов и файлов > 5 МБ")
    ap.add_argument("files", nargs="*")
    ap.add_argument("--all", action="store_true", help="все отслеживаемые git файлы")
    ap.add_argument("--staged", action="store_true", help="файлы из индекса (добавленные/изменённые)")
    ap.add_argument("--max-mb", type=float, default=MAX_MB_DEFAULT)
    ap.add_argument("--root", default=".")
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()
    files = list(args.files)
    if args.all:
        files += _git_files(root, "ls-files")
    if args.staged:
        files += _git_files(root, "diff", "--cached", "--name-only", "--diff-filter=ACMR")
    bad = 0
    for path in dict.fromkeys(files):
        for problem in check_file(path, root, args.max_mb):
            print(f"BLOCKED {path}: {problem}")
            bad += 1
    if bad:
        print(f"guard_media: {bad} нарушений — уберите файлы из коммита (git rm --cached <файл>)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
