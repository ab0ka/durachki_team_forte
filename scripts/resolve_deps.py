#!/usr/bin/env python3
"""Сведение зависимостей модулей против lock L1 (владелец A01). База — lock, версии вслепую не выбираются.

    python scripts/resolve_deps.py --lock-ref origin/data/labels:tools/dataprep/requirements-l1.lock.txt \\
        [--tree .] [--out scripts/DEPS_REPORT.md] [--write-resolved requirements-resolved.txt] [--strict]

Для каждого пакета из ``**/requirements*.txt`` модулей:

* ``OK`` — версия из lock удовлетворяет требованиям всех модулей;
* ``КОНФЛИКТ`` — версия из lock нарушает требование хотя бы одного модуля (кому и что);
* ``НЕТ В LOCK`` — пакета нет в lock: решение за L1 (добавить в lock или убрать из модуля); максимальная
  версия НЕ подставляется;
* ``ДУБЛЬ`` — пакет конфликтует с пакетом из lock по смыслу (``opencv-python`` при ``opencv-python-headless``);
* ``ЗАПРЕЩЁН`` — пакет запрещён правилами (AGPL/облачные AI-SDK).

``--write-resolved`` пишет requirements, где версии взяты ТОЛЬКО из lock, а нерешённое — комментарии ``# РЕШИТЬ``.
Код 1 с ``--strict``, если есть КОНФЛИКТ/ДУБЛЬ/ЗАПРЕЩЁН.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:  # packaging есть почти везде; иначе — вендоренная копия из pip
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.utils import canonicalize_name
    from packaging.version import InvalidVersion, Version
except ImportError:  # pragma: no cover
    from pip._vendor.packaging.requirements import InvalidRequirement, Requirement  # type: ignore
    from pip._vendor.packaging.utils import canonicalize_name  # type: ignore
    from pip._vendor.packaging.version import InvalidVersion, Version  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "data", "weights", "cache", "__pycache__", "frontend"}
DUPLICATES = {  # пакет модуля -> пакет lock, с которым он конфликтует при установке вместе
    "opencv-python": "opencv-python-headless",
    "opencv-contrib-python": "opencv-python-headless",
    "onnxruntime": "onnxruntime-gpu",
    "onnxruntime-gpu": "onnxruntime",
}
FORBIDDEN = {"ultralytics": "AGPL-3.0", "mmyolo": "GPL-3.0", "yolov5": "AGPL-3.0", "anthropic": "облачный AI-SDK",
             "google-generativeai": "облачный AI-SDK", "google-genai": "облачный AI-SDK",
             "google-cloud-vision": "облачный AI-SDK", "cohere": "облачный AI-SDK", "mistralai": "облачный AI-SDK"}


@dataclass
class Need:
    module: str
    spec: str
    line: str
    marker: Optional[str] = None


@dataclass
class Package:
    name: str
    lock: Optional[str] = None
    needs: list[Need] = field(default_factory=list)
    status: str = ""
    detail: str = ""


def parse_lock(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?\s*==\s*([^\s;]+)", line)
        if m:
            out[canonicalize_name(m.group(1))] = m.group(2)
    return out


def read_lock(path: Optional[str], ref: Optional[str]) -> dict[str, str]:
    if ref:
        text = subprocess.run(["git", "show", ref], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    elif path:
        text = Path(path).read_text(encoding="utf-8")
    else:
        return {}
    return parse_lock(text)


def module_of(rel: str) -> str:
    parts = rel.split("/")
    if parts[0] in ("ml", "backend") and len(parts) > 2:
        return "/".join(parts[:2])
    return parts[0] if len(parts) > 1 else "."


def collect(tree: Path) -> tuple[dict[str, Package], list[str]]:
    pkgs: dict[str, Package] = {}
    skipped: list[str] = []
    for f in sorted(tree.rglob("requirements*.txt")):
        rel = f.relative_to(tree).as_posix()
        if any(part in SKIP_DIRS for part in f.relative_to(tree).parts) or "lock" in f.name:
            continue
        for raw in f.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.split(" #", 1)[0].split("\t#", 1)[0].strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            try:
                req = Requirement(line)
            except InvalidRequirement:
                skipped.append(f"{rel}: {line}")
                continue
            name = canonicalize_name(req.name)
            pkg = pkgs.setdefault(name, Package(name))
            pkg.needs.append(Need(module_of(rel), str(req.specifier), line, str(req.marker) if req.marker else None))
    return pkgs, skipped


def resolve(pkgs: dict[str, Package], lock: dict[str, str]) -> None:
    for name, pkg in pkgs.items():
        pkg.lock = lock.get(name)
        if name in FORBIDDEN:
            pkg.status, pkg.detail = "ЗАПРЕЩЁН", FORBIDDEN[name]
            continue
        dup = DUPLICATES.get(name)
        if dup and dup in lock and name not in lock:
            pkg.status, pkg.detail = "ДУБЛЬ", f"в lock уже {dup}=={lock[dup]} — ставить вместе нельзя"
            continue
        if pkg.lock is None:
            specs = sorted({n.spec or "(любая)" for n in pkg.needs})
            pkg.status, pkg.detail = "НЕТ В LOCK", "требуют: " + ", ".join(specs) + " — решение за L1"
            continue
        try:
            v = Version(pkg.lock)
        except InvalidVersion:
            pkg.status, pkg.detail = "КОНФЛИКТ", f"версия в lock не по PEP 440: {pkg.lock}"
            continue
        bad = [n for n in pkg.needs if n.spec and not Requirement(f"x{n.spec}").specifier.contains(v,
                                                                                                    prereleases=True)]
        if bad:
            pkg.status = "КОНФЛИКТ"
            pkg.detail = "; ".join(f"{n.module}: {n.spec}" for n in bad) + f" ⟂ lock {pkg.lock}"
        else:
            pkg.status, pkg.detail = "OK", ""


def render_md(pkgs: dict[str, Package], lock_src: str, skipped: list[str]) -> str:
    order = {"ЗАПРЕЩЁН": 0, "КОНФЛИКТ": 1, "ДУБЛЬ": 2, "НЕТ В LOCK": 3, "OK": 4}
    rows = sorted(pkgs.values(), key=lambda p: (order.get(p.status, 9), p.name))
    counts: dict[str, int] = {}
    for p in rows:
        counts[p.status] = counts.get(p.status, 0) + 1
    lines = ["# Зависимости модулей против lock L1", "",
             f"Lock: `{lock_src}`. Сводка: " + ", ".join(f"{k} — {v}" for k, v in sorted(
                 counts.items(), key=lambda kv: order.get(kv[0], 9))) + ".", "",
             "Версии берутся только из lock. Для пакетов «НЕТ В LOCK» максимальная версия не подставляется: "
             "решение за L1 — добавить в lock или убрать из модуля.", "",
             "| Пакет | Статус | Lock | Модули (требование) | Деталь |", "|---|---|---|---|---|"]
    for p in rows:
        mods = "; ".join(sorted({f"{n.module} ({n.spec or 'любая'}{', ' + n.marker if n.marker else ''})"
                                 for n in p.needs}))
        lines.append(f"| `{p.name}` | {p.status} | {p.lock or '—'} | {mods} | {p.detail or '—'} |")
    if skipped:
        lines += ["", "Не разобраны (не PEP 508):", ""] + [f"- `{s}`" for s in skipped]
    return "\n".join(lines) + "\n"


def write_resolved(pkgs: dict[str, Package], path: Path) -> None:
    out = ["# Сгенерировано scripts/resolve_deps.py: версии ТОЛЬКО из lock L1; нерешённое — комментарии."]
    for p in sorted(pkgs.values(), key=lambda x: x.name):
        if p.status == "OK":
            out.append(f"{p.name}=={p.lock}")
        else:
            out.append(f"# РЕШИТЬ {p.name}: {p.status} — {p.detail}")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Сведение зависимостей модулей против lock L1")
    ap.add_argument("--tree", default=str(ROOT))
    ap.add_argument("--lock", default=None, help="путь к lock-файлу")
    ap.add_argument("--lock-ref", default=None, help="git-ссылка вида origin/data/labels:tools/dataprep/...lock.txt")
    ap.add_argument("--out", default=None, help="markdown-отчёт")
    ap.add_argument("--json", default=None)
    ap.add_argument("--write-resolved", default=None)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args(argv)
    lock = read_lock(args.lock, args.lock_ref)
    if not lock:
        print("lock не задан или пуст (--lock / --lock-ref) — все пакеты будут «НЕТ В LOCK»", file=sys.stderr)
    pkgs, skipped = collect(Path(args.tree).resolve())
    resolve(pkgs, lock)
    md = render_md(pkgs, args.lock_ref or args.lock or "—", skipped)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
    else:
        print(md)
    if args.json:
        Path(args.json).write_text(json.dumps({k: p.__dict__ | {"needs": [n.__dict__ for n in p.needs]}
                                               for k, p in pkgs.items()}, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    if args.write_resolved:
        write_resolved(pkgs, Path(args.write_resolved))
    bad = [p for p in pkgs.values() if p.status in ("КОНФЛИКТ", "ДУБЛЬ", "ЗАПРЕЩЁН")]
    print(f"пакетов: {len(pkgs)}; конфликтов/дублей/запрещённых: {len(bad)}", file=sys.stderr)
    return 1 if args.strict and bad else 0


if __name__ == "__main__":
    sys.exit(main())
