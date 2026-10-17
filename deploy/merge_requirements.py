#!/usr/bin/env python3
"""Сводит requirements.txt всех модулей в deploy/requirements.in и строит lock НАСТОЯЩИМ resolver-ом.

Скрипт НЕ выбирает версии сам. Он:
  1) собирает все */requirements.txt (+ deploy/platform.in — платформенные пины torch);
  2) ищет конфликты: несовместимые диапазоны одного пакета, взаимоисключающие дистрибутивы
     (opencv-python vs opencv-python-headless), запрещённые лицензии (ultralytics …);
  3) пишет пересечение ограничений в deploy/requirements.in (без «максимальной версии»);
  4) с --lock TARGET вызывает `uv pip compile` (resolver) → deploy/locks/TARGET.lock.txt с хэшами.

Конфликт = код возврата 1 и отчёт; решение принимают люди (владелец модуля / A01 / интегратор)
правкой requirements модуля или записью в deploy/constraints.txt (передаётся resolver-у как -c).

  python deploy/merge_requirements.py                       # сбор + проверка → requirements.in
  python deploy/merge_requirements.py --check               # только отчёт, файлы не пишет
  python deploy/merge_requirements.py --lock windows-cu124  # + lock для ноутбука (нужен uv и сеть)
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

try:
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.specifiers import SpecifierSet
    from packaging.utils import canonicalize_name
    from packaging.version import InvalidVersion, Version
except ImportError:  # голый python:3.11-slim — берём vendored-копию из pip
    from pip._vendor.packaging.requirements import InvalidRequirement, Requirement  # type: ignore
    from pip._vendor.packaging.specifiers import SpecifierSet  # type: ignore
    from pip._vendor.packaging.utils import canonicalize_name  # type: ignore
    from pip._vendor.packaging.version import InvalidVersion, Version  # type: ignore

try:
    from deploy.license_policy import BANNED_PACKAGES
except ImportError:  # запуск как `python deploy/merge_requirements.py`
    from license_policy import BANNED_PACKAGES  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "env", "__pycache__", "weights", "data",
             "cache", "dist", "build", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
TORCH_FAMILY = {"torch", "torchvision", "torchaudio"}

# Цели lock-файлов. torch_backend → uv сам ходит в индекс PyTorch за torch/torchvision.
LOCK_TARGETS: dict[str, dict[str, str]] = {
    "windows-cu124": {"python_platform": "x86_64-pc-windows-msvc", "torch_backend": "cu124", "profile": "gpu"},
    "windows-cpu": {"python_platform": "x86_64-pc-windows-msvc", "torch_backend": "cpu", "profile": "cpu"},
    "linux-cu124": {"python_platform": "x86_64-manylinux_2_28", "torch_backend": "cu124", "profile": "gpu"},
    "linux-cpu": {"python_platform": "x86_64-manylinux_2_28", "torch_backend": "cpu", "profile": "cpu"},
}

# Взаимоисключающие дистрибутивы одного import-пакета. Детерминированная замена (не выбор версии):
# сервер без GUI → headless; contrib ⊃ base. Каждая замена печатается в отчёте и шапке .in.
OPENCV = {"opencv-python", "opencv-python-headless", "opencv-contrib-python", "opencv-contrib-python-headless"}
ONNXRUNTIME = {"onnxruntime", "onnxruntime-gpu", "onnxruntime-directml", "onnxruntime-openvino"}

DEV_PATTERNS = ("pytest", "pytest-*", "ruff", "black", "mypy", "flake8", "isort", "pre-commit",
                "ipykernel", "ipython", "jupyter*", "notebook", "coverage", "types-*")

# Известные проблемы установки — только предупреждение, решение за владельцем модуля.
KNOWN_TROUBLE = {
    "flash-attn": "нет колёс под Windows/cu124 — сборка из исходников с nvcc; transformers умеет sdpa",
    "xformers": "колёса жёстко пинят конкретный torch — почти всегда конфликт с platform.in",
    "llama-cpp-python": "без готового CUDA-колеса компилируется; для copilot лучше llama.cpp server",
    "decord": "последний релиз 0.6.0 без колёс cp311 под Windows; альтернатива — PyAV/eva-decord",
    "bitsandbytes": "проверить версию с колёсами под Windows (>=0.43)",
}


@dataclass
class ReqLine:
    name: str                  # canonical
    extras: frozenset[str]
    specifier: str
    marker: str
    url: str
    source: str
    lineno: int
    raw: str


@dataclass
class Merged:
    name: str
    marker: str
    extras: set[str] = field(default_factory=set)
    specs: list[tuple[str, str]] = field(default_factory=list)  # (specifier, source)
    urls: set[str] = field(default_factory=set)
    sources: list[str] = field(default_factory=list)
    note: str = ""

    def spec_set(self) -> SpecifierSet:
        return SpecifierSet(",".join(s for s, _ in self.specs if s))

    def line(self) -> str:
        extras = f"[{','.join(sorted(self.extras))}]" if self.extras else ""
        if self.urls:
            body = f"{self.name}{extras} @ {sorted(self.urls)[0]}"
        else:
            body = f"{self.name}{extras}{self.spec_set()}"
        if self.marker:
            body += f" ; {self.marker}"
        srcs = ", ".join(sorted(set(self.sources)))
        return f"{body}  # {srcs}" + (f" ({self.note})" if self.note else "")


@dataclass
class Report:
    sources: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    banned: list[str] = field(default_factory=list)
    replaced: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    options: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)


def find_requirement_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
        if "requirements.txt" in filenames:
            out.append(Path(dirpath) / "requirements.txt")
    platform_in = root / "deploy" / "platform.in"
    if platform_in.is_file():
        out.append(platform_in)
    return sorted(out)


def _logical_lines(text: str) -> list[tuple[int, str]]:
    lines, buf, start = [], "", 0
    for i, raw in enumerate(text.splitlines(), 1):
        line = re.sub(r"(^|\s)#.*$", "", raw).rstrip()
        if not buf:
            start = i
        if line.endswith("\\"):
            buf += line[:-1] + " "
            continue
        buf += line
        if buf.strip():
            lines.append((start, buf.strip()))
        buf = ""
    if buf.strip():
        lines.append((start, buf.strip()))
    return lines


def parse_file(path: Path, root: Path, report: Report, _seen: set[Path] | None = None) -> list[ReqLine]:
    seen = _seen if _seen is not None else set()
    path = path.resolve()
    if path in seen:
        return []
    seen.add(path)
    try:
        rel = path.relative_to(root.resolve()).as_posix()
    except ValueError:
        rel = path.as_posix()
    out: list[ReqLine] = []
    for lineno, line in _logical_lines(path.read_text(encoding="utf-8-sig")):
        where = f"{rel}:{lineno}"
        m = re.match(r"^(-r|--requirement|-c|--constraint)\s*=?\s*(\S+)$", line)
        if m:
            target = (path.parent / m.group(2)).resolve()
            if not target.is_file():
                report.warnings.append(f"{where}: не найден {m.group(2)}")
            elif m.group(1) in ("-r", "--requirement"):
                out += parse_file(target, root, report, seen)
            else:
                report.constraints.append(str(target))
            continue
        if line.startswith("-"):
            if line.startswith(("-e", "--editable")):
                report.warnings.append(f"{where}: editable пропущен в prod: {line}")
            elif "download.pytorch.org" in line:
                report.dropped.append(f"{where}: {line} (torch ставится через --torch-backend)")
            else:
                if line not in report.options:
                    report.options.append(line)
                report.warnings.append(f"{where}: опция индекса/сборки перенесена как есть: {line}")
            continue
        line = re.sub(r"\s--hash[=\s]\S+", "", line)
        try:
            req = Requirement(line)
        except InvalidRequirement as exc:
            report.warnings.append(f"{where}: не разобрано ({exc}): {line}")
            continue
        out.append(ReqLine(
            name=canonicalize_name(req.name), extras=frozenset(req.extras), specifier=str(req.specifier),
            marker=str(req.marker) if req.marker else "", url=req.url or "", source=rel, lineno=lineno, raw=line,
        ))
    return out


def _versions_in(spec: str) -> list[Version]:
    out = []
    for m in re.finditer(r"(?:===|==|!=|~=|>=|<=|>|<)\s*([^,;\s]+)", spec):
        v = m.group(1).rstrip(".*")
        try:
            out.append(Version(v))
        except InvalidVersion:
            pass
    return out


def _candidates(versions: list[Version]) -> list[Version]:
    """Пробные версии для проверки совместимости диапазонов (без индекса пакетов)."""
    cands: set[Version] = set()
    for v in versions:
        rel = list(v.release) + [0] * (3 - len(v.release))
        a, b, c = rel[:3]
        cands.add(v)
        for t in ((a, b, c + 1), (a, b + 1, 0), (a + 1, 0, 0), (a, b, c, 1)):
            cands.add(Version(".".join(map(str, t))))
        if c > 0:
            cands.add(Version(f"{a}.{b}.{c - 1}"))
        if b > 0:
            cands.add(Version(f"{a}.{b - 1}.9999"))
        if a > 0:
            cands.add(Version(f"{a - 1}.9999.9999"))
    if not cands:
        cands.add(Version("1.0"))
    return sorted(cands)


def satisfiable(specs: list[str]) -> bool:
    joined = ",".join(s for s in specs if s)
    if not joined:
        return True
    ss = SpecifierSet(joined)
    return any(ss.contains(v, prereleases=True) for v in _candidates(_versions_in(joined)))


def _is_dev(name: str) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in DEV_PATTERNS)


def _family_target(names: set[str], family: set[str], profile: str) -> str | None:
    present = names & family
    if not present:
        return None
    if family is OPENCV:
        return "opencv-contrib-python-headless" if any("contrib" in n for n in present) else "opencv-python-headless"
    return "onnxruntime-gpu" if profile == "gpu" else "onnxruntime"


def merge(lines: list[ReqLine], profile: str, report: Report, keep_dev: bool = False) -> list[Merged]:
    names = {ln.name for ln in lines}
    rename: dict[str, str] = {}
    for family in (OPENCV, ONNXRUNTIME):
        target = _family_target(names, family, profile)
        if target:
            for n in sorted(names & family):
                if n != target:
                    rename[n] = target
                    report.replaced.append(f"{n} → {target} (взаимоисключающие дистрибутивы, профиль {profile})")
    groups: dict[tuple[str, str], Merged] = {}
    for ln in lines:
        name = rename.get(ln.name, ln.name)
        if name in BANNED_PACKAGES:
            report.banned.append(f"{name} ({BANNED_PACKAGES[name]}) — {ln.source}:{ln.lineno}")
            continue
        if not keep_dev and _is_dev(name):
            report.dropped.append(f"{name} — dev-инструмент ({ln.source})")
            continue
        if name in KNOWN_TROUBLE:
            msg = f"{name}: {KNOWN_TROUBLE[name]} ({ln.source})"
            if msg not in report.warnings:
                report.warnings.append(msg)
        g = groups.setdefault((name, ln.marker), Merged(name=name, marker=ln.marker))
        g.extras |= set(ln.extras)
        g.sources.append(ln.source)
        if ln.url:
            g.urls.add(ln.url)
        if ln.specifier:
            g.specs.append((ln.specifier, f"{ln.source}:{ln.lineno}"))
    for g in groups.values():
        if len(g.urls) > 1:
            report.conflicts.append(f"{g.name}: разные прямые ссылки {sorted(g.urls)}")
        if g.urls and g.specs:
            report.conflicts.append(f"{g.name}: прямая ссылка и версия одновременно {g.specs}")
        if not satisfiable([s for s, _ in g.specs]):
            detail = "; ".join(f"{s} ← {src}" for s, src in g.specs)
            report.conflicts.append(f"{g.name}{' ; ' + g.marker if g.marker else ''}: несовместимо: {detail}")
            g.note = "CONFLICT"
    return [groups[k] for k in sorted(groups)]


def render_in(merged: list[Merged], report: Report, profile: str) -> str:
    head = [
        "# AUTO-GENERATED: deploy/merge_requirements.py — не редактировать руками.",
        f"# Профиль: {profile}; источников: {len(report.sources)}; конфликтов: {len(report.conflicts)}.",
        "# Точные версии и хэши — ТОЛЬКО в deploy/locks/<target>.lock.txt (uv pip compile).",
    ]
    head += [f"# источник: {s}" for s in report.sources]
    head += [f"# замена: {r}" for r in report.replaced]
    head += [f"# исключено: {d}" for d in report.dropped]
    head += [f"# КОНФЛИКТ: {c}" for c in report.conflicts]
    body = list(report.options) + [m.line() for m in merged]
    return "\n".join(head + [""] + body) + "\n"


def print_report(report: Report, stream=sys.stderr) -> None:
    def block(title: str, items: list[str]) -> None:
        if items:
            print(f"[merge] {title} ({len(items)}):", file=stream)
            for it in items:
                print(f"    - {it}", file=stream)
    print(f"[merge] источников: {len(report.sources)}", file=stream)
    block("ЗАПРЕЩЁННЫЕ ЛИЦЕНЗИИ", report.banned)
    block("КОНФЛИКТЫ (нужно решение владельцев модулей / A01)", report.conflicts)
    block("замены дистрибутивов", report.replaced)
    block("исключено", report.dropped)
    block("предупреждения", report.warnings)


def lock(in_path: Path, target: str, out_dir: Path, constraints: list[str], uv: str | None = None) -> Path:
    """Настоящий resolver: uv pip compile под платформу цели. Без uv — ошибка, а не «угадывание»."""
    cfg = LOCK_TARGETS[target]
    uv = uv or shutil.which("uv")
    if not uv:
        raise RuntimeError("uv не найден: установите https://docs.astral.sh/uv/ (setup.ps1 ставит его сам)")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{target}.lock.txt"
    cmd = [uv, "pip", "compile", str(in_path), "-o", str(out), "--python-version", "3.11",
           "--python-platform", cfg["python_platform"], "--torch-backend", cfg["torch_backend"],
           "--generate-hashes", "--emit-index-annotation", "--quiet"]
    for c in constraints:
        cmd += ["-c", c]
    print("[merge] resolver:", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True)
    return out


def run(root: Path, profile: str, out: Path | None, check_only: bool = False, keep_dev: bool = False,
        allow_conflicts: bool = False) -> tuple[int, Report, str]:
    report = Report()
    files = find_requirement_files(root)
    lines: list[ReqLine] = []
    for f in files:
        report.sources.append(f.resolve().relative_to(root.resolve()).as_posix())
        lines += parse_file(f, root, report)
    merged = merge(lines, profile, report, keep_dev=keep_dev)
    text = render_in(merged, report, profile)
    code = 2 if report.banned else (1 if report.conflicts else 0)
    if out is not None and not check_only and (code == 0 or (code == 1 and allow_conflicts)):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    return code, report, text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=REPO_ROOT)
    ap.add_argument("--profile", choices=("gpu", "cpu"), default=None,
                    help="по умолчанию — из --lock цели, иначе gpu")
    ap.add_argument("--out", type=Path, default=None, help="по умолчанию <root>/deploy/requirements.in")
    ap.add_argument("--check", action="store_true", help="только отчёт, ничего не писать")
    ap.add_argument("--keep-dev", action="store_true", help="не выкидывать pytest/ruff/...")
    ap.add_argument("--allow-conflicts", action="store_true",
                    help="всё равно записать .in (конфликт увидит resolver) — для диагностики")
    ap.add_argument("--lock", choices=sorted(LOCK_TARGETS), action="append", default=[],
                    help="построить lock через uv pip compile (можно несколько раз)")
    ap.add_argument("--locks-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    root = args.root.resolve()
    profile = args.profile or (LOCK_TARGETS[args.lock[0]]["profile"] if args.lock else "gpu")
    out = args.out or root / "deploy" / "requirements.in"
    code, report, _ = run(root, profile, out, args.check, args.keep_dev, args.allow_conflicts)
    print_report(report)
    if code:
        print(f"[merge] код {code}: {'запрещённые пакеты' if code == 2 else 'конфликты'} — lock не строим",
              file=sys.stderr)
        return code
    if not args.check:
        print(f"[merge] записано: {out}", file=sys.stderr)
    constraints = list(report.constraints)
    agreed = root / "deploy" / "constraints.txt"
    if agreed.is_file():
        constraints.append(str(agreed))
    for target in args.lock:
        if LOCK_TARGETS[target]["profile"] != profile:
            print(f"[merge] пропуск {target}: профиль {LOCK_TARGETS[target]['profile']} != {profile}",
                  file=sys.stderr)
            continue
        try:
            path = lock(out, target, args.locks_dir or root / "deploy" / "locks", constraints)
        except (RuntimeError, subprocess.CalledProcessError) as exc:
            print(f"[merge] resolver не смог построить {target}: {exc}", file=sys.stderr)
            return 3
        print(f"[merge] lock: {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
