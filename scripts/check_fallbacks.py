#!/usr/bin/env python3
"""Сверка fallback-копий контракта (``<модуль>/_contract_fallback.py``) с каноном (владелец A01).

После мерджа A01 модули импортируют ``contracts.sentinel_contracts``, а не свою копию. Если копия разошлась
с каноном (добавлены/изменены поля), модуль мог опираться на другой контракт — это ошибка (код 1).

    python scripts/check_fallbacks.py            # отчёт
    python scripts/check_fallbacks.py --drop     # (интегратор, после мерджа A01) удалить совпадающие копии

Удалять безопасно: ветка ``except ImportError: from ._contract_fallback import *`` при наличии ``contracts``
не выполняется. Оставшиеся упоминания выводятся для ручной чистки.
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANON = Path("contracts") / "sentinel_contracts.py"
SKIP = {".git", ".venv", "venv", "node_modules", "data", "weights", "cache"}


def structure(src: str) -> list:
    """Подпись контракта без комментариев/докстрингов: классы, поля, дефолты, методы, константы."""
    out: list = []
    for n in ast.parse(src).body:
        if isinstance(n, ast.ClassDef):
            members = []
            for b in n.body:
                if isinstance(b, ast.AnnAssign):
                    members.append((ast.unparse(b.target), ast.unparse(b.annotation),
                                    ast.unparse(b.value) if b.value else None))
                elif isinstance(b, ast.FunctionDef):
                    members.append(("def " + b.name, ast.unparse(b.args), None))
            out.append(("class", n.name, [ast.unparse(d) for d in n.decorator_list], tuple(members)))
        elif isinstance(n, (ast.Assign, ast.AnnAssign)):
            out.append(("assign", ast.unparse(n)))
    return out


def find_fallbacks(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("_contract_fallback.py")
                  if not any(part in SKIP for part in p.relative_to(root).parts))


def compare(path: Path, canon_src: str) -> tuple[str, str]:
    src = path.read_text(encoding="utf-8")
    if src.replace("\r\n", "\n") == canon_src.replace("\r\n", "\n"):
        return "identical", ""
    try:
        a, b = structure(src), structure(canon_src)
    except SyntaxError as e:
        return "DIVERGED", f"не парсится: {e}"
    if a == b:
        return "same-structure", "отличаются только комментарии/форматирование"
    extra = [x[1] for x in a if x not in b]
    missing = [x[1] for x in b if x not in a]
    return "DIVERGED", f"лишнее/изменённое: {extra}; нет/изменено в каноне: {missing}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Сверка fallback-копий контракта с каноном")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--drop", action="store_true", help="удалить копии, совпадающие с каноном")
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()
    canon_path = root / CANON
    if not canon_path.is_file():
        print(f"нет канона {CANON} — нечего сверять")
        return 0
    canon = canon_path.read_text(encoding="utf-8")
    rc = 0
    for p in find_fallbacks(root):
        status, why = compare(p, canon)
        rel = p.relative_to(root).as_posix()
        print(f"{status:15} {rel}" + (f" — {why}" if why else ""))
        if status == "DIVERGED":
            rc = 1
        elif args.drop:
            p.unlink()
            print(f"{'':15} удалён {rel}")
    if args.drop:
        refs = [f"{q.relative_to(root).as_posix()}:{i}" for q in root.rglob("*.py")
                if not any(part in SKIP for part in q.relative_to(root).parts)
                for i, line in enumerate(q.read_text(encoding="utf-8", errors="ignore").splitlines(), 1)
                if "_contract_fallback" in line]
        if refs:
            print("Оставшиеся упоминания (except ImportError-ветки, можно упростить вручную):")
            for r in refs:
                print("   ", r)
    return rc


if __name__ == "__main__":
    sys.exit(main())
