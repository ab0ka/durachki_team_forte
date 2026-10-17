#!/usr/bin/env python3
"""Какие файлы ветки тронуты НЕ владельцем (по таблице владения из задания). Владелец: A01.

Примеры::

    # текущая ветка против main; агент из заголовка PR или явно
    python scripts/check_ownership.py --base origin/main --agent A05
    python scripts/check_ownership.py --base origin/main --title "[A05] Детекция"
    # конкретная ветка
    python scripts/check_ownership.py --base origin/main --head origin/claude/xyz
    # отчёт по всем удалённым веткам (для интегратора)
    python scripts/check_ownership.py --base origin/main --all-branches

Код возврата 1 — есть чужие файлы при явно заданном агенте (``--agent``/``--title``) или запрещённые файлы
(видео/веса/кадры). Если агент только угадан по большинству файлов — отчёт информационный (код 0).
Ветки с несвязанной историей (у каждого агента свой initial commit) сравниваются по дереву (two-dot).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ownership import agent_from_title, forbidden_reason, is_shared, owner_of  # noqa: E402


def _git(*args: str) -> tuple[int, str]:
    p = subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8")
    return p.returncode, p.stdout


def changed_files(base: str, head: str) -> tuple[list[str], str]:
    """Список изменённых файлов head относительно base и режим сравнения."""
    rc, out = _git("diff", "--name-only", f"{base}...{head}")
    if rc == 0:
        return [line for line in out.splitlines() if line], "merge-base"
    # нет общей истории — сравниваем деревья, удаления не считаем (их «делает» несвязанная история)
    rc, out = _git("diff", "--name-only", "--diff-filter=ACMRT", base, head)
    if rc != 0:
        raise SystemExit(f"git diff не удался для {base} .. {head}")
    return [line for line in out.splitlines() if line], "tree (несвязанные истории)"


def analyze(files: list[str], agent: str | None) -> dict:
    owners = Counter(owner_of(f) or "?" for f in files if not is_shared(f))
    inferred = None
    if agent is None and owners:
        inferred = owners.most_common(1)[0][0]
    who = agent or inferred
    foreign = [(f, owner_of(f) or "ничей") for f in files if not is_shared(f) and owner_of(f) != who]
    forbidden = [(f, forbidden_reason(f)) for f in files if forbidden_reason(f)]
    return {"agent": who, "explicit": agent is not None, "owners": dict(owners), "foreign": foreign,
            "forbidden": forbidden, "n_files": len(files)}


def report(name: str, res: dict, mode: str) -> int:
    tag = "явно" if res["explicit"] else "угадан"
    print(f"== {name}: агент {res['agent']} ({tag}), файлов {res['n_files']}, сравнение: {mode}")
    for f, why in res["forbidden"]:
        print(f"   FORBIDDEN {f}: {why}")
    for f, owner in res["foreign"]:
        print(f"   FOREIGN   {f} (владелец {owner})")
    if not res["foreign"] and not res["forbidden"]:
        print("   OK — только свои файлы")
    bad = bool(res["forbidden"]) or (res["explicit"] and bool(res["foreign"]))
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", default="origin/main")
    ap.add_argument("--head", default="HEAD")
    ap.add_argument("--agent", help="A01..A18 (иначе из --title или по большинству файлов)")
    ap.add_argument("--title", help="заголовок PR вида '[A05] ...'")
    ap.add_argument("--files", nargs="*", help="проверить явный список файлов вместо git diff")
    ap.add_argument("--all-branches", action="store_true", help="все удалённые ветки origin/* кроме base")
    ap.add_argument("--report-only", action="store_true", help="всегда код 0")
    args = ap.parse_args(argv)
    agent = (args.agent or agent_from_title(args.title) or "").upper() or None

    rc = 0
    if args.files is not None:
        rc = report("--files", analyze(args.files, agent), "список")
    elif args.all_branches:
        _, out = _git("for-each-ref", "--format=%(refname:short)", "refs/remotes/origin")
        for ref in sorted(out.split()):
            if ref in (args.base, "origin/HEAD", "origin"):
                continue
            files, mode = changed_files(args.base, ref)
            rc |= report(ref, analyze(files, None), mode)
    else:
        files, mode = changed_files(args.base, args.head)
        rc = report(args.head, analyze(files, agent), mode)
    return 0 if args.report_only else rc


if __name__ == "__main__":
    sys.exit(main())
