#!/usr/bin/env python3
"""Матрица совместимости веток агентов с контрактом (read-only; владелец A01) -> contracts/COMPAT.md.

    python scripts/compat_matrix.py --tree <дерево всех веток> [--out contracts/COMPAT.md]

``--tree`` — каталог, где поверх ветки A01 разложены деревья всех веток ``origin/claude/*``
(агенты пишут в разные папки, так что это эквивалент мерджа). Собрать его::

    for ref in $(git for-each-ref --format='%(refname:short)' refs/remotes/origin/claude); do
        git archive "$ref" | tar -x -C /tmp/integ; done; git archive HEAD | tar -x -C /tmp/integ

Что проверяется:
1. публичные функции модулей из контракта: есть ли символ, совместима ли сигнатура с позиционным вызовом
   оркестратора (A03 вызывает функции стадий позиционно через getattr), совпадают ли имена/умолчания;
2. статически: прямые (без try/except) импорты ``_contract_fallback``, дубли классов контракта,
   сигналы вне ``SIGNALS``/``EXTRA_SIGNALS``, «неизвестно = 0» (``signals.get(x, 0)``), потребители сериализатора,
   производители ``meta.coverage``.
Импорт модулей — в отдельных процессах с ``SENTINEL_MOCK=1`` (без весов).
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contracts.extensions import EXTRA_SIGNALS  # noqa: E402
from contracts.sentinel_contracts import SIGNALS  # noqa: E402

# (владелец, модуль, функция, ожидаемые параметры, ожидаемые умолчания, на критическом пути)
EXPECTED: list[tuple[str, str, str, list[str], dict[str, Any], bool]] = [
    ("A04", "ml.ingest", "load_video", ["path", "sample_fps"], {"sample_fps": 5.0}, True),
    ("A05", "ml.detect", "detect", ["clip", "every_n"], {"every_n": 1}, True),
    ("A06", "ml.track", "track", ["clip", "dets"], {}, True),
    ("A07", "ml.pose", "estimate", ["clip", "tracks"], {}, False),
    ("A07", "ml.pose", "pose_signals", ["poses", "tracks", "zones"], {}, False),
    ("A08", "ml.scene", "scene_states", ["clip", "dets", "tracks", "zones", "pose_states"], {}, True),
    ("A09", "ml.rules", "run_rules", ["states", "tracks", "zones", "catalog"], {}, True),
    ("A10", "ml.temporal", "score", ["clip"], {}, False),
    ("A11", "ml.vlm", "verify", ["clip", "events", "budget_s"], {}, False),
    ("A11", "ml.vlm", "discover", ["clip", "budget_s"], {}, False),
    ("A12", "ml.fusion", "fuse", ["clip", "events", "risk", "catalog", "job_id"], {}, True),
    ("A13", "norms", "load_catalog", [], {}, True),
    ("A03", "backend.pipeline", "run", ["video_path", "job_id", "progress", "deadline_s"], {"deadline_s": 240}, True),
    ("A03", "backend.pipeline", "warmup", [], {}, False),
]
CONTRACT_CLASSES = {"Detection", "Track", "Zone", "PoseFrame", "SceneState", "Event", "EvidenceFrame", "RuleRef",
                    "Violation", "AnalysisResult"}
NOTES_BEGIN = "<!-- notes:begin (ручной раздел, сохраняется при перегенерации) -->"
NOTES_END = "<!-- notes:end -->"
SKIP_DIRS = {"contracts", "testkit", "scripts", ".git", "node_modules", "frontend", "docs", "pitch", "__pycache__"}

_PROBE = r"""
import importlib, inspect, json, sys
mod, func = sys.argv[1], sys.argv[2]
out = {"ok": False}
try:
    m = importlib.import_module(mod)
    fn = getattr(m, func, None)
    if fn is None:
        out["error"] = f"нет символа {mod}.{func}"
    else:
        sig = inspect.signature(fn)
        out["ok"] = True
        out["params"] = [[p.name, p.kind.name, None if p.default is inspect.Parameter.empty else repr(p.default)]
                         for p in sig.parameters.values()]
        out["qualname"] = f"{getattr(fn, '__module__', mod)}.{getattr(fn, '__qualname__', func)}"
except Exception as e:  # noqa: BLE001
    out["error"] = f"{type(e).__name__}: {e}"[:300]
print(json.dumps(out, ensure_ascii=False))
"""


@dataclass
class Row:
    owner: str
    symbol: str
    expected: str
    actual: str
    verdict: str          # OK | ЗАМЕЧАНИЕ | РАСХОЖДЕНИЕ | НЕТ
    notes: list[str] = field(default_factory=list)
    critical: bool = False


def probe(tree: Path, module: str, func: str, timeout: float = 60.0) -> dict:
    env = dict(os.environ, SENTINEL_MOCK="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONPATH=str(tree))
    try:
        p = subprocess.run([sys.executable, "-c", _PROBE, module, func], cwd=tree, env=env, capture_output=True,
                           text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"импорт дольше {timeout:.0f} с"}
    line = (p.stdout.strip().splitlines() or ["{}"])[-1]
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return {"ok": False, "error": (p.stderr.strip().splitlines() or ["нет вывода"])[-1][:300]}


def compare(owner: str, module: str, func: str, exp: list[str], exp_def: dict, critical: bool, res: dict) -> Row:
    expected = f"{module}.{func}({', '.join(exp)})"
    if not res.get("ok"):
        return Row(owner, f"{module}.{func}", expected, res.get("error", "?"), "НЕТ", critical=critical)
    params = res["params"]
    actual = f"{res.get('qualname', func)}({', '.join(n + ('=' + d if d is not None else '') for n, _, d in params)})"
    notes: list[str] = []
    positional = [p for p in params if p[1] in ("POSITIONAL_ONLY", "POSITIONAL_OR_KEYWORD")]
    has_var = any(p[1] == "VAR_POSITIONAL" for p in params)
    verdict = "OK"
    if len(positional) < len(exp) and not has_var:
        verdict = "РАСХОЖДЕНИЕ"
        notes.append(f"позиционно принимает {len(positional)} арг., ожидается {len(exp)}")
    extra_required = [p[0] for p in positional[len(exp):] if p[2] is None] + \
        [p[0] for p in params if p[1] == "KEYWORD_ONLY" and p[2] is None]
    if extra_required:
        verdict = "РАСХОЖДЕНИЕ"
        notes.append(f"лишние обязательные параметры: {extra_required}")
    names = [p[0] for p in positional[:len(exp)]]
    if names != exp[:len(names)]:
        if verdict == "OK":
            verdict = "ЗАМЕЧАНИЕ"
        notes.append(f"имена {names} ≠ {exp} (позиционный вызов работает, вызов по имени — нет)")
    for k, v in exp_def.items():
        got = next((p[2] for p in params if p[0] == k), None)
        if got is None:
            if verdict == "OK":
                verdict = "ЗАМЕЧАНИЕ"
            notes.append(f"нет умолчания {k}={v!r}")
        elif got != repr(v) and str(got) != str(v):
            if verdict == "OK":
                verdict = "ЗАМЕЧАНИЕ"
            notes.append(f"умолчание {k}={got}, по контракту {v!r}")
    return Row(owner, f"{module}.{func}", expected, actual, verdict, notes, critical)


# ----------------------------------------------------------------------------- статические проверки

def _py_files(tree: Path):
    for p in tree.rglob("*.py"):
        rel = p.relative_to(tree)
        if not any(part in SKIP_DIRS for part in rel.parts):
            yield p, rel.as_posix()


def static_scan(tree: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {"unguarded_fallback": [], "duplicate_classes": [], "unknown_signals": [],
                                 "unknown_as_zero": [], "serializer_calls": [], "coverage_producers": []}
    known = set(SIGNALS) | set(EXTRA_SIGNALS)
    sig_re = re.compile(r"""(?:signals|sig|sigs)\s*(?:\[\s*|\.get\(\s*)["']([a-z_]+)["']""")
    zero_re = re.compile(r"""(?:signals|sig|sigs)\.get\(\s*["']([a-z_]+)["']\s*,\s*0(?:\.0)?\s*\)""")
    for path, rel in _py_files(tree):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if rel.endswith("_contract_fallback.py"):
            continue
        try:
            t = ast.parse(text)
        except SyntaxError:
            continue
        guarded = {id(m) for n in ast.walk(t) if isinstance(n, ast.Try) for h in n.handlers for m in ast.walk(h)}
        for n in ast.walk(t):
            if isinstance(n, ast.ImportFrom) and n.module and "_contract_fallback" in n.module and id(n) not in guarded:
                out["unguarded_fallback"].append(f"{rel}:{n.lineno}")
            if isinstance(n, ast.ClassDef) and n.name in CONTRACT_CLASSES:
                out["duplicate_classes"].append(f"{rel}:{n.lineno} class {n.name}")
        for i, line in enumerate(text.splitlines(), 1):
            for m in sig_re.finditer(line):
                if m.group(1) not in known:
                    out["unknown_signals"].append(f"{rel}:{i} `{m.group(1)}`")
            for m in zero_re.finditer(line):
                out["unknown_as_zero"].append(f"{rel}:{i} `{m.group(1)}`")
            if "result_to_api(" in line or "result_to_jury_api(" in line or "result_to_ui_report(" in line:
                if "def " not in line:
                    out["serializer_calls"].append(f"{rel}:{i}")
            if re.search(r"""meta\[\s*["']coverage["']\s*\]\s*=|["']coverage["']\s*:""", line):
                out["coverage_producers"].append(f"{rel}:{i}")
    return out


# ----------------------------------------------------------------------------- markdown

def render(rows: list[Row], scan: dict[str, list[str]], branches: list[dict], header: str) -> str:
    lines = [header, "", "## 1. Публичные функции модулей против контракта", "",
             "| Владелец | Ожидаемый вызов (контракт) | Фактический символ | Итог | Расхождение / заметка |",
             "|---|---|---|---|---|"]
    for r in rows:
        crit = " **(критический путь)**" if r.critical else ""
        lines.append(f"| {r.owner} | `{r.expected}`{crit} | `{r.actual}` | {r.verdict} | "
                     f"{'; '.join(r.notes) or '—'} |")
    lines += ["", "## 2. Статические проверки", ""]
    titles = {
        "unguarded_fallback": "Прямые импорты `_contract_fallback` без `try/except` (сломаются после удаления "
                              "копий; классы из копии не совпадут по идентичности с каноническими)",
        "duplicate_classes": "Свои классы с именами классов контракта (риск isinstance-расхождений)",
        "unknown_signals": "Сигналы вне `SIGNALS`/`EXTRA_SIGNALS` (обсудить добавление в словарь)",
        "unknown_as_zero": "«Неизвестно = 0»: `signals.get(x, 0)` (неизвестно ≠ нет — нужен явный None/пропуск)",
        "serializer_calls": "Вызовы сериализатора (ответ жюри должен идти через `result_to_jury_api`)",
        "coverage_producers": "Кто пишет `meta.coverage` (без него `no_violation` уходит жюри как `failed`)",
    }
    for key, title in titles.items():
        items = scan.get(key, [])
        lines.append(f"### {title}")
        lines.append("")
        if items:
            lines += [f"- `{x}`" if "`" not in x else f"- {x}" for x in items[:60]]
            if len(items) > 60:
                lines.append(f"- … ещё {len(items) - 60}")
        else:
            lines.append("- нет")
        lines.append("")
    if branches:
        lines += ["## 3. Ветки (состояние на момент генерации)", "",
                  "| Ветка | Агент | HEAD | Общая история с `main` |", "|---|---|---|---|"]
        for b in branches:
            lines.append(f"| `{b['ref']}` | {b['agent']} | `{b['sha']}` | {b['related']} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def branch_info() -> list[dict]:
    from ownership import owner_of  # scripts/ в sys.path ниже

    def git(*a: str) -> str:
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    out = []
    for ref in git("for-each-ref", "--format=%(refname:short)", "refs/remotes/origin/claude").split():
        files = git("ls-tree", "-r", "--name-only", ref).splitlines()
        owners: dict[str, int] = {}
        for f in files:
            o = owner_of(f)
            if o and o != "A01":
                owners[o] = owners.get(o, 0) + 1
        agent = "L2 (интеграция)" if "integration" in ref else max(owners, key=owners.get) if owners else "A01"
        related = subprocess.run(["git", "merge-base", "origin/main", ref], cwd=ROOT, capture_output=True).returncode
        out.append({"ref": ref, "agent": agent, "sha": git("rev-parse", "--short", ref),
                    "related": "да" if related == 0 else "нет (несвязанная история)"})
    return sorted(out, key=lambda b: b["agent"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Матрица совместимости веток с контрактом -> COMPAT.md")
    ap.add_argument("--tree", required=True, help="каталог с наложенными деревьями всех веток")
    ap.add_argument("--out", default=str(ROOT / "contracts" / "COMPAT.md"))
    ap.add_argument("--header", default="# Совместимость веток с контрактом v1")
    ap.add_argument("--json", default=None, help="дополнительно сохранить сырой результат")
    args = ap.parse_args(argv)
    tree = Path(args.tree).resolve()
    sys.path.insert(0, str(ROOT / "scripts"))
    rows = [compare(o, m, f, e, d, c, probe(tree, m, f)) for o, m, f, e, d, c in EXPECTED]
    scan = static_scan(tree)
    branches = branch_info()
    out_path = Path(args.out)
    notes = ""
    if out_path.is_file():  # ручной раздел между маркерами сохраняется при перегенерации
        old = out_path.read_text(encoding="utf-8")
        if NOTES_BEGIN in old and NOTES_END in old:
            notes = old[old.index(NOTES_BEGIN):old.index(NOTES_END) + len(NOTES_END)]
    body = render(rows, scan, branches, args.header)
    if notes:
        head, rest = body.split("\n", 1)
        body = f"{head}\n\n{notes}\n{rest}"
    out_path.write_text(body, encoding="utf-8")
    if args.json:
        Path(args.json).write_text(json.dumps({"rows": [r.__dict__ for r in rows], "scan": scan,
                                               "branches": branches}, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    bad = [r for r in rows if r.critical and r.verdict in ("НЕТ", "РАСХОЖДЕНИЕ")]
    for r in rows:
        print(f"{r.verdict:12} {r.symbol}: {'; '.join(r.notes) or r.actual}")
    print(f"критических несовместимостей: {len(bad)}; отчёт: {args.out}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
