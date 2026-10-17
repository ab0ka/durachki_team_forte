"""Валидатор каталога норм: схема + анти-галлюцинационная проверка цитат.

    python -m norms.validate                 # проверить norms/catalog.yaml
    python -m norms.validate --strict        # предупреждения = ошибки (verified:false и т.п.)
    python -m norms.validate --json          # машиночитаемый отчёт
    python -m norms.validate --write-fallback  # пересобрать norms/_fallback.py из YAML

Что проверяется:
  1. Схема: version, все типы V01..V08 и V99, поля name / observable / not_observable /
     description_template ({t} и {facts}) / severity / rules[].
  2. Каждая норма — ровно ключи RuleRef (document, clause, url, quote, verified).
  3. Каждая норма ссылается (по url) на документ из раздела ``documents`` с файлом в sources/.
  4. КАЖДАЯ цитата дословно содержится в соответствующем norms/sources/*.md
     (сравнение после нормализации пробелов/кавычек/тире — не смысла!).
  5. ``verified: true`` допустим только если пункт указан и файл-источник — полный текст
     (front matter ``retrieval: fulltext``), а не поисковые фрагменты.
  6. Встроенный fallback (norms/_fallback.py) совпадает с YAML.
"""
from __future__ import annotations

import argparse
import json
import pprint
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from . import CATALOG_PATH, NORMS_DIR, RULE_KEYS, SEVERITIES, TYPE_IDS

RETRIEVAL_KINDS = ("fulltext", "partial", "search_snippets")
_TYPE_ID_RE = re.compile(r"^V\d{2}$")
_FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)

_TRANSLATE = str.maketrans({
    " ": " ", " ": " ", " ": " ", " ": " ",  # неразрывные/узкие пробелы
    "­": None, "​": None, "﻿": None,              # мягкий перенос, zero-width
    "–": "-", "—": "-", "‒": "-", "−": "-", "‑": "-", "‐": "-",
    "«": '"', "»": '"', "“": '"', "”": '"', "„": '"', "″": '"',
    "‘": "'", "’": "'", "‚": "'", "′": "'",
})


def normalize(text: str) -> str:
    """Нормализация ТОЛЬКО типографики: пробелы, кавычки, тире, регистр. Слова не трогаем."""
    text = text.translate(_TRANSLATE)
    text = re.sub(r"\s+", " ", text)
    return text.strip().casefold()


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Простой front matter ``key: value`` между строками ``---`` (без вложенности)."""
    m = _FRONT_MATTER_RE.match(text)
    if not m:
        return {}, text
    meta: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" in line and not line.lstrip().startswith("#"):
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().strip('"').strip("'")
    return meta, text[m.end():]


@dataclass
class QuoteCheck:
    type_id: str
    index: int
    document: str
    clause: Optional[str]
    source_file: str
    retrieval: str
    verified: bool
    status: str  # OK | NOT_FOUND | NO_SOURCE | EMPTY


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    quotes: list[QuoteCheck] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "errors": self.errors, "warnings": self.warnings,
                "quotes": [asdict(q) for q in self.quotes]}


def _norm_url(url: str) -> str:
    return str(url).strip().rstrip("/").replace("http://", "https://").replace("//www.", "//")


def _is_str(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())


def _load_sources(catalog: dict[str, Any], base_dir: Path, rep: Report) -> dict[str, dict[str, Any]]:
    """url документа → {id, file, meta, body_norm}."""
    out: dict[str, dict[str, Any]] = {}
    docs = catalog.get("documents")
    if not isinstance(docs, dict) or not docs:
        rep.errors.append("documents: нет раздела documents (реестр источников с файлами в sources/)")
        return out
    for doc_id, doc in docs.items():
        where = f"documents.{doc_id}"
        if not isinstance(doc, dict):
            rep.errors.append(f"{where}: должен быть словарём")
            continue
        for k in ("title", "url", "source_file"):
            if not _is_str(doc.get(k)):
                rep.errors.append(f"{where}.{k}: обязательное непустое поле")
        if not _is_str(doc.get("url")) or not _is_str(doc.get("source_file")):
            continue
        path = base_dir / doc["source_file"]
        entry: dict[str, Any] = {"id": doc_id, "file": doc["source_file"], "meta": {}, "body": None}
        if not path.is_file():
            rep.errors.append(f"{where}.source_file: файл {doc['source_file']} не найден")
        else:
            meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
            entry["meta"], entry["body"] = meta, normalize(body)
            if not meta:
                rep.errors.append(f"{doc['source_file']}: нет front matter (url, title, accessed, retrieval)")
            else:
                if _norm_url(meta.get("url", "")) != _norm_url(doc["url"]):
                    rep.errors.append(f"{doc['source_file']}: url в шапке ({meta.get('url')}) "
                                      f"не совпадает с documents.{doc_id}.url")
                if meta.get("retrieval") not in RETRIEVAL_KINDS:
                    rep.errors.append(f"{doc['source_file']}: retrieval должен быть одним из {RETRIEVAL_KINDS}")
                if not meta.get("accessed"):
                    rep.warnings.append(f"{doc['source_file']}: не указана дата доступа (accessed)")
        if "adilet.zan.kz" not in doc["url"]:
            rep.warnings.append(f"{where}.url: не adilet.zan.kz ({doc['url']}) — предпочтительна ссылка на adilet")
        out[_norm_url(doc["url"])] = entry
    return out


def _check_rule(type_id: str, i: int, rule: Any, sources: dict[str, dict[str, Any]], rep: Report) -> None:
    where = f"types.{type_id}.rules[{i}]"
    if not isinstance(rule, dict):
        rep.errors.append(f"{where}: должен быть словарём")
        return
    keys = set(rule)
    missing, extra = set(RULE_KEYS) - keys, keys - set(RULE_KEYS)
    if missing:
        rep.errors.append(f"{where}: нет ключей {sorted(missing)}")
    if extra:
        rep.errors.append(f"{where}: лишние ключи {sorted(extra)} (RuleRef(**rule) упадёт) — "
                          f"комментарии держите в YAML-комментариях или в documents")
    if not _is_str(rule.get("document")):
        rep.errors.append(f"{where}.document: обязательное непустое поле")
    clause = rule.get("clause")
    if clause is not None and not _is_str(clause):
        rep.errors.append(f"{where}.clause: строка или null")
    url = rule.get("url")
    if not _is_str(url) or not str(url).startswith("https://"):
        rep.errors.append(f"{where}.url: нужен https-URL")
    quote = rule.get("quote")
    if not isinstance(quote, str):
        rep.errors.append(f"{where}.quote: строка (допустима пустая только при clause: null)")
        quote = ""
    verified = rule.get("verified")
    if not isinstance(verified, bool):
        rep.errors.append(f"{where}.verified: true/false")
        verified = False
    src = sources.get(_norm_url(url)) if _is_str(url) else None
    if src is None:
        rep.errors.append(f"{where}: url {url} не найден в разделе documents")
        status, retrieval, file = "NO_SOURCE", "", ""
    else:
        retrieval, file = src["meta"].get("retrieval", ""), src["file"]
        if not quote.strip():
            status = "EMPTY"
            if clause is not None:
                rep.errors.append(f"{where}: указан пункт {clause!r}, но нет цитаты")
            else:
                rep.warnings.append(f"{where}: норма без пункта и цитаты (косвенная ссылка на документ)")
        elif src["body"] is None:
            status = "NO_SOURCE"
        elif normalize(quote) in src["body"]:
            status = "OK"
        else:
            status = "NOT_FOUND"
            rep.errors.append(f"{where}: цитата НЕ найдена дословно в {file}: {quote[:90]!r}…")
    if verified:
        if clause is None:
            rep.errors.append(f"{where}: verified=true без номера пункта")
        if status != "OK":
            rep.errors.append(f"{where}: verified=true, но цитата не подтверждена источником ({status})")
        elif retrieval != "fulltext":
            rep.errors.append(f"{where}: verified=true допустим только при сверке с полным текстом "
                              f"(retrieval: fulltext), а {file} — {retrieval!r}")
    else:
        rep.warnings.append(f"{where}: verified=false — нужна сверка человеком с adilet.zan.kz")
    rep.quotes.append(QuoteCheck(type_id, i, str(rule.get("document", "")), clause, file,
                                 retrieval, bool(verified), status))


def _check_type(type_id: str, t: Any, sources: dict[str, dict[str, Any]], rep: Report) -> None:
    where = f"types.{type_id}"
    if not _TYPE_ID_RE.match(str(type_id)):
        rep.errors.append(f"{where}: ключ типа должен иметь вид V01..V99")
    if not isinstance(t, dict):
        rep.errors.append(f"{where}: должен быть словарём")
        return
    if not _is_str(t.get("name")):
        rep.errors.append(f"{where}.name: обязательное непустое поле")
    for k in ("observable", "not_observable"):
        v = t.get(k)
        if not isinstance(v, list) or not v or not all(_is_str(x) for x in v):
            rep.errors.append(f"{where}.{k}: непустой список строк")
    tpl = t.get("description_template")
    if not _is_str(tpl):
        rep.errors.append(f"{where}.description_template: обязательное поле")
    else:
        for ph in ("{t}", "{facts}"):
            if ph not in tpl:
                rep.errors.append(f"{where}.description_template: нет плейсхолдера {ph}")
        try:
            tpl.format(t="00:22", facts="факт")
        except (KeyError, IndexError, ValueError) as exc:
            rep.errors.append(f"{where}.description_template: недопустимый плейсхолдер ({exc})")
    if t.get("severity") not in SEVERITIES:
        rep.errors.append(f"{where}.severity: одно из {SEVERITIES}")
    rules = t.get("rules")
    if not isinstance(rules, list):
        rep.errors.append(f"{where}.rules: список (может быть пустым)")
        return
    if not rules:
        rep.warnings.append(f"{where}.rules: нет ни одной нормы")
    for i, r in enumerate(rules):
        _check_rule(type_id, i, r, sources, rep)


def validate_catalog(catalog: Any, base_dir: Path = NORMS_DIR, *, strict: bool = False,
                     required_types: tuple[str, ...] = TYPE_IDS) -> Report:
    """Проверяет каталог; base_dir — папка, относительно которой лежат source_file."""
    rep = Report()
    if not isinstance(catalog, dict):
        rep.errors.append("каталог должен быть словарём")
        return rep
    if catalog.get("version") != 1:
        rep.errors.append("version: ожидается 1")
    types = catalog.get("types")
    if not isinstance(types, dict):
        rep.errors.append("types: обязательный словарь")
        return rep
    for tid in required_types:
        if tid not in types:
            rep.errors.append(f"types.{tid}: обязательный тип отсутствует")
    sources = _load_sources(catalog, Path(base_dir), rep)
    for tid, t in types.items():
        _check_type(str(tid), t, sources, rep)
    if strict and rep.warnings:
        rep.errors.extend(f"[strict] {w}" for w in rep.warnings)
        rep.warnings = []
    return rep


def check_fallback(catalog: dict[str, Any], rep: Report) -> None:
    try:
        from ._fallback import CATALOG as FALLBACK
    except ImportError as exc:  # pragma: no cover
        rep.errors.append(f"_fallback.py не импортируется: {exc}")
        return
    if FALLBACK != catalog:
        rep.errors.append("norms/_fallback.py расходится с catalog.yaml — "
                          "запустите `python -m norms.validate --write-fallback`")


def write_fallback(catalog: dict[str, Any], path: Path = NORMS_DIR / "_fallback.py") -> None:
    body = pprint.pformat(catalog, indent=1, width=110, sort_dicts=False)
    path.write_text(
        '"""АВТОГЕНЕРАЦИЯ из norms/catalog.yaml — не редактировать руками.\n\n'
        "Встроенный fallback для norms.load_catalog(), если YAML/PyYAML недоступны.\n"
        'Пересобрать: python -m norms.validate --write-fallback\n"""\n'
        "from typing import Any\n\n"
        f"CATALOG: dict[str, Any] = {body}\n",
        encoding="utf-8",
    )


def _load_yaml_strict(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _print_human(rep: Report, path: Path) -> None:
    print(f"Каталог: {path}")
    print(f"{'тип':<4} {'#':<2} {'цитата':<10} {'verified':<8} {'источник':<14} документ / пункт")
    for q in rep.quotes:
        doc = (q.document[:58] + "…") if len(q.document) > 59 else q.document
        print(f"{q.type_id:<4} {q.index:<2} {q.status:<10} {str(q.verified).lower():<8} "
              f"{q.retrieval or '-':<14} {doc} / {q.clause or '—'}")
    n_ok = sum(q.status == "OK" for q in rep.quotes)
    n_quotes = sum(q.status != "EMPTY" for q in rep.quotes)
    print(f"\nЦитат найдено дословно: {n_ok}/{n_quotes}; норм всего: {len(rep.quotes)}; "
          f"verified=true: {sum(q.verified for q in rep.quotes)}")
    for w in rep.warnings:
        print(f"WARN  {w}")
    for e in rep.errors:
        print(f"ERROR {e}")
    print("OK: каталог валиден" if rep.ok else f"FAIL: ошибок {len(rep.errors)}")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m norms.validate", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    ap.add_argument("--base-dir", type=Path, default=None,
                    help="папка, относительно которой лежат source_file (по умолчанию — папка каталога)")
    ap.add_argument("--strict", action="store_true", help="предупреждения считать ошибками")
    ap.add_argument("--json", action="store_true", help="вывести отчёт в JSON")
    ap.add_argument("--no-fallback-check", action="store_true")
    ap.add_argument("--write-fallback", action="store_true", help="пересобрать norms/_fallback.py")
    args = ap.parse_args(argv)

    catalog = _load_yaml_strict(args.catalog)
    if args.write_fallback:
        write_fallback(catalog)
        print("norms/_fallback.py обновлён")
    rep = validate_catalog(catalog, args.base_dir or args.catalog.parent, strict=args.strict)
    if not args.no_fallback_check and args.catalog.resolve() == CATALOG_PATH.resolve():
        check_fallback(catalog, rep)
    if args.json:
        print(json.dumps(rep.to_dict(), ensure_ascii=False, indent=2))
    else:
        _print_human(rep, args.catalog)
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
