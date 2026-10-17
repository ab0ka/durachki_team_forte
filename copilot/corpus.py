"""Корпус норм для RAG: norms/catalog.yaml (A13) + norms/sources/*.md -> чанки с источником.

Каждый чанк несёт документ, пункт и URL — это и есть единица цитирования. Ссылки на нормы
берутся ТОЛЬКО из каталога/корпуса (правило 5); фикстура copilot/fixtures — лишь для тестов.
"""
from __future__ import annotations

import glob
import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from .config import default_config, fixture_allowed, resolve_path

log = logging.getLogger(__name__)

_FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_URL_RE = re.compile(r"https?://[^\s)\]>»\"']+")
_ADILET_RE = re.compile(r"https?://adilet\.zan\.kz[^\s)\]>»\"']*")
# «п. 12», «пункт 12.3», «пп. 2)», «подпункт 2», «статья 5», «ст. 5», «глава 3»
_CLAUSE_RE = re.compile(
    r"(?<![\wА-Яа-яЁё])(?P<kind>подпункт(?:а|е|ом|у)?|пп\.|пункт(?:а|е|ом|у|ы|ов)?|п\.|стать(?:я|и|е|ю|ей)|ст\.)"
    r"\s*(?P<num>\d+(?:[.\-]\d+)*\)?)",
    re.I,
)
_BARE_NUM_RE = re.compile(r"^\s*(\d{1,3}(?:\.\d{1,3})*)\.\s+\S")


def norm_text(s: Optional[str]) -> str:
    s = (s or "").lower().replace("ё", "е")
    s = re.sub(r"[«»\"'“”„`]", "", s)
    return re.sub(r"\s+", " ", s).strip(" .,;:")


def norm_clause(clause: Optional[str]) -> str:
    """Канонический ключ пункта: «пункт 3», «п.3», «п. 3» -> «п.3»; «статья 5» -> «ст.5»."""
    if not clause:
        return ""
    s = norm_text(clause)
    m = _CLAUSE_RE.search(s)
    if m:
        kind = m.group("kind").lower()
        num = m.group("num").rstrip(")").replace("-", ".")
        if kind.startswith(("подпункт", "пп")):
            prefix = "пп."
        elif kind.startswith(("стат", "ст.")):
            prefix = "ст."
        else:
            prefix = "п."
        return prefix + num
    m2 = re.match(r"^\s*(\d+(?:\.\d+)*)", s)
    if m2:
        return "п." + m2.group(1)
    return s.replace(" ", "")


def norm_url(url: Optional[str]) -> str:
    u = (url or "").strip().lower()
    u = re.sub(r"^https?://", "", u)
    u = u.split("#", 1)[0].rstrip("/")
    return u


@dataclass
class Chunk:
    id: str
    text: str                  # текст для показа и цитаты
    document: str
    clause: Optional[str]
    url: str
    kind: str                  # catalog_rule | source
    type_ids: list[str] = field(default_factory=list)
    source_file: str = ""
    heading: str = ""
    verified: bool = False

    @property
    def embed_text(self) -> str:
        head = " · ".join(x for x in (self.document, self.heading, self.clause or "") if x)
        return f"{head}. {self.text}" if head else self.text

    @property
    def key(self) -> tuple[str, str]:
        return norm_text(self.document), norm_clause(self.clause)

    def citation(self) -> dict[str, Any]:
        return {"document": self.document, "clause": self.clause, "url": self.url}


@dataclass
class Corpus:
    chunks: list[Chunk]
    catalog: dict[str, Any]
    is_fixture: bool = False
    warnings: list[str] = field(default_factory=list)

    # --- каталог ---
    @property
    def types(self) -> dict[str, Any]:
        types = self.catalog.get("types") if isinstance(self.catalog, dict) else None
        return types if isinstance(types, dict) else {}

    def type_name(self, type_id: str) -> str:
        info = self.types.get(type_id) or {}
        return str(info.get("name") or type_id)

    def type_info(self, type_id: str) -> dict[str, Any]:
        info = self.types.get(type_id)
        return info if isinstance(info, dict) else {}

    def rules_for_type(self, type_id: str) -> list[Chunk]:
        return [c for c in self.chunks if c.kind == "catalog_rule" and type_id in c.type_ids]

    # --- проверка цитат ---
    def fingerprint(self) -> str:
        h = hashlib.sha1()
        for c in self.chunks:
            h.update(c.embed_text.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()[:16]

    def documents(self) -> set[str]:
        return {norm_text(c.document) for c in self.chunks if c.document}

    def find(self, document: Optional[str] = None, clause: Optional[str] = None,
             url: Optional[str] = None) -> Optional[Chunk]:
        """Найти чанк, подтверждающий цитату (документ+пункт, либо URL+пункт)."""
        doc_k = norm_text(document)
        cl_k = norm_clause(clause)
        url_k = norm_url(url)
        for c in self.chunks:
            c_doc, c_cl = c.key
            doc_ok = bool(doc_k) and (doc_k == c_doc or doc_k in c_doc or (len(c_doc) > 8 and c_doc in doc_k))
            url_ok = bool(url_k) and url_k == norm_url(c.url)
            if not (doc_ok or url_ok):
                continue
            if cl_k and cl_k != c_cl:
                continue
            return c
        return None

    def clause_exists(self, clause: str, document: Optional[str] = None) -> bool:
        cl_k = norm_clause(clause)
        if not cl_k:
            return False
        doc_k = norm_text(document)
        for c in self.chunks:
            c_doc, c_cl = c.key
            if c_cl == cl_k and (not doc_k or doc_k in c_doc or c_doc in doc_k):
                return True
        return False

    def known_urls(self) -> set[str]:
        return {norm_url(c.url) for c in self.chunks if c.url}


# ----------------------------------------------------------------------------- loading

def _read_yaml(path: Path) -> Any:
    import yaml

    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _load_catalog(cfg: dict[str, Any], warnings: list[str]) -> tuple[dict[str, Any], bool]:
    """Возвращает (catalog, is_fixture)."""
    ccfg = cfg.get("corpus", {})
    path = resolve_path(ccfg.get("catalog_path"))
    if path and path.is_file():
        try:
            data = _read_yaml(path) or {}
            if isinstance(data, dict):
                return data, bool(data.get("fixture"))
        except Exception as exc:  # битый YAML не должен ронять ассистента
            warnings.append(f"catalog read error: {exc}")
    # A13 может отдавать каталог функцией (если файл лежит в другом месте/собирается)
    if path and path == resolve_path("norms/catalog.yaml"):
        try:
            from norms import load_catalog  # type: ignore

            data = load_catalog()
            if isinstance(data, dict) and data.get("types"):
                return data, False
        except Exception:
            pass
    if fixture_allowed(cfg):
        fx = resolve_path(ccfg.get("fixture_dir", "copilot/fixtures/norms"))
        fx_path = fx / "catalog.yaml" if fx else None
        if fx_path and fx_path.is_file():
            warnings.append("norms/catalog.yaml не найден — используется ТЕСТОВАЯ фикстура (не нормы РК)")
            return _read_yaml(fx_path) or {}, True
    warnings.append("каталог норм недоступен — ссылки на нормы не будут приводиться")
    return {"version": 1, "types": {}}, False


def catalog_chunks(catalog: dict[str, Any]) -> list[Chunk]:
    out: list[Chunk] = []
    types = catalog.get("types") if isinstance(catalog, dict) else None
    if not isinstance(types, dict):
        return out
    for type_id, info in types.items():
        if not isinstance(info, dict):
            continue
        for j, rule in enumerate(info.get("rules") or []):
            if not isinstance(rule, dict) or not rule.get("document"):
                continue
            quote = str(rule.get("quote") or "").strip()
            name = str(info.get("name") or type_id)
            out.append(Chunk(
                id=f"cat:{type_id}:{j}",
                text=quote or name,
                document=str(rule.get("document")),
                clause=(str(rule["clause"]) if rule.get("clause") else None),
                url=str(rule.get("url") or ""),
                kind="catalog_rule",
                type_ids=[str(type_id)],
                heading=name,
                verified=bool(rule.get("verified", False)),
            ))
    # одна и та же норма может относиться к нескольким типам — объединяем type_ids
    merged: dict[tuple[str, str, str], Chunk] = {}
    for c in out:
        k = (c.key[0], c.key[1], norm_text(c.text))
        if k in merged:
            for t in c.type_ids:
                if t not in merged[k].type_ids:
                    merged[k].type_ids.append(t)
        else:
            merged[k] = c
    return list(merged.values())


def _split_window(text: str, max_chars: int, overlap: int) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    sentences = re.split(r"(?<=[.!?;])\s+", text)
    parts: list[str] = []
    cur = ""
    for s in sentences:
        while len(s) > max_chars:  # очень длинное «предложение» режем жёстко
            if cur:
                parts.append(cur)
                cur = ""
            parts.append(s[:max_chars])
            s = s[max_chars - overlap:] if overlap < max_chars else s[max_chars:]
        if len(cur) + len(s) + 1 <= max_chars:
            cur = f"{cur} {s}".strip()
        else:
            if cur:
                parts.append(cur)
            tail = cur[-overlap:] if overlap and cur else ""
            if tail and " " in tail:
                tail = tail.split(" ", 1)[1]
            cur = f"{tail} {s}".strip() if tail else s
    if cur:
        parts.append(cur)
    return parts


def _clause_from_heading(title: str) -> Optional[str]:
    m = _CLAUSE_RE.search(title)
    if m and m.start() < 4:
        kind = m.group("kind").lower()
        num = m.group("num")
        if kind.startswith(("подпункт", "пп")):
            return f"пп. {num}"
        if kind.startswith(("стат", "ст.")):
            return f"ст. {num.rstrip(')')}"
        return f"п. {num.rstrip(')')}"
    m2 = re.match(r"^\s*(\d{1,3}(?:\.\d{1,3})*)\.?\s*$", title)
    if m2:
        return f"п. {m2.group(1)}"
    return None


def chunk_markdown(text: str, source_file: str = "", max_chars: int = 900, overlap: int = 120,
                   min_chars: int = 40) -> list[Chunk]:
    """Чанкинг markdown-источника нормы: front matter / H1 -> документ; заголовки и «N.» -> пункты."""
    meta: dict[str, Any] = {}
    m = _FRONT_MATTER_RE.match(text)
    if m:
        try:
            import yaml

            meta = yaml.safe_load(m.group(1)) or {}
            if not isinstance(meta, dict):
                meta = {}
        except Exception:
            meta = {}
        text = text[m.end():]
    lines = text.splitlines()
    document = str(meta.get("title") or meta.get("document") or meta.get("name") or "").strip()
    if not document:
        for ln in lines:
            hm = _HEADING_RE.match(ln)
            if hm and len(hm.group(1)) == 1:
                document = hm.group(2).strip()
                break
    if not document:
        document = Path(source_file).stem if source_file else "Документ"
    url = str(meta.get("url") or "").strip()
    if not url:
        um = _ADILET_RE.search(text) or _URL_RE.search(text)
        url = um.group(0) if um else ""
    doc_id = str(meta.get("doc_id") or meta.get("id") or Path(source_file).stem or "doc")

    sections: list[tuple[Optional[str], str, list[str]]] = []  # (clause, heading_path, lines)
    headings: dict[int, str] = {}
    cur_clause: Optional[str] = None
    cur_lines: list[str] = []
    cur_heading = ""

    def flush() -> None:
        nonlocal cur_lines
        body = "\n".join(cur_lines).strip()
        if body:
            sections.append((cur_clause, cur_heading, [body]))
        cur_lines = []

    for ln in lines:
        hm = _HEADING_RE.match(ln)
        if hm:
            level = len(hm.group(1))
            title = hm.group(2).strip()
            if level == 1 and norm_text(title) == norm_text(document):
                continue
            flush()
            headings = {k: v for k, v in headings.items() if k < level}
            clause = _clause_from_heading(title)
            if clause:
                cur_clause = clause
            else:
                headings[level] = title
                cur_clause = None
            cur_heading = " / ".join(headings[k] for k in sorted(headings))
            continue
        bm = _BARE_NUM_RE.match(ln)
        if bm and not ln.lstrip().startswith(("#", ">")):
            flush()
            cur_clause = f"п. {bm.group(1)}"
        cur_lines.append(ln.strip().lstrip("> ").strip())
    flush()

    chunks: list[Chunk] = []
    for i, (clause, heading, body_lines) in enumerate(sections):
        body = re.sub(r"\s+", " ", " ".join(body_lines)).strip()
        if len(body) < min_chars and not clause:
            continue
        for j, part in enumerate(_split_window(body, max_chars, overlap)):
            chunks.append(Chunk(
                id=f"src:{doc_id}:{i}:{j}",
                text=part,
                document=document,
                clause=clause,
                url=url,
                kind="source",
                source_file=source_file,
                heading=heading,
                verified=bool(meta.get("verified", False)),
            ))
    return chunks


def _source_files(cfg: dict[str, Any], use_fixture: bool) -> list[Path]:
    ccfg = cfg.get("corpus", {})
    if use_fixture:
        fx = resolve_path(ccfg.get("fixture_dir", "copilot/fixtures/norms"))
        pattern = str(fx / "sources" / "*.md") if fx else ""
    else:
        p = resolve_path(ccfg.get("sources_glob"))
        pattern = str(p) if p else ""
    return [Path(p) for p in sorted(glob.glob(pattern))] if pattern else []


def build_corpus(catalog: dict[str, Any], source_texts: Iterable[tuple[str, str]] = (),
                 cfg: Optional[dict[str, Any]] = None, is_fixture: bool = False,
                 warnings: Optional[list[str]] = None) -> Corpus:
    """Собрать корпус из каталога и пар (имя_файла, markdown). Удобно для тестов."""
    ccfg = (cfg or default_config()).get("corpus", {})
    chunks = catalog_chunks(catalog)
    for name, text in source_texts:
        chunks.extend(chunk_markdown(
            text, source_file=name,
            max_chars=int(ccfg.get("chunk_max_chars", 900)),
            overlap=int(ccfg.get("chunk_overlap_chars", 120)),
            min_chars=int(ccfg.get("chunk_min_chars", 40)),
        ))
    # привязка источников к типам: если пункт совпадает с пунктом из каталога
    by_key: dict[tuple[str, str], list[str]] = {}
    for c in chunks:
        if c.kind == "catalog_rule":
            by_key.setdefault(c.key, []).extend(c.type_ids)
    for c in chunks:
        if c.kind == "source" and c.key in by_key:
            c.type_ids = sorted(set(by_key[c.key]))
    return Corpus(chunks=chunks, catalog=catalog, is_fixture=is_fixture, warnings=list(warnings or []))


def load_corpus(cfg: Optional[dict[str, Any]] = None) -> Corpus:
    cfg = cfg or default_config()
    warnings: list[str] = []
    catalog, is_fixture = _load_catalog(cfg, warnings)
    texts: list[tuple[str, str]] = []
    for path in _source_files(cfg, use_fixture=is_fixture):
        try:
            texts.append((path.name, path.read_text(encoding="utf-8")))
        except OSError as exc:
            warnings.append(f"source read error {path.name}: {exc}")
    corpus = build_corpus(catalog, texts, cfg=cfg, is_fixture=is_fixture, warnings=warnings)
    log.info("copilot corpus: %d chunks (fixture=%s)", len(corpus.chunks), is_fixture)
    return corpus
