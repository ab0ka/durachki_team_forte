"""Пост-проверка ответа: каждая цитата должна существовать в каталоге/корпусе, иначе удаляется.

Проверяем:
* маркеры [Nk] — только из выданного контекста; превращаются в сноски [1], [2] по порядку citations;
* упоминания «п. 15», «пункт 15», «статья 5» — пункт должен быть в контексте или в корпусе;
* названия актов («Правила «…»», «Закон «…»») — документ должен быть в корпусе;
* URL — только известные корпусу;
* таймкоды — только из результата анализа или из вопроса/истории.
Предложения с выдуманными ссылками/таймкодами вырезаются.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from .corpus import Corpus, norm_clause, norm_text, norm_url, _CLAUSE_RE
from .llm import strip_think
from .prompts import NormRef, REFUSAL_SCOPE, REFUSAL_SOURCES, REFUSAL_VIDEO, fmt_tc

_MARKER_RE = re.compile(r"[\[(]\s*([NnНн])\s*(\d{1,2})\s*[\])]")
_VMARK_RE = re.compile(r"\[\s*[Vv]\s*(\d{1,2})\s*\]")
_URL_RE = re.compile(r"https?://[^\s)\]>»\"']+")
_ACT_RE = re.compile(
    r"(?P<kind>Правил\w*|Закон\w*|Постановлени\w*|Приказ\w*|Инструкци\w*|Кодекс\w*|Положени\w*|Регламент\w*)"
    r"\s*(?:РК|Республики Казахстан|НБРК|Национального Банка[^«\"]{0,40})?\s*[«\"](?P<title>[^»\"]{5,200})[»\"]",
)
_TIME_RE = re.compile(
    r"(?<![\d:.])(?:(?P<h>\d{1,2}):)?(?P<m>\d{1,2}):(?P<s>\d{2})(?![\d:])"
    r"|(?<![\d.,])(?P<sec>\d{1,4}(?:[.,]\d)?)\s*(?:-?(?:й|я|ой|ая|ую))?\s*(?:с\b|сек\b|сек\.|секунд\w*)",
    re.I,
)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+|\n+")

REFUSAL_PATTERNS = [
    r"установить нельзя", r"нельзя установить", r"не\s+устанавлива", r"в источниках нет", r"нет информации",
    r"нет данных", r"не могу ответить", r"не выполняется", r"не определяется", r"невозможно определить",
    re.escape(REFUSAL_SCOPE.lower()),
]


def is_refusal(text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t) for p in REFUSAL_PATTERNS)


@dataclass
class GuardResult:
    answer: str
    citations: list[dict[str, Any]]
    grounded: bool
    refusal: bool
    removed: list[str] = field(default_factory=list)
    raw_invalid: bool = False
    warnings: list[str] = field(default_factory=list)


def _times_in(text: str) -> list[float]:
    out: list[float] = []
    for m in _TIME_RE.finditer(text or ""):
        if m.group("sec"):
            try:
                out.append(float(m.group("sec").replace(",", ".")))
            except ValueError:
                pass
        elif m.group("m") is not None:
            h = int(m.group("h") or 0)
            out.append(h * 3600 + int(m.group("m")) * 60 + int(m.group("s")))
    return out


def _doc_known(title: str, corpus: Optional[Corpus], refs: Sequence[NormRef]) -> bool:
    t = norm_text(title)
    docs = {norm_text(r.chunk.document) for r in refs}
    if corpus is not None:
        docs |= corpus.documents()
    if not t:
        return True
    t_words = set(re.findall(r"\w{4,}", t))
    for d in docs:
        if t in d or d in t:
            return True
        d_words = set(re.findall(r"\w{4,}", d))
        if t_words and len(t_words & d_words) / len(t_words) >= 0.6:
            return True
    return False


def check_answer(raw: str, refs: Sequence[NormRef], corpus: Optional[Corpus], result: dict[str, Any],
                 question: str = "", history: Sequence[dict[str, str]] = (),
                 tolerance_s: float = 1.5) -> GuardResult:
    text = strip_think(raw)
    removed: list[str] = []
    by_n = {r.n: r for r in refs}
    viols = {int(v.get("_n", i + 1)): v for i, v in enumerate(result.get("violations") or [])}

    # --- 1. маркеры норм [Nk] -> сноски по порядку появления
    order: list[int] = []

    def _marker(m: re.Match[str]) -> str:
        n = int(m.group(2))
        if n not in by_n:
            removed.append(m.group(0))
            return ""
        if n not in order:
            order.append(n)
        return f"[{order.index(n) + 1}]"

    text = _MARKER_RE.sub(_marker, text)

    # --- 2. маркеры нарушений [Vk] -> «(№k, мм:сс)»
    def _vmark(m: re.Match[str]) -> str:
        n = int(m.group(1))
        v = viols.get(n)
        if v is None:
            removed.append(m.group(0))
            return ""
        return f"(№{n}, {fmt_tc(v.get('timecode_sec'))})"

    text = _VMARK_RE.sub(_vmark, text)

    # --- 3. по предложениям: пункты, акты, URL, таймкоды
    allowed_times = [float(t) for t in _result_times(result)]
    allowed_times += _times_in(question)
    for turn in history:
        allowed_times += _times_in(str(turn.get("content") or ""))
    ref_clauses = {(norm_text(r.chunk.document), norm_clause(r.chunk.clause)) for r in refs if r.chunk.clause}
    known_urls = {norm_url(r.chunk.url) for r in refs if r.chunk.url}
    if corpus is not None:
        known_urls |= corpus.known_urls()

    kept: list[str] = []
    for sent in _split_sentences(text):
        bad = False
        for um in _URL_RE.finditer(sent):
            if norm_url(um.group(0)) not in known_urls:
                removed.append(um.group(0))
                sent = sent.replace(um.group(0), "").strip()
        for am in _ACT_RE.finditer(sent):
            if not _doc_known(am.group("title"), corpus, refs):
                removed.append(am.group(0))
                bad = True
        for cm in _CLAUSE_RE.finditer(sent):
            ck = norm_clause(cm.group(0))
            if any(ck == rc for _, rc in ref_clauses):
                continue
            if corpus is not None and corpus.clause_exists(cm.group(0)):
                continue  # пункт существует в каталоге/корпусе
            removed.append(cm.group(0))
            bad = True
        for tsec in _times_in(sent):
            if not any(abs(tsec - a) <= tolerance_s for a in allowed_times):
                removed.append(f"{tsec:g} с")
                bad = True
        if not bad and sent.strip():
            kept.append(sent.strip())
    answer = ""
    prev_item = False
    for part in kept:
        is_item = bool(re.match(r"^(?:[-*•]|\d+[.)])\s+", part))
        sep = "\n" if (is_item or prev_item or answer.endswith(":")) else " "
        answer = f"{answer}{sep}{part}" if answer else part
        prev_item = is_item
    answer = answer.strip()
    answer = re.sub(r"[ \t]+([.,;:!?])", r"\1", answer)
    answer = re.sub(r"[ \t]{2,}", " ", answer)

    citations: list[dict[str, Any]] = []
    for n in order:
        r = by_n[n]
        if f"[{order.index(n) + 1}]" not in answer:
            continue  # сноска ушла вместе с вырезанным предложением
        cite = r.chunk.citation()
        cite.update({"ref": order.index(n) + 1, "quote": r.chunk.text, "verified": r.chunk.verified})
        citations.append(cite)

    refusal = is_refusal(answer)
    raw_invalid = bool(removed)
    refers_result = bool(re.search(r"№\d+, \d{2}:\d{2}", answer)) or bool(_times_in(answer))
    supported = bool(citations) or refusal or refers_result or bool(
        re.search(r"признак\w* (возможного )?нарушени|не обнаружен", answer.lower()))
    grounded = bool(answer) and not raw_invalid and supported
    warnings = [f"удалены неподтверждённые ссылки: {', '.join(dict.fromkeys(removed))}"] if removed else []
    return GuardResult(answer=answer, citations=citations, grounded=grounded, refusal=refusal,
                       removed=list(dict.fromkeys(removed)), raw_invalid=raw_invalid, warnings=warnings)


def _split_sentences(text: str) -> list[str]:
    # списки «- …» и «1) …» сохраняем построчно
    parts: list[str] = []
    for line in (text or "").splitlines():
        line = line.rstrip()
        if not line.strip():
            continue
        if re.match(r"^\s*(?:[-*•]|\d+[.)])\s+", line):
            parts.append(line.strip())
        else:
            parts.extend(s for s in _SENT_SPLIT_RE.split(line) if s.strip())
    return parts


def _result_times(result: dict[str, Any]) -> list[float]:
    from .prompts import result_times

    return result_times(result)


__all__ = ["GuardResult", "check_answer", "is_refusal", "REFUSAL_SOURCES", "REFUSAL_VIDEO"]
