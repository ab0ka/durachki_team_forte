"""Пост-проверка ответа: пользователю уходит только то, что подтверждено контекстом.

Проверяем (по предложениям):
* маркеры [Nk] — только из выданного контекста; превращаются в сноски [1], [2] по порядку citations;
* упоминания «п. 15», «пункт 15», «статья 5» — пункт должен быть среди норм, переданных в контексте;
* названия актов («Правила «…»», «Закон «…»») — документ должен быть в корпусе;
* URL — только известные корпусу;
* таймкоды — только из результата анализа или из вопроса/истории;
* денежные суммы — только встречающиеся в результате/вопросе (по видео сумма не устанавливается);
* опора на контекст — предложение без лексической опоры на результат/нормы считается домыслом.
Предложения, не прошедшие проверку, вырезаются; если не осталось ничего — честный отказ.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from .corpus import Corpus, norm_clause, norm_text, norm_url, _CLAUSE_RE
from .llm import strip_think
from .prompts import NormRef, REFUSAL_NO_DATA, REFUSAL_SCOPE, REFUSAL_SOURCES, REFUSAL_VIDEO, fmt_tc

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
_AMOUNT_RE = re.compile(
    r"(?:[$€₸₽]\s*\d[\d\s\u00a0]*(?:[.,]\d+)?)"
    r"|(?:(?<![\w.])\d[\d\s\u00a0]*(?:[.,]\d+)?\s*(?:тыс\.?|млн\.?|млрд\.?)?\s*"
    r"(?:тенге|тг\b|₸|руб\w*|₽|доллар\w*|\$|usd\b|евро|€|eur\b))",
    re.I,
)
_BOUNDARY_RE = re.compile(r"([.!?…]+)(\s+)")
_ITEM_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
# сокращения, после точки которых предложение НЕ заканчивается («п. 9», «ст. 5», «22 с. кассир»)
_ABBREV = {"п", "пп", "ст", "ч", "гл", "разд", "прил", "т.е", "т.д", "т.п", "т.к", "др", "г", "гг", "с",
           "сек", "мин", "им", "рис", "см", "напр", "тыс", "млн", "млрд", "руб", "тг", "no", "№", "стр", "абз"}
_WORD_RE = re.compile(r"[a-zа-яё]{4,}", re.I)
_STOP = set("""это этот эта эти этого этой того тому также если или либо чтобы тоже только уже ещё еще
его её ее их она они оно был была были быть есть может могут можно нужно должен должна должны который
которая которые которое чем где когда тогда здесь там весь вся всего всему всех очень более менее через
после перед между около какой какая какие такой такая такие свой своя свои себя этом этих""".split())
# служебная лексика ответов ассистента (не факты) — не требует опоры на контекст
_ANSWER_VOCAB = {w[:4] for w in """признак возможного нарушение проверяющий решение принимает подсказка заключение
видео основание норма источник пункт данные установить нельзя факты наблюдаемые кадры таймкод уверенность
анализ результат система фрагмент просмотреть проверить вручную документ сноска интервал секунда минута
согласно требование запрещено допускается указано видно отмечено зафиксировано обнаружено рекомендуется
сверить подтвердить отклонить алгоритм оценка вероятность человек определяется является случае ответ
информация относящиеся найденные данных имеющимся""".split()}

REFUSAL_PATTERNS = [
    r"установить нельзя", r"нельзя установить", r"не\s+устанавлива", r"в источниках нет", r"нет информации",
    r"нет данных", r"не могу ответить", r"не выполняется", r"не определяется", r"невозможно определить",
    re.escape(REFUSAL_SCOPE.lower()),
]
UNVERIFIED_ANSWER = f"{REFUSAL_NO_DATA}: ответ модели не подтверждён результатом анализа и источниками."


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


def _is_boundary(line: str, start: int, m: re.Match[str]) -> bool:
    """Конец предложения? Не режем после сокращений («п. 9») и перед строчной буквой/цифрой."""
    if m.group(1) != ".":
        return True
    before = line[start:m.start(1)]
    lw = re.search(r"([\w.№]+)$", before)
    last = lw.group(1).lower() if lw else ""
    nxt = line[m.end():m.end() + 1]
    if last in _ABBREV or (len(last) == 1 and last.isalpha()):
        return False
    if nxt and (nxt.isdigit() or nxt.islower()):
        return False
    return True


def _split_sentences(text: str) -> list[str]:
    parts: list[str] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if _ITEM_RE.match(line):
            parts.append(line)
            continue
        start = 0
        for m in _BOUNDARY_RE.finditer(line):
            if _is_boundary(line, start, m):
                parts.append(line[start:m.end(1)].strip())
                start = m.end()
        tail = line[start:].strip()
        if tail:
            parts.append(tail)
    return parts


def complete_prefix(text: str) -> str:
    """Часть текста до последнего завершённого предложения (для потоковой выдачи проверенного текста)."""
    text = text or ""
    cut = text.rfind("\n") + 1
    line_start = cut
    line = text[line_start:]
    start = 0
    for m in _BOUNDARY_RE.finditer(line):
        if _is_boundary(line, start, m):
            cut = line_start + m.end(1)
            start = m.end()
    return text[:cut]


def _stems(text: str) -> set[str]:
    return {w.lower().replace("ё", "е")[:4] for w in _WORD_RE.findall(text or "") if w.lower() not in _STOP}


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


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s or "")


def check_answer(raw: str, refs: Sequence[NormRef], corpus: Optional[Corpus], result: dict[str, Any],
                 question: str = "", history: Sequence[dict[str, str]] = (),
                 tolerance_s: float = 1.5, min_support: float = 0.3) -> GuardResult:
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

    # --- 3. контекст, на который ответ имеет право опираться
    user_text = " ".join([question] + [str(t.get("content") or "") for t in history if t.get("role") == "user"])
    allowed_times = [float(t) for t in _result_times(result)] + _times_in(user_text)
    for turn in history:
        allowed_times += _times_in(str(turn.get("content") or ""))
    ref_clauses = {norm_clause(r.chunk.clause) for r in refs if r.chunk.clause}
    known_urls = {norm_url(r.chunk.url) for r in refs if r.chunk.url}
    if corpus is not None:
        known_urls |= corpus.known_urls()
    result_text = json.dumps(result, ensure_ascii=False, default=str)
    context_text = " ".join([result_text, user_text] + [f"{r.chunk.document} {r.chunk.heading} {r.chunk.text}" for r in refs])
    if corpus is not None:
        context_text += " " + " ".join(corpus.type_name(str(v.get("type_id") or "")) for v in viols.values())
    # суммы, явно упомянутые в контексте (результат/вопрос/нормы); совпадение с таймкодом не считается
    context_amounts = {_digits(m.group(0)) for m in _AMOUNT_RE.finditer(context_text)}
    context_stems = _stems(context_text) | _ANSWER_VOCAB

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
            # только пункты из переданных модели норм: модель не видела остальной корпус,
            # а «п. 12» почти наверняка найдётся в каком-нибудь документе каталога
            if norm_clause(cm.group(0)) in ref_clauses:
                continue
            removed.append(cm.group(0))
            bad = True
        for tsec in _times_in(sent):
            if not any(abs(tsec - a) <= tolerance_s for a in allowed_times):
                removed.append(f"{tsec:g} с")
                bad = True
        for mm in _AMOUNT_RE.finditer(sent):
            d = _digits(mm.group(0))
            if d and d not in context_amounts:
                removed.append(f"сумма: {mm.group(0).strip()}")
                bad = True
        if not bad and not is_refusal(sent):
            stems = _stems(re.sub(r"\(№\d+, [\d:]+\)|\[\d+\]", " ", sent))
            if len(stems) >= 4 and len(stems & context_stems) / len(stems) < min_support:
                removed.append(f"без опоры на контекст: {sent[:60]}")
                bad = True
        if not bad and sent.strip():
            kept.append(sent.strip())

    answer = ""
    prev_item = False
    for part in kept:
        is_item = bool(_ITEM_RE.match(part))
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
        cite.update({"ref": order.index(n) + 1, "quote": r.chunk.text, "verified": r.chunk.verified,
                     "relation": r.relation, "status": r.chunk.status, "note": r.status_ru,
                     "events": list(r.events)})
        if corpus is not None:
            short = corpus.short_name(r.chunk.document, r.chunk.url)
            if short != r.chunk.document:
                cite["short"] = short
        citations.append(cite)

    raw_invalid = bool(removed)
    if not answer and raw_invalid:
        answer = UNVERIFIED_ANSWER
    refusal = is_refusal(answer)
    refers_result = bool(re.search(r"№\d+, \d{2}:\d{2}", answer)) or bool(_times_in(answer))
    supported = bool(citations) or refusal or refers_result or bool(
        re.search(r"признак\w* (возможного )?нарушени|не обнаружен", answer.lower()))
    grounded = bool(answer) and not raw_invalid and supported
    warnings = [f"удалены неподтверждённые фрагменты: {', '.join(dict.fromkeys(removed))}"] if removed else []
    return GuardResult(answer=answer, citations=citations, grounded=grounded, refusal=refusal,
                       removed=list(dict.fromkeys(removed)), raw_invalid=raw_invalid, warnings=warnings)


def _result_times(result: dict[str, Any]) -> list[float]:
    from .prompts import result_times

    return result_times(result)


__all__ = ["GuardResult", "UNVERIFIED_ANSWER", "check_answer", "complete_prefix", "is_refusal",
           "REFUSAL_NO_DATA", "REFUSAL_SOURCES", "REFUSAL_VIDEO"]
