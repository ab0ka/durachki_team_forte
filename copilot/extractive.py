"""Детерминированный составитель ответа (без генерации) + классификатор намерений вопроса.

Используется: (1) как ``mock``-бэкенд при SENTINEL_MOCK=1; (2) как явно включаемый бэкенд
``extractive`` (только по конфигу, НЕ автоматический fallback); (3) для обязательных отказов
на вопросы о невидимом (сумма, подлинность, умысел, личность…) — до вызова LLM.
Ответ собирается только из фактов результата и найденных норм, с маркерами [Nk]/[Vk].
"""
from __future__ import annotations

import re
from typing import Any, Optional, Sequence

from .corpus import Corpus
from .prompts import NormRef, REFUSAL_SCOPE, REFUSAL_SOURCES, REFUSAL_VIDEO, fmt_tc, unobservable_items, verdict_text

# порядок важен: первое совпадение побеждает
_INTENT_PATTERNS: list[tuple[str, str]] = [
    ("identity", r"кто\s+(это|он|она|они|такой|такая|эти)\b|как\s+(его|её|ее|их)?\s*зовут|фамили|\bфио\b|личност|"
                 r"опозна|распозна\w*\s+лиц|идентифиц|по\s+лицу|узна(ть|й)\s+(человека|лицо)"),
    ("unobservable", r"сколько\s+(денег|наличн|купюр|тенге|средств)|как(ую|ая)\s+сумм|сумм[ауы]\s+(денег|наличн|в\s+сумке)|"
                     r"подлинн|фальшив|поддел|умыс|умышлен|намеренн|нарочно|специально|хотел[аи]?\s+(ли\s+)?(укра|похит|скрыть)|"
                     r"собирал(ся|ась)|укра(л|ла|сть)|похитил|кража|ворует|проводк|допуск|уполномоч|имеет\s+ли\s+(он|она)?\s*право|"
                     r"что\s+(он|она|они)\s+(говор|сказал)|о\s+ч[её]м\s+(говор|разговар)|содержани\w*\s+разговор"),
    ("offtopic", r"\bстих|анекдот|погод|рецепт|переведи|напиши\s+(код|программ|письмо|эссе)|сочини|курс\s+(доллар|валют)"),
    ("fine", r"штраф|наказан|санкци|ответственност|уволь|выговор"),
    ("evidence", r"кадр|доказательств|скриншот|фото|изображени"),
    ("checklist", r"что\s+(мне\s+)?(проверить|делать|сделать)|как\s+проверить|рекомендац|следующ\w+\s+шаг|на\s+что\s+обратить"),
    ("confidence", r"уверен|вероятност|точност|насколько\s+(надёжн|надежн|точн)|ошиб(ка|ся|лась)|ложн"),
    ("when", r"когда|в\s+какой\s+момент|на\s+какой\s+секунд|таймкод|в\s+какое\s+время|время\s+нарушени"),
    ("why", r"почему|на\s+каком\s+основании|за\s+что|объясни|поясни|что\s+не\s+так|в\s+ч[её]м\s+(нарушение|проблема)|чем\s+нарушен"),
    ("norm", r"норм|пункт|правил|закон|документ|стать[яи]|требовани|регламент|инструкц|запрещ|разрешен|можно\s+ли|"
             r"допускается|ссылк|основани"),
    ("summary", r"сколько\s+нарушени|какие\s+нарушени|есть\s+ли\s+нарушени|вердикт|итог|что\s+(найдено|обнаружено|видно)|"
                r"резюм|кратко|опиши|что\s+произошло|нарушени"),
]

_TYPE_HINTS: dict[str, str] = {
    "V01": r"посторон|клиент\w*\s+за|чуж|неуполномоч",
    "V02": r"покинул|отош[её]л|ушл?[аи]?\b|без\s+присмотр|рабоч\w+\s+мест",
    "V03": r"сейф",
    "V04": r"двер|перегородк",
    "V05": r"телефон|смартфон|мобильн",
    "V06": r"\bе(ст|л|да)\b|пь[её]т|напит|кофе|чай|еда|пищ",
    "V07": r"инкасс|сумк",
    "V08": r"пересч|скрыт|вне\s+обзора|руки",
}

_ORDINALS = {"перв": 1, "втор": 2, "трет": 3, "четв": 4, "пят": 5, "шест": 6}


def classify(question: str) -> str:
    q = (question or "").lower().replace("ё", "е")
    for intent, pattern in _INTENT_PATTERNS:
        if re.search(pattern.replace("ё", "е"), q):
            return intent
    return "general"


def target_violations(question: str, result: dict[str, Any]) -> list[dict[str, Any]]:
    """Какие нарушения имеет в виду вопрос: по номеру («второе», «№2») или по типу (телефон -> V05)."""
    viols = result.get("violations") or []
    q = (question or "").lower()
    nums = {int(n) for n in re.findall(r"(?:№|номер|нарушени\w*)\s*(\d{1,2})", q)}
    for stem_, n in _ORDINALS.items():
        if re.search(stem_ + r"\w*\s+нарушени", q):
            nums.add(n)
    picked = [v for v in viols if v.get("_n") in nums]
    if picked:
        return picked
    types = {tid for tid, pat in _TYPE_HINTS.items() if re.search(pat, q)}
    picked = [v for v in viols if v.get("type_id") in types]
    return picked or list(viols)


def refusal_for(intent: str, result: dict[str, Any], corpus: Optional[Corpus]) -> Optional[str]:
    if intent == "identity":
        return (f"{REFUSAL_VIDEO}: система не распознаёт и не идентифицирует людей по лицу. "
                "Роли (кассир, посетитель, инкассатор) определяются только по зоне, поведению и предметам.")
    if intent == "unobservable":
        items = "; ".join(unobservable_items(result, corpus))
        return (f"{REFUSAL_VIDEO}. По видео фиксируются только видимые действия; не устанавливаются: {items}. "
                "Это проверяется по документам и учётной системе.")
    if intent == "offtopic":
        return f"{REFUSAL_SCOPE}."
    return None


def _viol_line(v: dict[str, Any], refs: Sequence[NormRef], corpus: Optional[Corpus]) -> str:
    tid = v.get("type_id") or ""
    name = v.get("type") or (corpus.type_name(tid) if corpus and tid else "") or "возможное нарушение"
    marks = " ".join(r.marker for r in refs if tid and tid in r.chunk.type_ids)
    conf = v.get("confidence")
    conf_s = f", уверенность {float(conf):.2f}" if conf not in (None, "") else ""
    desc = str(v.get("description") or "").strip().rstrip(".")
    line = f"- [V{v.get('_n')}] {name}{conf_s}"
    if desc:
        line += f". Что видно: {desc}"
    if marks:
        line += f". Норма: {marks}"
    return line + "."


def compose(intent: str, question: str, result: dict[str, Any], refs: Sequence[NormRef],
            corpus: Optional[Corpus]) -> str:
    """Собрать ответ с маркерами [Nk]/[Vk] (пройдёт ту же пост-проверку, что и ответ LLM)."""
    refusal = refusal_for(intent, result, corpus)
    if refusal:
        return refusal
    viols = result.get("violations") or []
    targets = target_violations(question, result)
    status = result.get("status")
    if status in {"processing", "failed"}:
        return f"Результат недоступен: {verdict_text(result)}. Нет данных для ответа."

    if intent == "fine":
        hits = [r for r in refs if re.search(r"штраф|ответствен|санкц|наказан", r.chunk.text.lower())]
        if not hits:
            return f"{REFUSAL_SOURCES}: в найденных нормах не указаны штрафы или ответственность."
        return "В источниках: " + " ".join(f"«{r.chunk.text}» {r.marker}" for r in hits[:2])

    if intent == "norm":
        rel = [r for r in refs if not targets or any(t.get("type_id") in r.chunk.type_ids for t in targets)] or list(refs)
        if not rel:
            return f"{REFUSAL_SOURCES}."
        lines = ["Относящиеся нормы из источников:"]
        for r in rel[:3]:
            lines.append(f"- {r.chunk.document}" + (f", {r.chunk.clause}" if r.chunk.clause else "") + f": «{r.chunk.text}» {r.marker}")
        return "\n".join(lines)

    if not viols:
        base = "Признаков нарушения по результату анализа не обнаружено."
        if intent in {"why", "when", "evidence"}:
            return base + " Нет данных о нарушениях, поэтому таймкодов и кадров-доказательств нет."
        if intent == "checklist":
            return base + " Рекомендуется выборочно просмотреть видео: автоматический анализ может пропустить нарушение."
        return base + " Это подсказка, а не заключение: окончательное решение принимает проверяющий."

    if intent == "when":
        parts = []
        for v in targets:
            span = f" (интервал {fmt_tc(v.get('t_start'))}–{fmt_tc(v.get('t_end'))})" if v.get("t_end") else ""
            parts.append(f"- [V{v.get('_n')}] {v.get('type') or v.get('type_id')}{span}.")
        return "Таймкоды возможных нарушений:\n" + "\n".join(parts)

    if intent == "evidence":
        parts = []
        for v in targets:
            frames = [fmt_tc(f.get("timestamp_sec")) for f in v.get("evidence_frames") or []]
            parts.append(f"- [V{v.get('_n')}]: " + (f"кадры {', '.join(frames)}." if frames else "кадров-доказательств нет."))
        return "Кадры-доказательства:\n" + "\n".join(parts)

    if intent == "confidence":
        parts = [f"- [V{v.get('_n')}]: уверенность {float(v.get('confidence') or 0):.2f}." for v in targets]
        return ("Уверенность модели по нарушениям:\n" + "\n".join(parts) +
                "\nЭто оценка алгоритма, а не вероятность нарушения: требуется просмотр фрагмента человеком.")

    if intent == "checklist":
        parts = []
        for v in targets:
            span = (f"{fmt_tc(v.get('t_start'))}–{fmt_tc(v.get('t_end'))}" if v.get("t_end")
                    else fmt_tc(v.get("timecode_sec")))
            parts.append(f"- Просмотреть фрагмент {span} с запасом до и после [V{v.get('_n')}] и сверить с описанием.")
        parts.append("- Сверить с документами то, что по видео не устанавливается: "
                     + "; ".join(unobservable_items(result, corpus)) + ".")
        return "Что проверить вручную:\n" + "\n".join(parts)

    # why / summary / general
    head = (f"Вердикт: {verdict_text(result)}; возможных нарушений: {len(viols)}."
            if intent in {"summary", "general"} else "Основания (только наблюдаемые факты):")
    lines = [head] + [_viol_line(v, refs, corpus) for v in targets]
    if not any(r for r in refs if any(t.get("type_id") in r.chunk.type_ids for t in targets)):
        lines.append(f"{REFUSAL_SOURCES} о конкретной норме для этого случая.")
    lines.append("Это признаки возможного нарушения, решение принимает проверяющий.")
    return "\n".join(lines)
