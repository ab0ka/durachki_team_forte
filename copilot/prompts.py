"""Системный промпт, нормализация результата анализа и сборка контекста для LLM.

Один и тот же билдер используется в рантайме (service) и при генерации датасета для LoRA
(train/make_dataset) — формат промпта на обучении и в проде совпадает.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from .corpus import Chunk, Corpus

SYSTEM_PROMPT = """Ты — Sentinel Copilot, локальный ассистент проверяющего банковской кассы (ForteBank).
Правила ответа:
1. Отвечай по-русски, кратко и по делу (не более 6 предложений или короткий список).
2. Используй ТОЛЬКО два источника: «РЕЗУЛЬТАТ АНАЛИЗА ВИДЕО» и «НОРМЫ» из контекста. Ничего не добавляй от себя.
3. Ссылайся на нормы только маркерами вида [N1], [N2] из раздела «НОРМЫ». Не придумывай документы, пункты, статьи и ссылки.
4. На нарушения из результата ссылайся маркерами [V1], [V2]; таймкоды бери только из результата.
5. Описывай только наблюдаемое на видео. Сумму, подлинность денег, умысел, личность людей, наличие допуска,
   содержание разговоров по видео установить нельзя — так и скажи: «По видео это установить нельзя».
6. Если нужной нормы нет в разделе «НОРМЫ», ответь: «В источниках нет информации об этом».
7. Говори «признаки возможного нарушения», а не «нарушение доказано». Ответ — подсказка, не юридическое заключение.
8. Людей не идентифицируй; роли (кассир, посетитель, инкассатор) — только как в результате анализа."""

REFUSAL_VIDEO = "По видео это установить нельзя"
REFUSAL_SOURCES = "В источниках нет информации об этом"
REFUSAL_SCOPE = "Я отвечаю только на вопросы по результату анализа видео и нормам"


def fmt_tc(sec: Any) -> str:
    """Таймкод мм:сс (чч:мм:сс для длинных)."""
    try:
        s = max(0.0, float(sec))
    except (TypeError, ValueError):
        return "—"
    total = int(round(s))
    h, rem = divmod(total, 3600)
    m, ss = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{ss:02d}" if h else f"{m:02d}:{ss:02d}"


def _to_dict(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_dict(v) for v in obj]
    return obj


def normalize_result(job_result: Any) -> dict[str, Any]:
    """dict (схема API) или AnalysisResult -> dict без base64-кадров (они не нужны LLM)."""
    data = _to_dict(job_result) if job_result is not None else {}
    if not isinstance(data, dict):
        data = {}
    out = dict(data)
    viols: list[dict[str, Any]] = []
    for i, v in enumerate(data.get("violations") or []):
        if not isinstance(v, dict):
            continue
        vv = dict(v)
        frames = []
        for f in vv.get("evidence_frames") or []:
            if isinstance(f, dict):
                frames.append({"timestamp_sec": f.get("timestamp_sec")})
        vv["evidence_frames"] = frames
        vv.setdefault("type_id", "")
        vv["_n"] = i + 1
        viols.append(vv)
    out["violations"] = viols
    return out


def result_times(result: dict[str, Any]) -> list[float]:
    """Все таймкоды, известные из результата (для проверки таймкодов в ответе)."""
    times: list[float] = []
    for v in result.get("violations") or []:
        for key in ("timecode_sec", "t_start", "t_end"):
            try:
                if v.get(key) is not None:
                    times.append(float(v[key]))
            except (TypeError, ValueError):
                pass
        for f in v.get("evidence_frames") or []:
            try:
                times.append(float(f.get("timestamp_sec")))
            except (TypeError, ValueError):
                pass
    meta = result.get("meta") or {}
    for key in ("duration", "duration_s", "video_duration"):
        try:
            if meta.get(key) is not None:
                times.append(float(meta[key]))
        except (TypeError, ValueError, AttributeError):
            pass
    return times


@dataclass
class NormRef:
    n: int
    chunk: Chunk
    score: float = 0.0

    @property
    def marker(self) -> str:
        return f"[N{self.n}]"


def verdict_text(result: dict[str, Any]) -> str:
    verdict = result.get("verdict")
    status = result.get("status")
    if status == "processing":
        return "анализ ещё выполняется"
    if status == "failed":
        return "анализ завершился с ошибкой"
    if verdict == "violation":
        return "обнаружены признаки возможного нарушения"
    if verdict == "no_violation":
        return "признаков нарушения не обнаружено"
    return "вердикт отсутствует"


def unobservable_items(result: dict[str, Any], corpus: Optional[Corpus]) -> list[str]:
    items: list[str] = []
    if corpus is not None:
        for v in result.get("violations") or []:
            for it in corpus.type_info(str(v.get("type_id") or "")).get("not_observable") or []:
                if it not in items:
                    items.append(str(it))
    for it in ("сумма и подлинность денег", "умысел", "личность людей", "наличие допуска", "содержание разговоров"):
        if it not in items:
            items.append(it)
    return items


def build_context(result: dict[str, Any], refs: Sequence[NormRef], corpus: Optional[Corpus] = None) -> str:
    lines = [f"РЕЗУЛЬТАТ АНАЛИЗА ВИДЕО (задача {result.get('job_id', '—')}):",
             f"Вердикт: {result.get('verdict') or '—'} ({verdict_text(result)})"]
    meta = result.get("meta") or {}
    dur = meta.get("duration") or meta.get("duration_s") if isinstance(meta, dict) else None
    if dur:
        lines.append(f"Длительность видео: {float(dur):.0f} с")
    viols = result.get("violations") or []
    if viols:
        lines.append(f"Возможные нарушения ({len(viols)}):")
        for v in viols:
            n = v.get("_n")
            tid = v.get("type_id") or ""
            name = v.get("type") or (corpus.type_name(tid) if corpus and tid else tid) or "—"
            conf = v.get("confidence")
            part = [f"[V{n}] Тип: {name}" + (f" ({tid})" if tid else ""),
                    f"таймкод {fmt_tc(v.get('timecode_sec'))} ({float(v.get('timecode_sec') or 0):.1f} с)"]
            if v.get("t_end"):
                part.append(f"интервал {fmt_tc(v.get('t_start'))}–{fmt_tc(v.get('t_end'))}")
            if conf is not None and conf != "":
                part.append(f"уверенность {float(conf):.2f}")
            lines.append("; ".join(part))
            if v.get("description"):
                lines.append(f"    Что видно: {v['description']}")
            frames = [fmt_tc(f.get("timestamp_sec")) for f in v.get("evidence_frames") or []]
            if frames:
                lines.append(f"    Кадры-доказательства: {', '.join(frames)}")
            own = [r.marker for r in refs if tid and tid in r.chunk.type_ids]
            if own:
                lines.append(f"    Относящиеся нормы: {', '.join(own)}")
    else:
        lines.append("Возможные нарушения: нет")
    lines.append("По видео НЕ устанавливается: " + "; ".join(unobservable_items(result, corpus)))
    lines.append("")
    if refs:
        lines.append("НОРМЫ (ссылайся только маркерами из этого списка):")
        for r in refs:
            c = r.chunk
            lines.append(f"{r.marker} {c.document}" + (f", {c.clause}" if c.clause else "") + (f" — {c.url}" if c.url else ""))
            lines.append(f"    «{c.text}»")
    else:
        lines.append("НОРМЫ: в источниках не найдено подходящих фрагментов.")
    return "\n".join(lines)


def build_messages(result: dict[str, Any], refs: Sequence[NormRef], corpus: Optional[Corpus],
                   question: str, history: Sequence[dict[str, str]] = ()) -> list[dict[str, str]]:
    msgs: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT + "\n\n" + build_context(result, refs, corpus)}]
    for turn in history:
        role = turn.get("role")
        content = str(turn.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            msgs.append({"role": role, "content": content})
    msgs.append({"role": "user", "content": question.strip()})
    return msgs
