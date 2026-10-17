"""«Заключение для проверяющего» в markdown — детерминированно, без LLM (работает и без модели).

Нормы берутся из результата (A12) и/или каталога по type_id и перепроверяются по каталогу/корпусу
(правило 5): неподтверждённые не выводятся. Описания — только наблюдаемые факты из результата.
"""
from __future__ import annotations

from typing import Any, Optional

from .config import default_config
from .corpus import Corpus, load_corpus
from .prompts import fmt_tc, normalize_result, unobservable_items, verdict_text


def _md(s: Any) -> str:
    return str(s or "").replace("\n", " ").replace("|", "\\|").strip()


def _verified_rules(v: dict[str, Any], corpus: Optional[Corpus]) -> tuple[list[dict[str, Any]], int]:
    """(подтверждённые нормы, сколько отброшено)."""
    out: list[dict[str, Any]] = []
    dropped = 0
    seen: set[tuple[str, str]] = set()
    for r in v.get("rules") or []:
        if not isinstance(r, dict):
            continue
        hit = corpus.find(r.get("document"), r.get("clause"), r.get("url")) if corpus else None
        if hit is None:
            dropped += 1
            continue
        if hit.key not in seen:
            seen.add(hit.key)
            # статус берём из каталога (корпуса), а не со слов результата
            out.append({"document": hit.document, "clause": hit.clause, "url": hit.url,
                        "quote": hit.text, "verified": hit.verified, "status": hit.status, "note": hit.note})
    if not out and corpus is not None and v.get("type_id"):
        for c in corpus.rules_for_type(str(v["type_id"])):
            if c.key not in seen:
                seen.add(c.key)
                out.append({"document": c.document, "clause": c.clause, "url": c.url, "quote": c.text,
                            "verified": c.verified, "status": c.status, "note": c.note})
    return out, dropped


def report(job_result: Any, corpus: Optional[Corpus] = None, cfg: Optional[dict[str, Any]] = None) -> str:
    cfg = cfg or default_config()
    if corpus is None:
        try:
            from .service import get_service

            corpus = get_service().corpus
        except Exception:
            corpus = load_corpus(cfg)
    res = normalize_result(job_result)
    disclaimer = str(cfg.get("report", {}).get("disclaimer") or "Подсказка, не юридическое заключение.").strip()
    viols = res.get("violations") or []
    meta = res.get("meta") or {}
    lines = ["# Заключение для проверяющего", "",
             f"> ⚠️ {disclaimer}", "",
             f"- **Задача:** `{_md(res.get('job_id') or '—')}`",
             f"- **Статус анализа:** {_md(res.get('status') or '—')}",
             f"- **Вердикт системы:** {verdict_text(res)}"]
    if isinstance(meta, dict):
        if meta.get("duration") or meta.get("duration_s"):
            lines.append(f"- **Длительность видео:** {float(meta.get('duration') or meta.get('duration_s')):.0f} с")
        if meta.get("version"):
            lines.append(f"- **Версия системы:** {_md(meta['version'])}")
    lines.append("")

    if res.get("status") in {"processing", "failed"}:
        lines += ["## Результат", "", f"Нет данных: {verdict_text(res)}.", ""]
    elif not viols:
        lines += ["## Возможные нарушения", "",
                  "Признаков нарушения не обнаружено. Автоматический анализ может пропустить нарушение — "
                  "рекомендуется выборочный просмотр видео.", ""]
    else:
        lines += [f"## Возможные нарушения ({len(viols)})", ""]
        for v in viols:
            tid = v.get("type_id") or ""
            name = v.get("type") or (corpus.type_name(tid) if corpus and tid else "") or "Возможное нарушение"
            lines.append(f"### {v.get('_n')}. {_md(name)} — {fmt_tc(v.get('timecode_sec'))}")
            lines.append("")
            if tid:
                lines.append(f"- **Код типа:** {_md(tid)}")
            if v.get("t_end"):
                lines.append(f"- **Интервал:** {fmt_tc(v.get('t_start'))}–{fmt_tc(v.get('t_end'))}")
            if v.get("confidence") not in (None, ""):
                lines.append(f"- **Уверенность алгоритма:** {float(v['confidence']):.2f} (не вероятность нарушения)")
            lines.append(f"- **Что видно:** {_md(v.get('description')) or 'нет описания'}")
            frames = [fmt_tc(f.get("timestamp_sec")) for f in v.get("evidence_frames") or []]
            lines.append(f"- **Кадры-доказательства:** {', '.join(frames) if frames else 'нет'}")
            rules, dropped = _verified_rules(v, corpus)
            if rules:
                lines.append("- **Нормы (из каталога):**")
                for r in rules:
                    ref = f"{_md(r['document'])}" + (f", {_md(r['clause'])}" if r.get("clause") else "")
                    link = f"[{ref}]({r['url']})" if r.get("url") else ref
                    tags = []
                    if r.get("status") == "candidate":
                        tags.append("норма-кандидат")
                    if not r.get("verified"):
                        tags.append("цитата требует сверки с первоисточником")
                    mark = f" _({'; '.join(tags)})_" if tags else ""
                    quote = f" — «{_md(r['quote'])}»" if r.get("quote") else ""
                    lines.append(f"  - {link}{quote}{mark}")
            else:
                lines.append("- **Норма:** без пункта — в каталоге нет подтверждённого пункта для этого типа")
            info = corpus.type_info(tid) if corpus is not None and tid else {}
            if info.get("legal_note"):
                lines.append(f"- **Пояснение каталога (не текст нормы):** {_md(info['legal_note'])}")
            excl = [str(x) for x in info.get("exclusions") or [] if str(x).strip()]
            if excl:
                lines.append("- **Когда это не нарушение (каталог):** " + "; ".join(_md(x) for x in excl))
            if dropped:
                lines.append(f"- _Отброшено ссылок, не найденных в каталоге норм: {dropped}_")
            lines.append("")

    lines += ["## Что проверить вручную", ""]
    for v in viols:
        t0 = max(0.0, float(v.get("t_start") or v.get("timecode_sec") or 0) - 5)
        t1 = float(v.get("t_end") or v.get("timecode_sec") or 0) + 5
        lines.append(f"- [ ] Просмотреть фрагмент {fmt_tc(t0)}–{fmt_tc(t1)} (нарушение №{v.get('_n')}) и подтвердить/отклонить.")
    lines.append("- [ ] По видео не устанавливается (проверить по документам): " + "; ".join(unobservable_items(res, corpus)) + ".")
    lines += ["", "## Ограничения", "",
              "- Описания содержат только наблюдаемые на видео факты; умысел, суммы и подлинность не оцениваются.",
              "- Люди не идентифицируются; роли определены по зоне, поведению и предметам.",
              "- Ссылки на нормы приводятся только из каталога норм системы."]
    if corpus is not None and corpus.is_fixture:
        lines.append("- ⚠️ Каталог норм РК недоступен: показаны ТЕСТОВЫЕ нормы-фикстуры, не использовать.")
    return "\n".join(lines) + "\n"
