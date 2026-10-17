"""«Заключение для проверяющего» в markdown — детерминированно, без LLM (работает и без модели).

Нормы берутся из результата (A12) и/или каталога по type_id и перепроверяются по каталогу/корпусу
(правило 5): неподтверждённые не выводятся. Описания — только наблюдаемые факты из результата.
"""
from __future__ import annotations

from typing import Any, Optional

from .config import default_config
from .corpus import Corpus, load_corpus
from .prompts import (completeness, completeness_text, explain_for, explain_lines, fmt_tc, normalize_result,
                      review_hints, unobservable_items, verdict_text)


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


def report(job_result: Any, corpus: Optional[Corpus] = None, cfg: Optional[dict[str, Any]] = None,
           fmt: str = "md") -> str:
    """Заключение для проверяющего: ``fmt="md"`` (по умолчанию) или ``fmt="html"`` (автономный файл)."""
    cfg = cfg or default_config()
    html_mode = str(fmt).lower() == "html"
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
             f"- **Вердикт системы:** {verdict_text(res)}",
             f"- **Полнота анализа:** {completeness_text(res)}"]
    if isinstance(meta, dict):
        if meta.get("duration") or meta.get("duration_s"):
            lines.append(f"- **Длительность видео:** {float(meta.get('duration') or meta.get('duration_s')):.0f} с")
        if meta.get("version"):
            lines.append(f"- **Версия системы:** {_md(meta['version'])}")
    lines.append("")

    comp = completeness(res)
    if comp["state"] == "incomplete" and res.get("status") != "failed":
        lines += ["> ⚠️ **Анализ неполный.** Отсутствие найденных нарушений в непроанализированной части "
                  "не означает, что их нет. Просмотрите видео вручную.", ""]
    if res.get("status") in {"processing", "failed"}:
        lines += ["## Результат", "", f"Нет данных: {verdict_text(res)}.", ""]
    elif not viols:
        lines += ["## Возможные нарушения", "",
                  ("Нарушений в проанализированной части не найдено — анализ неполный, вывод «нарушений нет» "
                   "сделать нельзя." if comp["state"] == "incomplete" else
                   "Признаков нарушения не обнаружено. Автоматический анализ может пропустить нарушение — "
                   "рекомендуется выборочный просмотр видео."), ""]
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
            why = explain_lines(explain_for(res, v))
            if why:
                lines.append("- **Почему отмечено (данные системы):**")
                lines.extend(f"  - {_md(x)}" for x in why)
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
            if html_mode:
                lines.append(f"@@FRAMES:{v.get('_n')}@@")
            lines.append("")

    hints = review_hints(res)
    if hints:
        lines += ["## Спорные места (не вошли в вердикт)", ""]
        for h in hints:
            reasons = ", ".join(_md(x) for x in h.get("reasons_ru") or []) or "ниже порога"
            lines.append(f"- {_md(h.get('type') or h.get('type_id') or 'кандидат')} — {fmt_tc(h.get('timecode_sec'))}: {reasons}")
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
    md = "\n".join(lines) + "\n"
    if not html_mode:
        return md
    rcfg = cfg.get("report", {})
    frames = _evidence_frames(job_result, int(rcfg.get("html_max_frames", 3)),
                              int(rcfg.get("html_max_frame_kb", 600))) if rcfg.get("html_include_frames", True) else {}
    return render_html(md, title=f"Заключение для проверяющего — {res.get('job_id') or ''}".strip(" —"), frames=frames)


# ----------------------------------------------------------------------------- HTML

_CSS = """body{font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;color:#1d2433;background:#fff;margin:0}
main{max-width:860px;margin:0 auto;padding:24px 16px}h1{font-size:24px;margin:0 0 12px}h2{font-size:19px;margin:24px 0 8px;
border-bottom:1px solid #e3e6ec;padding-bottom:4px}h3{font-size:16px;margin:18px 0 6px}blockquote{margin:12px 0;padding:10px 14px;
background:#fff6e5;border-left:4px solid #e0a100}ul{padding-left:22px}code{background:#f2f4f7;padding:0 4px;border-radius:3px}
figure{display:inline-block;margin:6px 8px 6px 0}figure img{max-width:260px;border:1px solid #d0d5dd;border-radius:4px}
figcaption{font-size:12px;color:#5b6475}a{color:#1a56db}@media print{a{color:#000}figure img{max-width:200px}}"""


def _inline(text: str) -> str:
    import html
    import re as _re

    s = html.escape(text, quote=True)

    def _link(m: "_re.Match[str]") -> str:
        label, url = m.group(1), m.group(2)
        if _re.match(r"^https?://", html.unescape(url)):
            return f'<a href="{url}" rel="noopener noreferrer">{label}</a>'
        return label

    s = _re.sub(r"\[([^\]]{1,300})\]\(([^)\s]{1,500})\)", _link, s)
    s = _re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = _re.sub(r"(?<![\w/])_(.+?)_(?![\w/])", r"<em>\1</em>", s)
    s = _re.sub(r"`([^`]{1,200})`", r"<code>\1</code>", s)
    return s


def render_html(md: str, title: str = "Заключение для проверяющего",
                frames: Optional[dict[int, list[tuple[float, str]]]] = None) -> str:
    """Подмножество markdown отчёта -> безопасный HTML (всё экранировано, скрипты запрещены CSP)."""
    import html
    import re as _re

    out: list[str] = []
    depth = 0

    def set_depth(d: int) -> None:
        nonlocal depth
        while depth < d:
            out.append("<ul>")
            depth += 1
        while depth > d:
            out.append("</ul>")
            depth -= 1

    for raw in md.splitlines():
        m_fr = _re.match(r"^@@FRAMES:(\d+)@@$", raw.strip())
        if m_fr:
            set_depth(0)
            for t, b64 in (frames or {}).get(int(m_fr.group(1)), []):
                cap = fmt_tc(t)
                out.append(f'<figure><img src="data:image/jpeg;base64,{b64}" alt="кадр {cap}">'
                           f"<figcaption>кадр {cap}</figcaption></figure>")
            continue
        h = _re.match(r"^(#{1,6})\s+(.*)$", raw)
        if h:
            set_depth(0)
            n = len(h.group(1))
            out.append(f"<h{n}>{_inline(h.group(2))}</h{n}>")
            continue
        if raw.startswith(">"):
            set_depth(0)
            out.append(f"<blockquote>{_inline(raw.lstrip('> ').strip())}</blockquote>")
            continue
        b = _re.match(r"^(\s*)- (.*)$", raw)
        if b:
            set_depth(len(b.group(1)) // 2 + 1)
            item = b.group(2)
            if item.startswith("[ ] "):
                item = "☐ " + item[4:]
            out.append(f"<li>{_inline(item)}</li>")
            continue
        set_depth(0)
        if raw.strip():
            out.append(f"<p>{_inline(raw.strip())}</p>")
    set_depth(0)
    csp = "default-src 'none'; img-src data:; style-src 'unsafe-inline'"
    return ("<!doctype html>\n<html lang=\"ru\"><head><meta charset=\"utf-8\">"
            f'<meta http-equiv="Content-Security-Policy" content="{csp}">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{html.escape(title)}</title><style>{_CSS}</style></head>"
            f"<body><main>\n" + "\n".join(out) + "\n</main></body></html>\n")


def _evidence_frames(job_result: Any, max_frames: int, max_kb: int) -> dict[int, list[tuple[float, str]]]:
    """Кадры-доказательства из исходного результата: только валидный base64 JPEG, ограничение размера."""
    import base64
    import binascii
    import re as _re

    from .prompts import _to_dict, to_float

    data = job_result
    if isinstance(data, (str, bytes)):
        try:
            import json

            data = json.loads(data)
        except ValueError:
            return {}
    data = _to_dict(data)
    out: dict[int, list[tuple[float, str]]] = {}
    viols = data.get("violations") if isinstance(data, dict) else None
    n = 0
    for v in viols if isinstance(viols, list) else []:
        if not isinstance(v, dict):
            continue
        n += 1
        for f in (v.get("evidence_frames") or [])[:max(0, max_frames)]:
            b64 = str((f or {}).get("image_base64") or "") if isinstance(f, dict) else ""
            if b64.startswith("data:"):
                b64 = b64.split(",", 1)[-1]
            if not b64 or len(b64) > max_kb * 1366 or not _re.fullmatch(r"[A-Za-z0-9+/=\s]+", b64):
                continue
            try:
                head = base64.b64decode(b64[:64] + "=" * (-len(b64[:64]) % 4), validate=False)
            except (binascii.Error, ValueError):
                continue
            if not head.startswith(b"\xff\xd8\xff"):
                continue
            out.setdefault(n, []).append((to_float(f.get("timestamp_sec"), 0.0) or 0.0, _re.sub(r"\s+", "", b64)))
    return out
