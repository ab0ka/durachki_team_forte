"""Kassa Sentinel — eval (владелец A16).

Инструменты оценки «как у жюри»: разметка (labels), метрики (metrics), прогон папки
(run), 100-балльная оценка (score_like_jury), HTML-отчёт (report), стресс-тест ракурса
(robustness), имитация жюри по публичному URL (jury_sim), leave-one-group-out (loo).

Импорт пакета не тянет тяжёлых зависимостей и не имеет побочных эффектов.
"""
from __future__ import annotations

__all__ = ["labels", "metrics", "run", "score_like_jury", "report", "robustness", "jury_sim", "loo", "schema"]
