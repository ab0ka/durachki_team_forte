# Migration note: contracts v1 → v1.1-additive (A01)

**Ломающих изменений нет.** `contracts/sentinel_contracts.py` байт в байт совпадает с замороженным v1 (sha256 проверяется тестом). Все dataclass-ы, сигнатуры и `LABELS`/`SIGNALS` прежние.

## Что добавлено

1. **Два сериализатора вместо одного** (`contracts/serialize.py`):
   * `result_to_jury_api(result)` — ровно обязательная схема ТЗ, без дополнительных полей. Норма из `Violation.rules[0]` дописывается в `description`, если её URL там ещё нет.
   * `result_to_ui_report(result)` — прежний расширенный формат («ТЗ + доп. поля»), плюс `verdict_state`, `analysis_complete`, `candidates`, `warnings`.
   * `result_to_api(result, *, profile="jury")` — имя из контракта сохранено. **По умолчанию теперь строгий профиль жюри**; прежний расширенный вывод — `profile="ui"`.
2. **Политика вердикта** (`resolve_outcome`, общая для обоих сериализаторов):
   * у `processing` и `failed` вердикт всегда `null`;
   * у `done` без вердикта и без нарушений ответ теперь `failed`, раньше выводился `no_violation`;
   * `done/no_violation` при `meta.completeness` «неполный анализ» → `failed`;
   * `violation` без нарушения с конечным таймкодом → `failed`;
   * при вердикте `no_violation` нарушения жюри не отдаются; в UI-report они лежат в `candidates`.
3. **Проверки** (`contracts/validate.py`):
   * `jury_response_errors()` — строгая схема плюс согласованность verdict/violations, конечные таймкоды (не больше длительности), валидный JPEG (FF D8 FF … FF D9) и не больше 3 кадров;
   * `job_response_errors(strict=True)` — проверка без лишних полей.
4. **Совместимые дополнения** (`contracts/extensions.py`, только в `AnalysisResult.meta`, у всего есть значения по умолчанию):

   | Ключ `meta` | Тип | Если ключа нет |
   |---|---|---|
   | `completeness` | `Completeness.to_dict()` | «нет сведений», анализ считается полным (совместимость) |
   | `provenance` | `Provenance.to_dict()` | — |
   | `duration_sec` | float | таймкоды не обрезаются по длительности |
   | `warnings` | list[str] | [] |
   | `violation_extra` | list[dict], по индексу нарушения | — |

   `verdict_state` бывает `violation`, `no_violation`, `unknown` или `pending`. Для неизвестного типа нарушения используйте `type_id="V99"`.

## Что сделать модулям

| Кто | Действие |
|---|---|
| A02 API | Маршрут `GET /api/v1/jobs/{id}` отдаёт `result_to_jury_api(result)`. UI-данные — на отдельном маршруте (например `/report`) через `result_to_ui_report`. |
| A03 pipeline | Заполнять `meta.completeness` (с `required_stages`), `meta.duration_sec` и `meta.provenance`. При исключении/таймауте ядра — `status="failed"` или `completeness.complete=False`. |
| A12 fusion | Ставить `verdict` явно; кадры — JPEG base64 без `data:`; `rules` только из `norms/catalog.yaml`. |
| A15 UI | Читать расширенные поля из UI-report, а не из ответа жюри. |
| A16 eval | Проверять ответы через `jury_response_errors`. |
