# Migration note: contracts v1 → v1.1-additive (A01)

**Ломающих изменений нет.** `contracts/sentinel_contracts.py` байт в байт совпадает с замороженным v1 (sha256 проверяется тестом). Все dataclass-ы, сигнатуры и `LABELS`/`SIGNALS` прежние.

## Что добавлено

1. **Два сериализатора вместо одного** (`contracts/serialize.py`):
   * `result_to_jury_api(result)` — ровно обязательная схема ТЗ, без дополнительных полей. Норма из `Violation.rules[0]` дописывается в `description`, если её URL там ещё нет.
   * `result_to_ui_report(result)` — прежний расширенный формат («ТЗ + доп. поля»), плюс `verdict_state`, `analysis_complete`, `candidates`, `warnings`.
   * `result_to_api(result, *, profile="ui")` — имя **и поведение** из контракта сохранены («схема ТЗ + доп. поля» = UI-report). Пробная интеграция показала, что на это опираются A03 (кэш) и A16 (eval); при строгом выводе по умолчанию кэш A03 падал с `KeyError: 'meta'`. Ответ жюри — `result_to_jury_api` (или `profile="jury"`).
2. **Политика вердикта** (`resolve_outcome`, общая для обоих сериализаторов):
   * у `processing` и `failed` вердикт всегда `null`;
   * mock-результат (`meta.mock`, `provenance.mock` или стадия со статусом `mock`) без явного `SENTINEL_MOCK=1` → `failed`. Production сам на mock не переключается;
   * `done/no_violation` — **только при подтверждённой полноте `meta.coverage`**, иначе `failed` с причиной. Неполный или упавший анализ ≠ «нарушений нет». Нет `meta.coverage` → тоже `failed`;
   * `violation` без нарушения с конечным таймкодом → `failed`;
   * у `done` без вердикта и без нарушений ответ `failed`: `no_violation` не выдумывается;
   * при вердикте `no_violation` нарушения жюри не отдаются; в UI-report они лежат в `candidates`;
   * если `timecode_sec` не конечен, берётся `t_start` (начало действия), затем первый кадр.
3. **Проверки** (`contracts/validate.py`):
   * `jury_response_errors()` — строгая схема плюс согласованность verdict/violations, конечные таймкоды (не больше длительности), валидный JPEG (FF D8 FF … FF D9) и не больше 3 кадров;
   * `job_response_errors(strict=True)` — проверка без лишних полей.
4. **Совместимые дополнения** (`contracts/extensions.py`, только в `meta`/`signals`, у всего есть значения по умолчанию):

   | Ключ | Тип | Если нет |
   |---|---|---|
   | `meta.coverage` | `compute_coverage(...)` → `{ratio, analyzed_sec, duration_sec, unknown_spans, stages, required, complete}` или число 0..1 | **`no_violation` не выдаётся** (`failed`) |
   | `meta.mock` / `meta.provenance.mock` | bool | считается не mock |
   | `meta.complete`, `meta.stages`, `meta.required_stages` | как у A03 | учитываются в проверке полноты |
   | `meta.duration_sec` (или `meta.video.duration`) | float | таймкоды не обрезаются по длительности |
   | `meta.warnings` | list[str] | [] |
   | `meta.violation_extra` | list[dict], по индексу нарушения | — |

   * **Неизвестно ≠ нет:** если сигнал не наблюдался (не видно человека или объекта, темно, перекрыто), его нет в `SceneState.signals`. Значение `0.0` означает «наблюдали: нет». Для чтения есть `signal_value()`, при отсутствии сигнала она возвращает `None`.
   * **Реестр типов** `TYPE_REGISTRY`: V01–V09 и V99 (V09 — личные вещи на кассовом столе). Названия запасные, источник истины — `norms/catalog.yaml`.
   * **Доп. словарь** (v1.1-additive): `EXTRA_SIGNALS` = `personal_items_on_desk`, `scene_visibility`, `second_outsider`; `EXTRA_LABELS` = `personal_item`.
   * `meta.completeness` из первой версии фазы 1 заменён на `meta.coverage`; им никто не пользовался.
5. **Флаги** (`contracts/config.yaml`; гипотезы по умолчанию выключены до замера L2):
   * `jury.embed_rule_in_description: false` — дописывать первую норму в `description` ответа жюри (было включено в фазе 1);
   * `coverage.min_ratio_for_no_violation: 0.98` — инвариант, порог подбирает L2;
   * переопределение: `SENTINEL_CONTRACTS_CONFIG`.
6. **Воспроизводимость:**
   * `canonical_json()` и `fingerprint()` (sha256) — для сравнения ответов разных прогонов;
   * `python -m contracts.examples.build` пересобирает примеры, тест сверяет их байт в байт.

## Что сделать модулям

| Кто | Действие |
|---|---|
| A02 API | Маршрут `GET /api/v1/jobs/{id}` отдаёт `result_to_jury_api(result)`: сейчас `backend/api/results.py` вызывает `result_to_api`, это расширенный формат — нужна замена одной строки. UI-данные — на отдельном маршруте (например `/report`) через `result_to_ui_report`. |
| A12 fusion (`ml/fusion/serialize.py: result_to_jury`) | Вызывать `contracts.serialize.result_to_jury_api` вместо `result_to_api`: функция задумана как «строгий ответ жюри». |
| A03 pipeline | Заполнять `meta.coverage = compute_coverage(duration, analyzed_until, unknown_spans=..., stages=meta["stages"], required=meta["required_stages"])`. Без этого любой `no_violation` уйдёт жюри как `failed`. Сейчас A03 сам выводит `no_violation` при `complete=True` (`orchestrator.py`, финализация) — без `meta.coverage` сериализатор это не пропустит. |
| A12 fusion | Ставить `verdict` явно; кадры — JPEG base64 без `data:`; `rules` только из `norms/catalog.yaml`. |
| A15 UI | Читать расширенные поля из UI-report, а не из ответа жюри. |
| A16 eval | Проверять ответы через `jury_response_errors`. |
