# contracts/ — общий контракт Kassa Sentinel (владелец A01)

| Файл | Что это |
|---|---|
| `sentinel_contracts.py` | **Замороженный контракт v1** (байт в байт с ТЗ). sha256 `3b870961e62dac86e9457029009a37a753c0fdc73543639690b0493d8e134cac` — проверяется тестом. |
| `__init__.py` | Реэкспорт dataclass-ов, `LABELS`, `SIGNALS`, `CONTRACT_VERSION`. |
| `extensions.py` | Совместимые дополнения v1.1 (только через `meta`/`extra`, со значениями по умолчанию): `completeness`, `provenance`, `verdict_state`/unknown. |
| `serialize.py` | Два сериализатора: `result_to_jury_api` (строго ТЗ) и `result_to_ui_report` (расширенный). `result_to_api` — прежнее имя = UI-report (`profile="jury"` — ответ жюри). Также `api_to_result`, `resolve_outcome`. |
| `api_schema.json` | JSON Schema (draft 2020-12) ответа `GET /api/v1/jobs/{id}`; `$defs/AnalyzeResponse` — ответ `POST /analyze`. |
| `validate.py` | Валидатор: обычный, строгий для жюри (`jury_response_errors`), смысловые проверки; CLI `python -m contracts.validate`. |
| `examples/*.json` | Примеры ответов: `job_*`/`jury_*` — строгие ответы жюри, `job_done_full`/`ui_*` — UI-report. |
| `MIGRATION.md` | Что изменилось относительно исходного описания и что сделать модулям. |

**Канонический коммит контракта:** ветка `claude/gifted-bardeen-i404eh`, коммит `ccf2923`. В нём `sentinel_contracts.py`, `__init__.py` и `.gitignore`.

## Как подключиться (для модулей A02–A18)

Пока ветка A01 не смерджена, используйте fallback-копию (как в задании):

```python
try:
    from contracts.sentinel_contracts import *  # noqa
except ImportError:  # пока ветка A01 не смерджена
    from ._contract_fallback import *  # noqa
```

После мерджа A01:

1. Импорт `from contracts.sentinel_contracts import *` начинает работать сам — код менять не нужно.
2. Интегратор удаляет `<модуль>/_contract_fallback.py` и ветку `except ImportError` (см. `scripts/merge_order.md`).
3. Запускайте тесты из корня: `pytest <ваша_папка>` (`pythonpath=["."]` задан в `pyproject.toml`).

### API (A02)

```python
from contracts.serialize import result_to_jury_api, result_to_ui_report
from contracts.validate import jury_response_errors

# GET /api/v1/jobs/{job_id} — ответ жюри: ТОЛЬКО поля ТЗ
body = result_to_jury_api(result)
assert not jury_response_errors(body)          # можно держать в тестах API

# UI/отладка — отдельный маршрут, например GET /api/v1/jobs/{job_id}/report
report = result_to_ui_report(result)           # + type_id, confidence, rules, rule, timeline, meta, progress, stage, ...
```

`result_to_api(result)` (имя из контракта) сохраняет прежнее поведение «схема ТЗ + доп. поля», т.е. это UI-report. На него уже опираются A03 (кэш читает `meta`/`timeline`) и A16 (`type_id`/`confidence`). Строгий ответ жюри — **только** `result_to_jury_api` (или `profile="jury"`).

### Пайплайн / fusion (A03, A12)

* Ставьте `verdict` явно. Сериализатор **не придумывает** `no_violation`. Если вердикта нет и нет нарушений, ответ будет `failed`.
* Сообщайте полноту анализа: `meta["completeness"] = Completeness(complete=..., stages=..., required_stages=[...]).to_dict()`. При неполном анализе `no_violation` не отдаётся (будет `failed`). Нарушения с доказательствами отдаются.
* Кладите `meta["duration_sec"]`: по нему таймкоды проверяются и обрезаются по длительности.
* Происхождение результата — `meta["provenance"] = Provenance(...).to_dict()`. Сведения по нарушению — `meta["violation_extra"][i]` (попадают только в UI-report).
* Кадры — JPEG в base64 без префикса `data:`. Невалидные кадры отбрасываются с предупреждением.

## Тесты

```bash
pytest contracts -q
python -m contracts.validate --jury contracts/examples/job_done_tz.json
```
