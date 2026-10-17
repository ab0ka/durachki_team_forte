# Порядок мерджа 18 веток и чек-лист интеграции (A01)

## 0. Исходное состояние истории

* В начале работы репозиторий был пуст. A01 создал `main` как **пустой корневой коммит** `02b6e4d`, чтобы было куда открывать PR. Ветка A01 растёт из него.
* **Канонический контракт:** ветка `claude/gifted-bardeen-i404eh`, коммит `ccf2923` (`contracts/sentinel_contracts.py`, `contracts/__init__.py`, `.gitignore`). Предложение: сделать его общей базой. Варианты: смерджить PR A01 первым или fast-forward `main` до этого коммита.
* У остальных агентов могут быть собственные initial commits (несвязанные истории). GitHub не откроет PR между несвязанными историями — их вливает интегратор локально:

```bash
git fetch origin
git checkout main
git merge --no-ff --allow-unrelated-histories origin/<ветка-агента> -m "Merge [Axx] <модуль>"
```

Перезапускать агентов не нужно: каждый писал только в свою папку, поэтому конфликтов почти не будет. Исключение — пустые `ml/__init__.py` и `backend/__init__.py`: при add/add-конфликте берите любую сторону, файлы пустые.

## 1. Рекомендуемый порядок

| # | Агент | Папка | Почему здесь |
|---|---|---|---|
| 1 | A01 | contracts/, testkit/, scripts/, .github/, корень | контракт, конфиги pytest/ruff, CI, тест-кит |
| 2 | A13 | norms/ | `norms.load_catalog()` нужен правилам и fusion |
| 3 | A04 | ml/ingest/ | `load_video` — вход всех CV-стадий |
| 4 | A05, A06, A07, A08 | ml/detect, ml/track, ml/pose, ml/scene | восприятие (зависят только от контракта и ingest) |
| 5 | A09, A10, A11 | ml/rules, ml/temporal, ml/vlm | события |
| 6 | A12 | ml/fusion/ | вердикт, доказательства, описания |
| 7 | A03 | backend/pipeline/ | оркестратор всего выше (моки — для отсутствующих) |
| 8 | A02 | backend/api/ | API жюри поверх `backend.pipeline.run` |
| 9 | A15 | frontend/ | UI поверх API |
| 10 | A14 | copilot/ | ассистент проверяющего |
| 11 | A16 | eval/ | метрики и симулятор жюри (нужен весь пайплайн) |
| 12 | A17 | deploy/ | Docker, офлайн-веса, Windows, Tunnel |
| 13 | A18 | docs/, pitch/ | документы для сдачи (последними, с актуальными цифрами) |

## 2. Чек-лист на каждую ветку

1. Проверить владение и запрещённые файлы:
   `python scripts/check_ownership.py --base origin/main --head origin/<ветка> --agent Axx`.
   Чужие файлы не вливать, а вернуть автору или перенести в его папку.
2. Слить ветку (см. команду выше).
3. Удалить fallback-копию контракта, когда A01 уже в `main`:
   * удалить `<модуль>/_contract_fallback.py`;
   * заменить в коде блок `try: from contracts.sentinel_contracts import * / except ImportError: from ._contract_fallback import *` на `from contracts.sentinel_contracts import *  # noqa`;
   * найти оставшиеся: `git grep -n "_contract_fallback"`.
4. Прогнать тесты модуля: `pytest <папка> -m "not gpu and not slow"`.
5. Прогнать сторож правил: `python scripts/check_licenses.py` (лицензии моделей, облачные API, видео/веса в git).
6. Прогнать сквозной mock-тест: `python -m testkit.e2e_stub` (схема ответа жюри обязательна).

## 3. Точки стыковки, которые легко пропустить

* **A02:** `GET /api/v1/jobs/{id}` отдаёт `contracts.serialize.result_to_jury_api(result)` — строго поля ТЗ. Расширенный отчёт для UI — на отдельном маршруте через `result_to_ui_report`. Подробности — в `contracts/MIGRATION.md`.
* **A03:** заполняет `meta.completeness`, `meta.duration_sec`, `meta.provenance` (`contracts/extensions.py`). Без этого `no_violation` выдаётся только «по совместимости».
* **A12:** ставит `verdict` явно; кадры — JPEG base64 без `data:`; нормы только из `norms/catalog.yaml`.
* **Все:** имена файлов датасета `X-1` (латинская X) и `Х-3` (кириллическая Х) нормализует ingest или eval. Дубликаты Х-4≡Х-47 и Х-5≡Х-48 надо учитывать в LOO.
* **CI:** обязательные проверки — `core`, `modules` (матрица), `e2e`. Необязательные — `frontend`, а `ownership` для PR без тега `[Axx]` только информирует.

## 4. Финальная проверка перед сдачей

```bash
make lint test-fast e2e
python scripts/check_licenses.py
python scripts/check_ownership.py --base origin/main --all-branches --report-only
```
