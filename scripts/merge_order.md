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
   * найти оставшиеся: `git grep -n "_contract_fallback"`;
   * сверить копии с каноном и удалить совпадающие: `python scripts/check_fallbacks.py` (отчёт), затем `--drop`. Скрипт падает (код 1) и на «висячем» импорте (`DANGLING`): копии уже нет, а модуль импортирует её без `try/except`. В `integration-cloud@2055fc4` копий нет, висячих импортов нет.
4. Прогнать тесты модуля: `pytest <папка> -m "not gpu and not slow"`.
5. Прогнать сторожей:
   * `python scripts/check_licenses.py` — лицензии моделей, облачные API;
   * `python scripts/guard_media.py --all` — медиа, веса, файлы > 5 МБ;
   * `python scripts/resolve_deps.py --lock-ref origin/data/labels:tools/dataprep/requirements-l1.lock.txt` — зависимости против lock L1.
6. Прогнать сквозной тест строго:
   * контракт стадий: `SENTINEL_MOCK=1 python -m testkit.e2e_stub --no-backend --strict-stages` и то же с `--suite spec`;
   * полная сборка (оркестратор + сериализатор жюри): `make e2e-full` — явный mock → `done`, без mock → `failed`. Код 3 («NOT RUN») — оркестратора нет, сборка не проверена.
7. Перегенерировать матрицу совместимости: `python scripts/compat_matrix.py --tree .` — в корне интеграционной ветки, где все модули уже на месте.

## 3. Точки стыковки, которые легко пропустить

* **A02:** `GET /api/v1/jobs/{id}` отдаёт `contracts.serialize.result_to_jury_api(result)` — строго поля ТЗ. Расширенный отчёт для UI — на отдельном маршруте через `result_to_ui_report`. Подробности — в `contracts/MIGRATION.md`.
* **A03:** заполняет `meta.completeness`, `meta.duration_sec`, `meta.provenance` (`contracts/extensions.py`). Без этого `no_violation` выдаётся только «по совместимости».
* **A12:** ставит `verdict` явно; кадры — JPEG base64 без `data:`; нормы только из `norms/catalog.yaml`.
* **Все:** имена файлов датасета `X-1` (латинская X) и `Х-3` (кириллическая Х) нормализует ingest или eval. Дубликаты Х-4≡Х-47 и Х-5≡Х-48 надо учитывать в LOO.
* **CI:** обязательные проверки — `core`, `modules` (матрица по присутствующим модулям), `e2e-stages`, `e2e-full` (только если в ветке есть оркестратор и модули критического пути; иначе пропущен). Необязательные — `frontend`, а `ownership` для PR без тега `[Axx]` только информирует. Глобального `SENTINEL_MOCK` нет.

## 4. Финальная проверка перед сдачей

```bash
make lint test-fast e2e
python scripts/check_licenses.py
python scripts/check_ownership.py --base origin/main --all-branches --report-only
```

## 5. Пробная интеграция (A01, 17.10, ~08:05 UTC; локально, ничего не пушилось)

> Исторический снимок до `integration-cloud`. Актуальное состояние и несовместимости — `contracts/COMPAT.md`, раздел 0.

Деревья всех 17 веток наложены во временную папку поверх ветки A01: агенты пишут в разные папки, так что это эквивалент мерджа. Окружение: CPU, Python 3.11, только `numpy`, `opencv-python-headless`, `PyYAML`, `jsonschema`, `pytest`; без GPU, весов и `requirements.txt` модулей.

**Конфликты файлов:**

* `.gitignore`, `contracts/__init__.py` и `contracts/sentinel_contracts.py` есть в 9 ветках, и везде они **побайтно совпадают** с A01.
* 4 ветки (A14, A15, A16, A17) содержат коммит A01 `ccf2923` как предка; остальные — несвязанные истории.
* Ни один из 525 файлов других веток не скрывается `.gitignore` A01.
* Запрещённых файлов (видео, веса, кадры) нет.

**Стыковка по контракту (`python -m testkit.e2e_stub --no-backend`):**

* все 12 стадий реальных модулей (ingest → detect → track → pose → pose_signals → scene → norms → rules → temporal → vlm verify/discover → fusion) прошли проверки `testkit.contract_checks`;
* ответ проходит строгую схему жюри — 8 сценариев по 20 с, ≈19 с на ролик (CPU, mock);
* вердикты в mock-режиме — `no_violation`: моки без весов нарушений не находят, это ожидаемо.

**Что требует внимания:**

| Где | Что | Чьё |
|---|---|---|
| `backend.pipeline.run` | На синтетическом ролике 6 с (mock, CPU) не вернул результат за 110 с при `deadline_s=40`; три ролика по 8 с не уложились в 300 с. Повтор: `python <скрипт с run(...)>` в корне интеграции (из stdin не запускать — spawn). | A03 |
| `backend/pipeline/tests/test_pipeline.py::test_partial_with_violation` | Ждёт вердикт `violation` при `status=processing`. Это противоречит правилу «у processing/failed нет бинарного вердикта» (`contracts/MIGRATION.md`). | A03 |
| `backend/pipeline/tests/test_cache_budget.py` | Предупреждения первого и второго (кэшированного) прогона различаются: «ml.ingest недоступен» только в первом. | A03 |
| `ml/fusion` (2 теста: `test_no_catalog_reports_with_warning`, `test_tune::test_cli_end_to_end`) | Отдельно A12 — 58/58, A12 + контракт A01 — 58/58, **A12 + `norms/` (A13) — 2 падения**: при `catalog={}` fusion, видимо, подхватывает реальный каталог или пороги. | A12 × A13 |
| `backend/api/results.py`, `ml/fusion/serialize.py` | Переключить ответ жюри на `result_to_jury_api` (см. `contracts/MIGRATION.md`). | A02, A12 |
| `backend/api`, `copilot` (нет `fastapi`), `ml/ingest` (нет `av`), `ml/vlm` (нет `PIL`), `eval` (нет `openpyxl`) | Падения из-за не установленных в пробном окружении зависимостей. В CI они ставятся через `scripts/ci_requirements.py`. | окружение |

Без замечаний: `ml/track` 35, `ml/pose` 39, `ml/scene` 34, `ml/rules` 118, `ml/temporal` 29, `norms` 36, `deploy` 38, `contracts` 88, `testkit` 109.
