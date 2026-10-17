# Совместимость веток с контрактом v1

<!-- notes:begin (ручной раздел, сохраняется при перегенерации) -->
## 0. Итог и записки владельцам

Снимок: HEAD веток — в разделе 3. Перегенерация: `python scripts/compat_matrix.py --tree <наложенные деревья всех веток>`; команда сборки дерева — в docstring скрипта.

**Критический путь** (ingest → detect → track → scene → rules → fusion, `norms.load_catalog`, `backend.pipeline.run`): **необъяснённых несовместимостей сигнатур нет**.
- Все 14 функций контракта совместимы с позиционным вызовом оркестратора (A03 вызывает стадии через `getattr`), имена и умолчания совпадают (раздел 1).
- Все 14 fallback-копий контракта побайтно совпадают с каноном (`scripts/check_fallbacks.py`).
- Дублей классов контракта нет.

Ниже — расхождения **поведения** и точные записки владельцам, без массового рефакторинга.

| Владелец | Где | Что сделать | Почему | Срочность |
|---|---|---|---|---|
| A03 | `backend/pipeline/orchestrator.py`, финализация (`result.verdict = "violation" if result.violations else "no_violation"`) | Писать `meta["coverage"] = contracts.extensions.compute_coverage(duration, analyzed_until, unknown_spans=…, stages=meta["stages"], required=meta["required_stages"])` и не выводить `no_violation` самому. | Без `meta.coverage` сериализатор отдаёт жюри `failed` вместо `no_violation`: неполный анализ ≠ «нарушений нет». | **Критично**: иначе каждое чистое видео уходит как `failed`. |
| A03 | `backend/pipeline/serde.py:176-186` | Кэш и partial сериализовать без потерь (`to_jsonable(result)` / `asdict`), а не через `result_to_api`. | API-представление применяет политику вердикта и округление, кэш перестаёт быть тождественным (`test_cache_budget` падает). | До сдачи. |
| A03 | `backend/pipeline/tests/test_pipeline.py::test_partial_with_violation` | Не ждать вердикт при `status="processing"`. | По правилу у processing вердикта нет. | Ниже. |
| A04 | `ml/ingest/video.py:955` | Отдать уже посчитанное decode-покрытие оркестратору для `meta.coverage`: недекодированный хвост — это `unknown_spans`. | Источник данных для полноты. | Вместе с A03. |
| A02 | `backend/api/results.py:76-80` | Ответ `GET /api/v1/jobs/{id}` — `result_to_jury_api(result)` вместо `result_to_api(result)`. | `result_to_api` — UI-report с доп. полями; ответ жюри строго по ТЗ. | До сдачи. |
| A12 | `ml/fusion/serialize.py:99-106` (`result_to_jury`) | Вызывать `contracts.serialize.result_to_jury_api`. | Функция задумана как строгий ответ жюри, а `result_to_api` — UI-report. | До сдачи. |
| A05 | `ml/detect/tests/test_labels.py:1`, `test_mock_pipeline.py:8`, `test_hf_local_weights.py:30` | Импортировать контракт через `try: from contracts.sentinel_contracts import … except ImportError: from .._contract_fallback import …`. | С подключённым `contracts` тест берёт `Detection` из копии, а детектор отдаёт канонический класс: `isinstance` = False, на интегрированном дереве падают 2 теста. | До мерджа A01. |
| A10 | `ml/temporal/tests/test_e2e_mock.py:47` | То же, что у A05: прямой импорт `_contract_fallback`. | После `check_fallbacks.py --drop` тест упадёт. | До удаления копий. |
| A06 | `ml/track/events.py:243` | Сигнал `x_cashier_lost_in_view` допустим: префикс `x_` = экспериментальное расширение, `contract_checks` его принимает. Если сигнал нужен правилам — предложить его в `EXTRA_SIGNALS`. | Вне `SIGNALS`. | — |
| A12 × A13 | `ml/fusion/tests` (`test_no_catalog_reports_with_warning`, `test_tune::test_cli_end_to_end`) | Разобрать вдвоём. | Отдельно A12 — 58/58, A12 + контракт A01 — 58/58, A12 + `norms/` — 2 падения: при `catalog={}` fusion, видимо, подхватывает реальный каталог или пороги. | До сдачи. |
| A13 | `norms/catalog.yaml` | Добавить V09 (личные вещи на кассовом столе; есть в спецификации команды и в `TYPE_REGISTRY`). | По FAQ 32 искать настоящий пункт; не нашли — честно «без пункта». | До сдачи. |
| A11 | `ml/vlm/requirements.txt` | Поддержать `transformers` 5.x (в lock L1 — 5.20.1); сейчас требование `<5,>=4.57.1`. | Версии lock вслепую не меняются; без этого VLM не встанет в окружение L1. | **Критично для VLM.** |
| A12 / A16 / A17 | `ml/fusion` (`pillow<13`), `eval` (`pytest<9`, `pyyaml==6.0.1`), `deploy` (`huggingface-hub<2`) | Ослабить верхние границы под lock L1 (`pillow` 13.0.0, `pytest` 9.1.1, `pyyaml` 6.0.3, `huggingface-hub` 2.2.0) или согласовать с L1. | Конфликт с lock (`scripts/DEPS_REPORT.md`). | До сдачи. |
| L1 / A17 | lock L1 | Решить: `fastapi`, `uvicorn`, `starlette`, `python-multipart`, `httpx` (API), `onnxruntime-gpu` (detect, pose), `onnx`, `albumentations`, `matplotlib`, `openpyxl`, `pydantic`, `sentencepiece` — в lock их нет. | `scripts/resolve_deps.py` не подставляет максимальные версии. | До сдачи. |
| A16 | `eval/tests/test_eval_run_split.py::test_run_inproc_end_to_end` | Добавить `meta.coverage` в фейковые результаты или ожидать `failed`. | Фейковые результаты `no_violation` без `meta.coverage` теперь `failed` — следствие правила полноты. | Ниже. |

**Неизвестно ≠ нет:** A09 (`ml/rules/engine.py`) уже различает «не наблюдался» и «нет» (`series.known`, нижняя и верхняя оценки). Шаблона «неизвестно = 0» (`signals.get(x, 0)`) вне тестов и mock-кода нет (раздел 2).

### Сквозной строгий прогон кейсов спецификации через реальные модули

Команда: `python -m testkit.e2e_stub --suite spec --no-backend --strict-stages --cases leave_cash,leave_clean,client_opposite,mobile,desk_phone,lights_off_cash --layouts A,B --orientations portrait,landscape --fps 5`. Синтетика, mock-режим, CPU.

- **24 из 24 прогонов:** все 12 стадий — настоящие модули, нарушений контракта нет — в обеих планировках и в ландшафте.
- **Ответы жюри** проходят строгую схему.
- **Все 24 ответа — `failed`.** Fusion (A12) отдаёт `no_violation` (моки без весов ничего не находят), а `meta.coverage` никто не пишет. Сериализатор отвечает: «нет нарушения» не подтверждено: полнота анализа не подтверждена — нет `meta.coverage`.

Это прямое подтверждение первой записки для A03: пока `meta.coverage` нет, ни одно чистое видео не получит `no_violation`. Качество распознавания эта синтетика не измеряет — это замер L2.

### Тесты модулей на интегрированном дереве

Окружение A01: CPU, Python 3.11, только `numpy`, `opencv-python-headless`, `PyYAML`, `jsonschema`, `pytest`, `SENTINEL_MOCK=1`. Пакеты модулей из lock L1 не ставились. «Окружение» — падение из-за отсутствующего пакета, а не из-за кода.

| Модуль | Результат | Причина падений |
|---|---|---|
| backend/api (A02) | 8 passed, 36 errors | окружение: нет `fastapi` |
| backend/pipeline (A03) | 30 passed, 2 failed | `test_cache_budget` (кэш через API-сериализатор), `test_partial_with_violation` (вердикт у processing) |
| ml/ingest (A04) | 77 passed, 10 failed | окружение: нет PyAV (`av` есть в lock L1) |
| ml/detect (A05) | 35 passed, 2 failed, 3 skipped | прямой импорт `_contract_fallback` в тестах |
| ml/track (A06) | 35 passed | — |
| ml/pose (A07) | 39 passed | — |
| ml/scene (A08) | 35 passed, 1 skipped | — |
| ml/rules (A09) | 119 passed | — |
| ml/temporal (A10) | 29 passed, 2 skipped | — |
| ml/vlm (A11) | 92 passed, 47 failed | в основном окружение (нет `PIL`); остальные A01 не разбирал |
| ml/fusion (A12) | 56 passed, 2 failed | взаимодействие A12 × A13 (см. выше) |
| norms (A13) | 36 passed | — |
| copilot (A14) | 2 errors | окружение: нет `fastapi` |
| eval (A16) | 44 passed, 3 failed, 4 errors, 2 skipped | `openpyxl` (окружение); `test_run_inproc_end_to_end` — правило полноты; `test_logo_exploratory_and_nested` (`TypeError: fit()`) A01 не разбирал |
| deploy (A17) | 38 passed | — |

### Неинтегрированные ветки (честно)

В `main` ничего не смерджено: `main` — пустой корневой коммит `02b6e4d`.

- **Открытый PR в `main`, общая история:** A01 #5, A14 #3, A15 #2, A16 #4, A17 #1.
- **Без PR, несвязанная история** (вливать `git merge --allow-unrelated-histories`, см. `scripts/merge_order.md`): A02, A03, A04, A05, A06, A07, A08, A09, A10, A11, A12, A13.
- **Ветки на origin нет:** A18 (`docs/`, `pitch/`).
- **Вне контрактных функций:** A14 (copilot), A15 (frontend, JS) и A17 (deploy) — для них проверены только тесты и сторож файлов.
<!-- notes:end -->

## 1. Публичные функции модулей против контракта

| Владелец | Ожидаемый вызов (контракт) | Фактический символ | Итог | Расхождение / заметка |
|---|---|---|---|---|
| A04 | `ml.ingest.load_video(path, sample_fps)` **(критический путь)** | `ml.ingest.video.load_video(path, sample_fps=5.0, config=None, backend=None, progress=None)` | OK | — |
| A05 | `ml.detect.detect(clip, every_n)` **(критический путь)** | `ml.detect.detect(clip, every_n=1, kw)` | OK | — |
| A06 | `ml.track.track(clip, dets)` **(критический путь)** | `ml.track.api.track(clip, dets)` | OK | — |
| A07 | `ml.pose.estimate(clip, tracks)` | `ml.pose.estimate(clip, tracks)` | OK | — |
| A07 | `ml.pose.pose_signals(poses, tracks, zones)` | `ml.pose.pose_signals(poses, tracks, zones)` | OK | — |
| A08 | `ml.scene.scene_states(clip, dets, tracks, zones, pose_states)` **(критический путь)** | `ml.scene.scene_states(clip, dets, tracks, zones, pose_states)` | OK | — |
| A09 | `ml.rules.run_rules(states, tracks, zones, catalog)` **(критический путь)** | `ml.rules.engine.run_rules(states, tracks, zones, catalog, config=None)` | OK | — |
| A10 | `ml.temporal.score(clip)` | `ml.temporal.score(clip, zones=None, budget_s=None)` | OK | — |
| A11 | `ml.vlm.verify(clip, events, budget_s)` | `ml.vlm.verify(clip, events, budget_s, limits=None)` | OK | — |
| A11 | `ml.vlm.discover(clip, budget_s)` | `ml.vlm.discover(clip, budget_s, limits=None, motion=None)` | OK | — |
| A12 | `ml.fusion.fuse(clip, events, risk, catalog, job_id)` **(критический путь)** | `ml.fusion.core.fuse(clip, events, risk, catalog, job_id)` | OK | — |
| A13 | `norms.load_catalog()` **(критический путь)** | `norms.load_catalog(path=None, use_cache=True)` | OK | — |
| A03 | `backend.pipeline.run(video_path, job_id, progress, deadline_s)` **(критический путь)** | `backend.pipeline.run(video_path, job_id, progress=None, deadline_s=240, kwargs)` | OK | — |
| A03 | `backend.pipeline.warmup()` | `backend.pipeline.warmup()` | OK | — |

## 2. Статические проверки

### Прямые импорты `_contract_fallback` без `try/except` (сломаются после удаления копий; классы из копии не совпадут по идентичности с каноническими)

- `ml/detect/tests/test_labels.py:1`
- `ml/detect/tests/test_mock_pipeline.py:8`
- `ml/detect/tests/test_hf_local_weights.py:30`
- `ml/temporal/tests/test_e2e_mock.py:47`

### Свои классы с именами классов контракта (риск isinstance-расхождений)

- нет

### Сигналы вне `SIGNALS`/`EXTRA_SIGNALS` (обсудить добавление в словарь)

- ml/temporal/extract.py:80 `model_id`
- ml/temporal/features.py:192 `model_id`
- ml/rules/tests/test_engine.py:126 `a`
- ml/rules/tests/test_engine.py:126 `a`
- ml/rules/tests/test_engine.py:127 `b`
- ml/rules/tests/test_engine.py:127 `b`
- ml/rules/tests/test_engine.py:128 `c`
- ml/track/events.py:243 `x_cashier_lost_in_view`
- ml/track/tests/test_events.py:60 `x_cashier_lost_in_view`

### «Неизвестно = 0»: `signals.get(x, 0)` (неизвестно ≠ нет — нужен явный None/пропуск)

- backend/pipeline/mocks.py:73 `motion_energy`
- backend/pipeline/mocks.py:74 `motion_energy`

### Вызовы сериализатора (ответ жюри должен идти через `result_to_jury_api`)

- `backend/pipeline/serde.py:183`
- `backend/api/results.py:78`
- `backend/api/results.py:80`
- `backend/api/results.py:83`
- `eval/run.py:76`
- `ml/fusion/serialize.py:106`
- `ml/fusion/serialize.py:130`
- `ml/fusion/__init__.py:9`
- `ml/fusion/tests/test_report.py:93`
- `ml/fusion/tests/test_report.py:96`

### Кто пишет `meta.coverage` (без него `no_violation` уходит жюри как `failed`)

- `ml/ingest/video.py:955`
- `ml/ingest/video.py:992`
- `ml/ingest/tests/test_ingest_video.py:235`

## 3. Ветки (состояние на момент генерации)

| Ветка | Агент | HEAD | Общая история с `main` |
|---|---|---|---|
| `origin/claude/gifted-bardeen-i404eh` | A01 | `a3b2b55` | да |
| `origin/claude/laughing-faraday-xe89m9` | A02 | `657424f` | нет (несвязанная история) |
| `origin/claude/elegant-knuth-yszqkw` | A03 | `feceebc` | нет (несвязанная история) |
| `origin/claude/zealous-keller-occtqr` | A04 | `2e98000` | нет (несвязанная история) |
| `origin/claude/hopeful-hypatia-csegbf` | A05 | `5675d4f` | нет (несвязанная история) |
| `origin/claude/wizardly-pascal-w4vt3o` | A06 | `932ba31` | нет (несвязанная история) |
| `origin/claude/nice-gates-fjt926` | A07 | `8d0209a` | нет (несвязанная история) |
| `origin/claude/practical-cerf-0dm6cc` | A08 | `fdf2a0e` | нет (несвязанная история) |
| `origin/claude/wonderful-pasteur-wbu36i` | A09 | `90d8365` | нет (несвязанная история) |
| `origin/claude/upbeat-ride-3bylsq` | A10 | `dcc414b` | нет (несвязанная история) |
| `origin/claude/inspiring-turing-2fm7rq` | A11 | `ce11d85` | нет (несвязанная история) |
| `origin/claude/serene-bell-hycc21` | A12 | `3767007` | нет (несвязанная история) |
| `origin/claude/vigilant-lovelace-ssecv4` | A13 | `fa94a9f` | нет (несвязанная история) |
| `origin/claude/compassionate-wright-9gha0v` | A14 | `9392c86` | да |
| `origin/claude/busy-pasteur-y15o46` | A15 | `fd3901c` | да |
| `origin/claude/wizardly-galileo-hg6mhp` | A16 | `ec0086d` | да |
| `origin/claude/pensive-wright-5l5q0v` | A17 | `7fd0fc9` | да |

