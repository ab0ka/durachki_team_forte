# Совместимость веток с контрактом v1

<!-- notes:begin (ручной раздел, сохраняется при перегенерации) -->
## 0. Состояние после аудита `integration-cloud@2055fc4` (17.10)

**Дерево проверки** — локальная симуляция слияния: `integration-cloud@2055fc4` + текущие головы 16 веток владельцев (раздел 3) + A01. Собрано через `git merge-tree`; ветки и рефы на origin не менялись. Без конфликтов сливается всё, кроме A02 (два тестовых файла, п. 3 ниже). Окружение: CPU, Python 3.11, без GPU, весов и реальных видео, только синтетика. Есть numpy 2.4.6, opencv 4.14.0, PyAV 18.1.0, Pillow 13.0.0, fastapi 0.143.3, scipy 1.17.1. Нет torch, transformers, onnxruntime, openpyxl, psutil, requests. Для повтора CI — отдельное окружение «CI-light»: ровно `pip install -r <(python scripts/ci_requirements.py --all)`, как в `ci.yml` (transformers 5.20.1 без torch, openpyxl и т.д.).

| Проверка на объединённом дереве | Результат |
|---|---|
| `python scripts/check_fallbacks.py` | rc 0: копий контракта нет, висячих импортов `_contract_fallback` (DANGLING) нет |
| `python scripts/guard_media.py --all` | rc 0 |
| `python scripts/check_licenses.py` | rc 0 |
| `python scripts/resolve_deps.py --lock-ref origin/data/labels:tools/dataprep/requirements-l1.lock.txt` (lock `b19406b`) | конфликтов 0; нет в lock: `albumentations`, `peft` (п. 6) |
| `pytest contracts testkit scripts -m "not gpu and not slow"` | 402 passed без `SENTINEL_MOCK`; тесты контракта с `SENTINEL_MOCK=1` — 141 passed |
| `SENTINEL_MOCK=1 python -m testkit.e2e_stub --require-full --expect-status done --scenarios clean,V01,V03,V07 --duration 8 --fps 30` | 4/4 через `backend.pipeline.run` → `done`, строгая схема жюри, ошибок 0, 34 с |
| то же, `--suite spec --cases leave_clean,leave_cash,client_opposite,mobile --layouts A,B --duration-range 8,10` | 8/8 → `done`, ошибок 0, 73 с |
| `python -m testkit.e2e_stub --require-full --expect-status failed --scenarios clean --duration 8 --fps 30` (без mock) | `failed`: «detect: weights … not found locally»; mock не отдан |
| три команды выше в окружении CI-light, на пустом кэше A03 (как job `e2e-full`) | 4/4 и 8/8 → `done` (36 с и 77 с), без mock → `failed`; ошибок 0. `make e2e-full` — rc 0, 38 с |
| тесты модулей по одному, без `SENTINEL_MOCK` | 12 из 15 модулей зелёные. В CI-light 13: `eval` — 85 passed, у нас не было `openpyxl`. Остаются п. 4 и 5 |
| все модули одной сессией pytest, без `SENTINEL_MOCK` и без корневого `conftest.py` | 2114 passed, 3 failed, 4 errors — ровно те же, что при раздельных прогонах (п. 4, 5, `openpyxl`). Падений, зависящих от порядка или окружения, нет |
| `ruff check <модуль>` (корневой конфиг A01: `E4, E7, E9, F, B`) | 5 из 15 чистые; ошибки у 10 модулей и в `scripts/integration_merge.py` (п. 7, 8) |

Вердикты в mock-режиме информативны: моки без весов ничего не находят. Качество распознавания эти прогоны не измеряют — это замер L2.

### Исправлено после аудита (не блокеры, повторно не поднимать)

| Владелец | Было | Стало | SHA |
|---|---|---|---|
| A01 | `testkit/e2e_stub` ставил `SENTINEL_MOCK` и `HF_HUB_OFFLINE` в `os.environ` | окружение не меняется; mock только явно | `5e9f33f` |
| A16 | `eval/tests/conftest.py` ставил `SENTINEL_MOCK=1` на всю сессию | фикстура `mock_mode` (monkeypatch) | `50bfe22` |
| A03 | нет `meta.coverage` | `compute_coverage` + `gaps` + `by_stage` | `af48ab5` |
| A11 | `transformers<5` | `transformers>=4.57.1` (lock 5.20.1) | `c174ab5` |
| A12 | `Pillow<13` | `Pillow>=10,<14` (lock 13.0.0) | `60c1a3a` |
| A12 | `result_to_jury` вызывал `result_to_api` (UI-report) | вызывает `result_to_jury_api` | `f256228` |
| A02 | строгий профиль всегда дописывал норму в description | `jury_embed_rule: false`, как у A01 | `a9e3bd7` |
| A05, A10 | прямые импорты `_contract_fallback` в тестах | переписаны при интеграции (`a6a96f3`, `383b394`); DANGLING нет | — |

### Актуальные несовместимости

| # | Владелец, голова | Где | Вход → факт | Ожидается | Срочность |
|---|---|---|---|---|---|
| 1 | A03 `db529bc` | `backend/pipeline/coverage.py:40` и `:68` (`interval_coverage`, `covered_intervals`): окно кадра = медианный шаг | Кадры с чередующимся шагом 0.2/0.3 с на 10 с, `max_gap_s=2.0` → покрыто 0.8. Сквозной прогон: синтетика 6 с, 10 fps, `SENTINEL_MOCK=1` → `sample_fps 4.5`, `coverage 0.9`, `failed` | 1.0: ни один шаг не превышает `max_gap_s`. Кадр t покрывает [t, t_next), если t_next − t ≤ `max_gap_s` (окно — шаг до следующего кадра, а не медиана). Порог 0.98 не снижать. | **Критично:** чистое видео с неравномерной выборкой (VFR; `sample_fps`, не кратный fps) уходит жюри как `failed` |
| 2 | A03 `db529bc` | `backend/pipeline/worker.py:566` (`atexit.register(self.shutdown)` в `ProcessExecutor.__init__`), воркер `daemon=False` (`:601`) | Библиотечный `run()` возвращается за 0.6 с, но процесс не завершается: убит по таймауту через 60 с. С `release_gpu()` выходит за 1.3 с, с `import multiprocessing.util` до импорта пайплайна — за 1 с | Процесс завершается сам. Причина: `multiprocessing.util` регистрирует свой atexit при первом spawn, то есть позже. Он срабатывает первым и ждёт (join) воркер, который ждёт команд. Регистрировать shutdown после старта воркера или импортировать `multiprocessing.util` до `atexit.register`. | До сдачи: `release_gpu()` не вызывают `eval/run.py:83` (A16, InprocRunner) и `deploy/offline_worker_smoke.py:244` (A17) — их процессы не завершатся. `testkit/e2e_stub` и `backend/api/worker.py` вызывают. |
| 3 | A02 `c2a869b` × интеграция `383b394` | `backend/api/tests/conftest.py`, `backend/api/tests/test_phase2.py` | Конфликт слияния: интеграция и A02 обе переписали импорт `_contract_fallback` | Взять сторону A02 (`git checkout --theirs`): сначала `contracts`, fallback под `except ImportError`. Проверено: `check_fallbacks` rc 0, `backend/api` 103 passed | При слиянии (L2) |
| 4 | A07 `e38f2b8` | `ml/pose/tests/test_pose_honesty.py:114` (`test_transition_unavailable_available_unavailable`, `3eb69ae`) | Без `onnxruntime` `model_info()["available"]` = False → тест падает. CI-light (`scripts/ci_requirements.py`) не ставит `onnxruntime-gpu`, так что job `modules` для `ml/pose` будет красным | Пропуск при `importlib.util.find_spec("onnxruntime") is None` или monkeypatch `_availability._has_package` в тесте | До сдачи |
| 5 | A14 `463aeaa` × A13 `6d4998e` | `copilot/tests/test_train.py:32` (`make_dataset.build(n=60, seed=3)`); каталог подхватывается в `copilot/corpus.py:193` | На дереве A14 — 4/4. С `norms/` (A13) — 1 падение: пример категории `unobservable` получает ответ «Вердикт: обнаружены признаки возможного нарушения…» вместо отказа «По имеющимся данным…» | Отказ на ненаблюдаемый вопрос не зависит от наличия каталога норм; тест детерминирован при любом каталоге | До сдачи |
| 6 | L1, lock `b19406b` | `ml/detect/requirements.txt:11` `albumentations>=1.4` (A05, только дообучение); `copilot/train/requirements-lora.txt:3` `peft>=0.13` (A14, LoRA, вне критического пути) | Пакетов нет в lock | L1: добавить в lock или вынести в отдельные training-requirements | Ниже (рантайм не затронут) |
| 7 | владельцы модулей | `ruff check <модуль>` на объединённом дереве; те же числа на головах владельцев с конфигом A01, интеграция ошибок не добавила | A02 `c2a869b` `backend/api` — 10 (F401×8, B023, B009); A03 `db529bc` `backend/pipeline` — 7 (F401×6, B023); A04 `afde1be` `ml/ingest` — 9 (F401×5, E702×4); A06 `353c826` `ml/track` — 17 (B023×12, B007×4, B010); A07 `e38f2b8` `ml/pose` — 3 (B007×2, F401); A10 `0d026a2` `ml/temporal` — 13 (E702×11, B023×2); A12 `f256228` `ml/fusion` — 1 (B007); A13 `6d4998e` `norms` — 1 (B007); A16 `50bfe22` `eval` — 13 (F401×5, E702×4, B009×2, B007, F841); A17 `f233feb` `deploy` — 4 (F401×3, F841) | `ruff check <модуль>` чисто: `--fix` для F401/B009/B010, остальное вручную. B023 (замыкание на переменную цикла) — проверить как возможную ошибку, а не только стиль. Проверку не ослаблять. | До сдачи: в CI интеграции job `modules` красный на шаге Ruff |
| 8 | L2 | `scripts/integration_merge.py:5` (есть только в `integration-cloud`) | `import json` не используется, несколько импортов в одной строке (F401, E401) → `core` в CI интеграции красный на шаге Ruff, остальные шаги `core` раньше пропускались | `ruff check --fix scripts/integration_merge.py` при слиянии. В новом `ci.yml` шаги независимы, так что ошибка ruff больше не скрывает `check_fallbacks` и `guard_media` | При слиянии (L2) |
| 9 | L2 | корневой `conftest.py` в `integration-cloud` (`2055fc4`, glue A03 в зоне A01): возвращает `SENTINEL_*` к снимку вокруг каждого теста вне `eval/` | Нужен был из-за утечек env у A01 и A16; обе исправлены (`5e9f33f`, `50bfe22`) | Удалить при слиянии A01. Проверено: общий прогон всех модулей без него не даёт новых падений (2114 passed) | При слиянии (L2) |

### Решения L2 (не ошибка кода)

- **Профиль ответа жюри.** A02 `backend/api/config.yaml:53` — `jury_profile: extended` по умолчанию (с `8e5104d`). Такой ответ валиден по схеме ТЗ (`additionalProperties: true`), но это не строгий профиль. Для сдачи — `jury_profile: strict`: ровно поля ТЗ, как `contracts.serialize.result_to_jury_api`. `extended` нужен eval A16.
- **`contracts/config.yaml` `jury.embed_rule_in_description: false`** — гипотеза. Включать только после замера.

### Проверка сборки для L2 (одна команда, в корне интеграционного дерева после слияния)

```bash
SENTINEL_MOCK=1 SENTINEL_CACHE_DIR=$(mktemp -d) SENTINEL_DATA_DIR=$(mktemp -d) \
  python -m testkit.e2e_stub --require-full --expect-status done --scenarios clean,V01,V03,V07 --duration 8 --fps 30
```

PowerShell: `$t="$env:TEMP\se2e-$(Get-Random)"; $env:SENTINEL_MOCK="1"; $env:SENTINEL_CACHE_DIR="$t\cache"; $env:SENTINEL_DATA_DIR="$t\data"; python -m testkit.e2e_stub --require-full --expect-status done --scenarios clean,V01,V03,V07 --duration 8 --fps 30; Remove-Item Env:SENTINEL_MOCK`.

Ожидается: «ПОЛНАЯ СБОРКА ПРОВЕРЕНА (backend.pipeline.run + канонический сериализатор жюри): ошибок нет», код 0. Код 3 — NOT RUN: нет `backend.pipeline.run`. Код 1 — ошибка; причина печатается построчно (`ERROR …`). Это проверка сборки, а не качества: mock-режим, синтетика. Полный набор, как в CI, — `make e2e-full`.
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
| A11 | `ml.vlm.verify(clip, events, budget_s)` | `ml.vlm.verify(clip, events, budget_s, limits=None, zones=None)` | OK | — |
| A11 | `ml.vlm.discover(clip, budget_s)` | `ml.vlm.discover(clip, budget_s, limits=None, motion=None)` | OK | — |
| A12 | `ml.fusion.fuse(clip, events, risk, catalog, job_id)` **(критический путь)** | `ml.fusion.core.fuse(clip, events, risk, catalog, job_id)` | OK | — |
| A13 | `norms.load_catalog()` **(критический путь)** | `norms.load_catalog(path=None, use_cache=True)` | OK | — |
| A03 | `backend.pipeline.run(video_path, job_id, progress, deadline_s)` **(критический путь)** | `backend.pipeline.run(video_path, job_id, progress=None, deadline_s=240, kwargs)` | OK | — |
| A03 | `backend.pipeline.warmup()` | `backend.pipeline.warmup(deadline_s=None)` | OK | — |

## 2. Статические проверки

### Прямые импорты `_contract_fallback` без `try/except` (сломаются после удаления копий; классы из копии не совпадут по идентичности с каноническими)

- нет

### Свои классы с именами классов контракта (риск isinstance-расхождений)

- нет

### Сигналы вне `SIGNALS`/`EXTRA_SIGNALS` (обсудить добавление в словарь)

- ml/temporal/extract.py:103 `model_id`
- ml/temporal/features.py:214 `model_id`
- ml/rules/tests/test_engine.py:126 `a`
- ml/rules/tests/test_engine.py:126 `a`
- ml/rules/tests/test_engine.py:127 `b`
- ml/rules/tests/test_engine.py:127 `b`
- ml/rules/tests/test_engine.py:128 `c`
- ml/rules/tests/test_interop.py:42 `a`
- ml/rules/tests/test_interop.py:49 `workplace_empty`
- ml/rules/tests/test_interop.py:51 `workplace_empty_conf`
- ml/rules/tests/test_interop.py:52 `frame_quality`
- ml/track/events.py:243 `x_cashier_lost_in_view`
- ml/track/tests/test_events.py:60 `x_cashier_lost_in_view`

### «Неизвестно = 0»: `signals.get(x, 0)` (неизвестно ≠ нет — нужен явный None/пропуск)

- backend/pipeline/mocks.py:73 `motion_energy`
- backend/pipeline/mocks.py:74 `motion_energy`

### Вызовы сериализатора (ответ жюри должен идти через `result_to_jury_api`)

- `backend/pipeline/serde.py:183`
- `backend/pipeline/tests/test_finalize.py:92`
- `backend/pipeline/tests/test_finalize.py:95`
- `backend/api/results.py:81`
- `backend/api/results.py:83`
- `backend/api/results.py:86`
- `eval/run.py:85`
- `ml/fusion/serialize.py:143`
- `ml/fusion/__init__.py:9`
- `ml/fusion/tests/test_report.py:93`
- `ml/fusion/tests/test_report.py:96`
- `ml/fusion/tests/test_golden.py:69`
- `ml/fusion/tests/test_golden.py:82`
- `ml/fusion/tests/test_golden.py:83`

### Кто пишет `meta.coverage` (без него `no_violation` уходит жюри как `failed`)

- `deploy/stub_api.py:39`
- `deploy/offline_worker_smoke.py:94`
- `deploy/tests/test_http_tools.py:112`
- `deploy/tests/test_http_tools.py:119`
- `backend/pipeline/cli.py:212`
- `backend/pipeline/compare.py:65`
- `backend/pipeline/orchestrator.py:929`
- `backend/pipeline/tests/test_compare.py:18`
- `backend/pipeline/tests/test_coverage.py:54`
- `backend/api/worker.py:608`
- `backend/api/results.py:142`
- `backend/api/tests/test_phase2.py:64`
- `backend/api/tests/test_phase2.py:65`
- `backend/api/tests/test_phase2.py:66`
- `backend/api/tests/test_phase2.py:67`
- `backend/api/tests/test_phase2.py:68`
- `backend/api/tests/test_phase2.py:70`
- `backend/api/tests/test_phase2.py:72`
- `backend/api/tests/test_phase2.py:88`
- `backend/api/tests/test_phase2.py:128`
- `backend/api/tests/test_phase2.py:154`
- `backend/api/tests/test_phase2.py:393`
- `backend/api/tests/test_phase2.py:416`
- `eval/jury_sim.py:71`
- `eval/tests/test_eval_ablate.py:100`
- `eval/tests/test_eval_metrics.py:180`
- `eval/tests/test_eval_metrics.py:182`
- `eval/tests/test_eval_metrics.py:183`
- `eval/tests/test_eval_metrics.py:200`
- `eval/tests/test_eval_metrics.py:204`
- `eval/tests/test_eval_metrics.py:207`
- `eval/tests/test_eval_metrics.py:208`
- `eval/tests/test_eval_run_split.py:71`
- `eval/tests/conftest.py:106`
- `eval/tests/test_eval_jury_sim.py:34`
- `eval/tests/test_eval_jury_sim.py:36`
- `eval/tests/test_eval_jury_sim.py:38`
- `ml/fusion/core.py:1092`
- `ml/fusion/tests/test_upstream.py:96`
- `ml/fusion/tests/test_upstream.py:111`
- `ml/fusion/tests/test_upstream.py:143`
- `ml/ingest/synthetic.py:109`
- `ml/ingest/bench.py:117`
- `ml/ingest/video.py:812`
- `ml/ingest/video.py:1035`
- `ml/ingest/video.py:1080`
- `ml/ingest/tests/test_ingest_video.py:237`
- `copilot/extractive.py:194`
- `copilot/train/make_dataset.py:31`
- `copilot/tests/test_completeness.py:29`
- `copilot/tests/test_completeness.py:30`

## 3. Ветки (состояние на момент генерации)

| Ветка | Агент | HEAD | Общая история с `main` |
|---|---|---|---|
| `origin/claude/gifted-bardeen-i404eh` | A01 | `5e9f33f` | да |
| `origin/claude/laughing-faraday-xe89m9` | A02 | `c2a869b` | нет (несвязанная история) |
| `origin/claude/elegant-knuth-yszqkw` | A03 | `db529bc` | нет (несвязанная история) |
| `origin/claude/zealous-keller-occtqr` | A04 | `afde1be` | нет (несвязанная история) |
| `origin/claude/hopeful-hypatia-csegbf` | A05 | `4e2a429` | нет (несвязанная история) |
| `origin/claude/wizardly-pascal-w4vt3o` | A06 | `353c826` | нет (несвязанная история) |
| `origin/claude/nice-gates-fjt926` | A07 | `e38f2b8` | нет (несвязанная история) |
| `origin/claude/practical-cerf-0dm6cc` | A08 | `a81a5c5` | нет (несвязанная история) |
| `origin/claude/wonderful-pasteur-wbu36i` | A09 | `f05223c` | нет (несвязанная история) |
| `origin/claude/upbeat-ride-3bylsq` | A10 | `0d026a2` | нет (несвязанная история) |
| `origin/claude/inspiring-turing-2fm7rq` | A11 | `b4b2ec5` | нет (несвязанная история) |
| `origin/claude/serene-bell-hycc21` | A12 | `f256228` | нет (несвязанная история) |
| `origin/claude/vigilant-lovelace-ssecv4` | A13 | `6d4998e` | нет (несвязанная история) |
| `origin/claude/compassionate-wright-9gha0v` | A14 | `463aeaa` | да |
| `origin/claude/busy-pasteur-y15o46` | A15 | `90083a0` | да |
| `origin/claude/wizardly-galileo-hg6mhp` | A16 | `50bfe22` | да |
| `origin/claude/pensive-wright-5l5q0v` | A17 | `f233feb` | да |
| `origin/claude/integration-cloud` | L2 (интеграция) | `2055fc4` | да |

