# ml/vlm — локальная VLM: верификация событий и поиск новых типов (A11)

Модуль берёт события-кандидаты (от правил A09, временной модели A10) и проверяет их по кадрам
локальной мультимодальной моделью. Модель возвращает **видимые факты, неуверенность и номера/время кадров**.
Нормы, суммы, личность и полномочия людей она не сообщает. Облачных API нет: модель работает в процессе
пайплайна (transformers) или в локальном llama.cpp.

> Результат VLM — подсказка для fusion (A12) и для человека, не истина. Гипотеза правила — тоже не истина:
> при расхождении модуль снижает уверенность и выставляет флаг противоречия, а решение принимает fusion.

## Публичные функции

```python
import ml.vlm as vlm

vlm.verify(clip, events, budget_s, *, limits=None) -> list[Event]    # контракт v1 + необязательный limits
vlm.discover(clip, budget_s, *, limits=None) -> list[Event]          # по умолчанию выключен -> []
vlm.warmup()            # загрузить модель заранее (держать тёплой)
vlm.unload()            # освободить VRAM
vlm.status()            # технический статус: backend, модель, loaded, причина отказа (без содержимого)
vlm.set_gpu_lock(lock)  # общий GPU-lock A03 (любой context manager): load+generate идут под ним
vlm.set_catalog(cat)    # каталог норм A13 явно (иначе norms.load_catalog() или встроенный fallback)
vlm.set_backend(be)     # подстановка бэкенда (тесты/интеграция)
```

`verify`/`discover` **не бросают исключений**. При сбое или недоступном бэкенде `verify` возвращает события
без изменений и пишет `extra.vlm.status` (`unavailable`, `error`, `skipped_budget`, …), а `discover` возвращает `[]`.

### Что `verify` делает с событием
- Возвращает список той же длины и в том же порядке (1:1). События не удаляются (`verify.drop_below: null`).
  `source` не меняется. Входные объекты не мутируются: на выходе копии.
- На событие берётся 4–6 кадров из окна `t_start−1 … t_end+1`, кадр `t_peak` входит всегда (для длинных событий
  окно ≤ 12 с вокруг пика). На кадры наносятся боксы `event.bboxes` и подпись `#k t=…s`. Длинная сторона ≤ 448 px.
- Промпт требует строгий JSON:
  `{"visible", "type_id", "observed_facts", "confidence", "best_frame", "visible_frames", "uncertain_because"}`.
- Ответ проходит валидацию: `type_id` должен входить в enum каталога, `confidence` — лежать в [0..1]
  (проценты пересчитываются, остальное отбраковывается), номера кадров — в диапазоне 1..N.
  Ошибки валидации попадают в `extra.vlm.flags`.
- Уверенность пересчитывается логит-пулингом:
  `logit(p) = logit(p0) + w·logit(p_vlm)`, где `p_vlm = c` при `visible=true`, иначе `1−c`;
  `p_vlm` обрезается в `[0.05, 0.95]`, `w = 0.6`; VLM не поднимает уверенность больше чем на `max_raise = 0.25`.
  **«Пола» для сильных правил нет.**
- `extra.vlm.agreement`:
  - `supports` — VLM подтверждает; её факты добавляются в `event.facts`.
  - `contradicts` — VLM не видит признака; `extra.vlm.contradiction = true`, уверенность снижена.
  - `uncertain` — ответ не информативен.
  - `type_mismatch` — VLM видит другой тип; `suggested_type_id`, `p` не меняется.
  - `not_checked` — проверки не было: бюджет, нет кадров, сбой.
- Тип переклассифицируется только для источников `temporal`/`vlm` при `confidence ≥ 0.7`; исходный тип
  сохраняется в `extra.vlm_original_type_id`. Тип события правил не меняется — только подсказка.
- Время: `extra.vlm.t_best` и `extra.vlm.t_visible` — времена кадров, которые указала VLM (после валидации).
- Факты проходят фильтр «невидимого»: суммы, проводки, личность, подлинность, умысел, нормы и пункты,
  полномочия и допуск, эмоции, содержимое закрытых объектов, вероятностные оговорки. Кроме того, факт должен
  называть видимую роль или объект, а писаться по-русски. В `extra.vlm.dropped_facts` попадают только категории,
  без текста. Если VLM подтвердила признак, а все её факты отбракованы, подставляется наблюдаемый признак
  из каталога: «Локальная VLM-модель отметила видимый признак: …».
- События одного типа, пересекающиеся по времени, проверяются одним вызовом (`extra.vlm.shared`).
- Порядок проверки — `uncertain_first`: сначала события с уверенностью около 0.5, там VLM полезнее всего.

### `discover` (выключен по умолчанию)
`discover.enabled: false`, пока локальный бенчмарк не покажет скорость и частоту ложных срабатываний. Включить
можно в config или через `discover_events(..., force=True)`.
- Окна по ~3 с. Обход: самые «подвижные» по motion energy первыми, плюс минимальное равномерное покрытие;
  хвост обрезает бюджет.
- Чек-лист: телефон, еда/напиток, открытый сейф без присмотра, посторонний в зоне кассира, наличные без
  присмотра, сумка инкассатора, открытая дверь, скрытые руки, «иное» → V99.
- Находка становится событием только при конкретном видимом факте, прошедшем фильтр, и при
  `confidence ≥ 0.55`. Приор 0.25, потолок 0.6, `source="vlm"`.
- После первой находки один вызов держится в резерве под повторную проверку (`reverify`).
  Если `verify` возражает, находка отбрасывается.

## Бюджет (общий для стадии)
Время задаёт A03 (`budget_s`). Остальные лимиты берутся из `config.yaml → budget`
или из аргумента `limits={...}` (A03 может урезать):

| лимит | verify | discover |
|---|---|---|
| вызовов | 6 | 4 (включая reverify) |
| кадров на вызов | 6 | 6 |
| визуальных токенов на вызов / всего | 1200 / 6000 | 1200 / 3600 |
| выходных токенов на вызов / всего | 220 / 1400 | 220 / 1000 |
| ретраев на вызов / всего | 1 / 2 | 1 / 1 |

Вызов не начинается, если `remaining < est_call_s × 1.15` (оценка — EMA фактических задержек).
У каждого вызова жёсткий таймаут `min(12 с, остаток)`: `generate(max_time=…)` или HTTP timeout.
Ретрай делается только при невалидном JSON (со строгим суффиксом). Таймаут не ретраится.
Число визуальных токенов оценивается как `(W/px)·(H/px)` (px = 32 у Qwen3-VL, 28 у Qwen2.5-VL).
Если кадры не влезают в лимит, они уменьшаются.

## Backend’ы
| backend | статус | что это |
|---|---|---|
| `transformers` | **основной** | Qwen3-VL-4B-Instruct, bnb 4-bit nf4 (vision-энкодер не квантуется), в процессе пайплайна. Один GPU-владелец: load и generate идут под общим lock. Ленивая загрузка, `unload()`. Проверки: свободная VRAM (`min_free_vram_mb`), CUDA OOM (ответ с ошибкой вместо падения), лицензия снапшота, офлайн-загрузка |
| `llamacpp_server` | опционально | локальный `llama-server` (MIT) по HTTP, OpenAI-совместимый. JSON-схема превращается в грамматику, при отказе сервера — fallback на `json_object`. Нелокальные URL запрещены, прокси обходятся |
| `llamacpp_python` | опционально, экспериментально | llama-cpp-python в процессе, один кадр-коллаж на вызов |
| `mock` | тесты | `SENTINEL_MOCK=1`; режимы `neutral`/`confirm`/`reject`/`garbage`/`phone` |

Если бэкенд недоступен (нет CUDA, весов или пакета), стадия **пропускается**. Подмены на mock нет.

## Конфигурация
Файл `ml/vlm/config.yaml`; все пороги лежат там. Переменные окружения:

| env | смысл |
|---|---|
| `SENTINEL_MOCK=1` | mock-бэкенд (сильнее `SENTINEL_VLM_BACKEND`: гарантирует прогон без весов) |
| `SENTINEL_VLM_BACKEND` | `transformers` / `llamacpp_server` / `llamacpp_python` / `auto` / `off` / `mock` |
| `SENTINEL_VLM_PRESET` | `qwen3vl4b` (по умолчанию) / `qwen3vl8b` / `qwen25vl7b_awq` / `qwen3vl2b` |
| `SENTINEL_VLM_MODEL` | путь к локальной папке весов или HF id (для `.gguf` — имя файла) |
| `SENTINEL_VLM_URL` | адрес llama-server (только локальный) |
| `SENTINEL_VLM_CONFIG` | частичный yaml поверх дефолтного |
| `SENTINEL_VLM_LANG` | `ru` / `en`; язык промпта, факты всегда по-русски |
| `HF_HUB_OFFLINE=1` | загрузка только из локального кэша (`local_files_only`) |

## Как запустить
```bash
pip install -r ml/vlm/requirements.txt          # torch — CUDA-сборкой, см. комментарий в файле
python -m pytest ml/vlm/tests -q                # без GPU/весов/сети, mock + синтетика

# один раз, с интернетом (A17): веса в HF-кэш, дальше офлайн
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct
set HF_HUB_OFFLINE=1                            # Windows (bash: export HF_HUB_OFFLINE=1)

# замер скорости/VRAM (агрегаты в stdout, ответы — только в локальный дамп outputs/vlm)
python -m ml.vlm.bench --synthetic 6 --n 5                     # без данных
python -m ml.vlm.bench --frames path/to/frames --n 10 --out bench.json
# ручная проверка момента видео
python -m ml.vlm.try video.mov --t 22 --type V05 [--mode verify|discover|both]
# калибровка на локальной разметке: доля срабатываний discover/verify на позитивах (размеченная секунда)
# и негативах (случайные моменты чистых роликов) — ТОЛЬКО агрегаты; имена X/Х нормализуются, дубли по md5 пропускаются
python -m ml.vlm.calibrate --videos data/videos --labels labels.csv --neg-per-video 2 --out calib.json
# GPU smoke-тест (локально): python -m pytest ml/vlm/tests -m gpu
```

### Приватность вывода
Реальные кадры, промпты и ответы модели CLI пишет **только локально**: `privacy.local_dump_dir`
(по умолчанию `outputs/vlm/`, в `.gitignore`). В stdout выходят технические агрегаты:
статус, задержки, согласие и уверенность, число фактов, категории отбракованных фактов, флаги валидации.
Тексты фактов печатаются только с флагом `--show-text`. Не включайте его, если вывод читает облачный ассистент,
в том числе локальный Claude. Сырой текст ответа в `Event.extra` не кладётся (`privacy.keep_raw_in_extra: false`).
В логи попадают только имена исключений.

## Модели и лицензии
Подробности — в `models.json`. Все веса под Apache-2.0. Лицензию локального снапшота проверяет
`license_guard`: читается поле `license:` из README.md в HF-кэше, без сети. Запрещённая лицензия
(например, `other` у Qwen2.5-VL-3B) блокирует загрузку.

| модель | HF id | лицензия | роль |
|---|---|---|---|
| Qwen3-VL-4B-Instruct | `Qwen/Qwen3-VL-4B-Instruct` | Apache-2.0 | основная |
| Qwen3-VL-8B-Instruct | `Qwen/Qwen3-VL-8B-Instruct` | Apache-2.0 | опц. |
| Qwen2.5-VL-7B-Instruct-AWQ | `Qwen/Qwen2.5-VL-7B-Instruct-AWQ` | Apache-2.0 | опц. (нужен autoawq) |
| Qwen3-VL-2B-Instruct | `Qwen/Qwen3-VL-2B-Instruct` | Apache-2.0 | запасная, только по замеру |
| Qwen3-VL-4B-Instruct-GGUF | `Qwen/Qwen3-VL-4B-Instruct-GGUF` | Apache-2.0 | опц., llama.cpp |

Библиотеки: transformers, accelerate, huggingface_hub — Apache-2.0; torch, numpy — BSD-3; bitsandbytes,
llama.cpp, llama-cpp-python, autoawq, PyYAML — MIT; Pillow — HPND; opencv-python-headless — Apache-2.0.

## Ограничения (честно)
- **Ничего не измерено на реальном железе.** В облачной сессии нет GPU и данных: VRAM, скорость и точность
  4B-модели на RTX 4060 неизвестны. Первым шагом локально запустить `bench`. Если 4B не влезает
  в 8 ГБ рядом с остальными моделями, сначала урезать бюджет (`frames.max_side`, `verify_n_frames`,
  `max_visual_tokens_per_call`, `max_output_tokens_per_call`), выставить `min_free_vram_mb` и перемерить.
  Переход на 2B — только как измеренное решение.
- Фильтр стоп-слов и якорей **не гарантирует отсутствие галлюцинаций**: правдоподобная выдумка
  («кассир держит телефон») проходит. Защита держится на валидации полей, низком весе VLM (`w = 0.6`),
  флагах для fusion и решении человека.
- Самооценка уверенности VLM не откалибрована. `vlm_weight` и `p_vlm_clip` нужно тюнить на LOO (A16).
- Промпты v1 написаны без реальных кадров и без few-shot (так и задумано). Качество формулировок
  проверяется только локально.
- `llamacpp_python` экспериментален: таймаут через stopping_criteria, мультикартинка заменена коллажем.
- Имена GGUF-файлов в пресетах — ожидаемые, их нужно сверить перед скачиванием.

## Для интегратора
1. **A03 (оркестратор):**
   - вызывать `vlm.verify(clip, events, budget_s)` после правил и temporal;
   - `vlm.discover(clip, budget_s)` — только если включён;
   - передать общий GPU-lock через `vlm.set_gpu_lock(lock)`, чтобы VLM не шла параллельно с детектором
     и энкодерами;
   - `vlm.warmup()` вызывать в `backend.pipeline.warmup()`;
   - `vlm.unload()` — если VRAM нужна другим;
   - лимиты урезать через `limits={...}` (ключи — поля `VLMLimits` в `budget.py`);
   - GPU-lock передавать такой, который A03 не держит в момент вызова `verify` (или `RLock`): VLM берёт его сама;
     обычный `Lock`, уже захваченный вызывающим, приведёт к дедлоку;
   - сигнал `motion_energy` из A08 можно отдать в `discover(..., motion=[(t, e), ...])`, чтобы не считать его заново;
   - при 5+ видео одновременно GPU-вызовы VLM сериализуются lock’ом, бюджет каждого видео — wall-clock.
2. **A12 (fusion):** читать `event.extra["vlm"]`:
   - `agreement`, `contradiction`, `p_prior`, `p_vlm`, `p_post`;
   - `suggested_type_id`, `t_best`, `t_visible`, `flags`;
   - `status`, `budget` — агрегаты стадии.

   Решение о вердикте и пороги — на стороне fusion. События `source="vlm"` из discover имеют низкий приор
   (≤ 0.6) и `extra["vlm_discover"]`.
3. **A13 (нормы):** из каталога используются `types.<Vxx>.name`, `observable`, `not_observable`.
   `rules` в промпт не попадают: VLM нормы не цитирует. Если ключа V99 нет, модуль добавляет его сам.
4. **A17 (deploy):**
   - скачать веса заранее и выставить `HF_HUB_OFFLINE=1`;
   - на Windows нужны CUDA-сборка torch и bitsandbytes ≥ 0.45;
   - для llama.cpp-варианта: `llama-server -m <gguf> --mmproj <mmproj> --port 8081 -ngl 99 -c 8192`
     и `SENTINEL_VLM_BACKEND=llamacpp_server`.
5. **A01 (контракт/CI):**
   - контракт не менялся;
   - `_contract_fallback.py` — копия `contracts/sentinel_contracts.py`, удалить после мерджа (импорт идёт
     через `ml/vlm/_contract.py`);
   - маркеры `gpu`/`slow` регистрируются в `ml/vlm/tests/conftest.py` — можно перенести в корневой pyproject;
   - тесты лежат в пакете `ml.vlm.tests` (`__init__.py`), файлы с префиксом `test_vlm_`, чтобы не
     конфликтовать с одноимёнными тестами других модулей.
   - Предложение: зафиксировать в контракте семантику `Event.extra["vlm"]["agreement"]`.
6. **A16 (eval):**
   - `python -m ml.vlm.calibrate` даёт долю срабатываний discover/verify на позитивах и негативах; это основа
     для решения о `discover.enabled` и подбора `vlm_weight`;
   - прогнать LOO с `SENTINEL_VLM_BACKEND=transformers`;
   - сравнить вердикты с VLM и без неё, частоту `contradicts` на чистых роликах;
   - покрутить `verify.vlm_weight` и `max_raise`;
   - решить, включать ли `discover`.

## Принятые решения (автономно)
- Основной backend — transformers в процессе: один GPU-владелец и общий lock. Отдельный llama-server был бы
  вторым владельцем VRAM, поэтому llama.cpp оставлен опцией.
- Модель по умолчанию — Qwen3-VL-4B (Apache-2.0) в 4-bit. 7B/8B не по умолчанию: память не измерена,
  а 8 ГБ делятся с другими моделями.
- Подмодули названы `verifier.py` и `discoverer.py`. Если назвать их `verify.py`/`discover.py`,
  `from .verify import …` затёр бы функцию `ml.vlm.verify` модулем.
- `python -m ml.vlm.try`: `try` — ключевое слово Python, поэтому `try.py` — тонкая обёртка над `cli_try.py`.
- Факты для жюри требуются по-русски; ответы на латинице отбрасываются и заменяются шаблоном из каталога.
- Неизвестный `type_id` от модели не подменяется на V99 молча: он становится `None` и получает флаг.

## Файлы
```
__init__.py            публичный API (лёгкий импорт)
config.yaml, config.py конфиг + env
budget.py              общий бюджет: время/вызовы/кадры/токены/ретраи
runtime.py             жизненный цикл бэкенда, GPU-lock, вызов с JSON-ретраем, ужатие кадров
verifier.py            verify_events()
discoverer.py          discover_events(), plan_windows()
scoring.py             логит-пулинг, согласие
parsing.py             извлечение/ремонт JSON, схемы, валидация enum/диапазонов
facts_filter.py        фильтр невидимого + якоря наблюдаемости
frames.py              выбор моментов, боксы, ресайз, коллаж, motion energy
prompting.py, prompts/ библиотека промптов <id>.<version>.<lang>.md (RU/EN)
catalog.py             каталог A13 или встроенный fallback (без норм)
backends/              mock, transformers, llama.cpp (server/python)
bench.py, try.py (cli_try.py), calibrate.py           CLI: скорость/VRAM, ручная проверка, калибровка
localdump.py, _video.py                                 локальный дамп, запасной загрузчик видео
_contract.py, _contract_fallback.py                     импорт контракта
tests/                 pytest: mock + синтетика, GPU-тест помечен @gpu
```
