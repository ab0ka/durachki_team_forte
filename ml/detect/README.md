# ml/detect — детекция объектов (A05)

`detect(clip, every_n=1) -> list[Detection]`: метки из `LABELS`, bbox нормированы 0..1 к кадру после учёта поворота.
`track_id=None` (трекинг — A06). В `Detection.source` записан исходный класс модели (см. «Семантика меток»).

## Статус

| Часть | Состояние | По умолчанию |
|---|---|---|
| **Baseline**: RT-DETRv2-S (`PekingU/rtdetr_v2_r18vd`, transformers), каждый сэмпл, батчи, препроцессинг на GPU, fp16 на CUDA (откат на fp32), CPU fallback | готово | **вкл.** |
| TTA: горизонтальный флип + WBF | готово, опционально | выкл. |
| Тайлы в стиле SAHI (своя реализация) + NMS и подавление вложенных боксов | готово, опционально | выкл. |
| Open-vocab OWLv2 на ключевых кадрах (~1 fps + всплески движения) | готово, опционально | выкл. |
| Дообученная модель (ONNX, onnxruntime) | готов код; модели нет, пока нет проверенной разметки | выкл. |
| Mock (`SENTINEL_MOCK=1`, цветовые маски синтетики) | готово, только для тестов | выкл. |

Включение опций: `SENTINEL_DETECT_EXTRAS=tta,tiles,open_vocab,finetuned` (или `all`), либо YAML-override через `SENTINEL_DETECT_CONFIG`.

### Что реально проверено (CPU, без GPU, без реальных данных и весов)
`python -m pytest ml/detect/tests`: **43 passed** (torch 2.14.1 CPU, transformers 5.20.1, onnxruntime, Python 3.13).
- маппинг классов, NMS/WBF/флип/тайлы/подавление вложенных боксов;
- mock end-to-end на синтетике (в памяти и из mp4-файла через `cv2.VideoWriter`): формат `Detection`, детерминизм, `every_n`, гейтинг ключевых кадров, адаптивный шаг при превышении бюджета, битые кадры, потоковая изоляция `last_stats()`;
- постобработка RT-DETR и OWLv2 на фейковых выходах модели: пересчёт боксов OWLv2 из квадрата с паддингом в портретный кадр, исходные классы в `source`;
- **реальный путь загрузки transformers без сети**: крошечная случайная RT-DETRv2 в `SENTINEL_WEIGHTS/detect/` → `warmup()` → `detect()`, в том числе с флипом и тайлами; сетевые вызовы в тесте запрещены;
- нет весов → `DetectorUnavailable`, без обращения к сети и без перехода на mock;
- дообучение на крошечной модели: проверенный COCO → 2 шага `Trainer` → экспорт ONNX → `OnnxDetrBackend`; отказ на непроверенной разметке;
- COCO-AP: `quality_claim=false` на непроверенной разметке.

**Не измерено** (нет GPU и данных): скорость на RTX 4060 и качество на реальных роликах.
Замер на целевом ноутбуке: `python -m ml.detect video.mov`, тайминги по стадиям выводятся в `stats.timings`.
Целевые значения из ТЗ: closed-set 5 fps × 145 с ≤ 15 с, open-vocab ≤ 10 с.

## Публичный API

```python
from ml.detect import detect, warmup, last_stats, available, DetectorUnavailable

dets = detect(clip, every_n=1)   # list[Detection]; кидает DetectorUnavailable, если нет локальных весов
last_stats()  # {"status": ok|degraded|unavailable, "timings": {...}, "backends": [...], "warnings": [...],
              #  "n_processed", "n_keyframes", "n_flip", "n_tiles", "n_open", "stride_final", ...}
warmup()      # загрузить модели и прогреть их; не кидает исключений, возвращает {"status", "backends", "warnings", "seconds"}
available()   # без загрузки моделей и без сети: {"ready": bool, "closed_set": {id: bool}, ...}
```
Дополнительный аргумент `detect(..., config={...})` переопределяет конфиг (нужен для тестов и тюнинга).
`detect()` потокобезопасен: вызовы сериализуются (один GPU на 8 ГБ), `last_stats()` хранится отдельно для каждого потока.

## Семантика меток (без домыслов)

| Метка | Источник | Что означает, а что НЕ означает |
|---|---|---|
| person | COCO person | человек; **роль (кассир/клиент/инкассатор) детектор не определяет** |
| cell_phone, cup, bottle, laptop | одноимённые классы COCO | — |
| food | COCO banana/apple/sandwich/orange/broccoli/carrot/hot dog/pizza/donut/cake | общее понятие «еда» |
| cup ← wine glass | COCO | посуда для напитков |
| cup ← bowl, document ← book | COCO, класс-заместитель | **сходство, а не тождество** (книга ≠ банковский документ). `source` = `...:book~proxy`; выключается `closed_set.proxy_labels: false` |
| bag ← handbag/backpack/suitcase | COCO | любая сумка; **не инкассаторская** — `cash_bag` из COCO никогда не выдаётся |
| cash, cash_bag, safe, safe_open, door, door_open, money_counter, keys | только опциональные OWLv2 и дообученная модель | визуальное сходство с текстовым запросом (`source` = `owlv2:green canvas bag`), а не назначение предмета |

Детектор не делает выводов о ролях, полномочиях или умысле людей по предметам. Скор — уверенность модели в исходном классе.

## Как запустить

```bash
pip install -r ml/detect/requirements.txt
# SETUP (единственный шаг с интернетом):
python -m ml.detect.weights --download          # → $SENTINEL_WEIGHTS/hf (по умолчанию <repo>/weights/hf)
python -m ml.detect.licenses --check            # сверка заявленных лицензий с карточками на HF Hub
# РАНТАЙМ (без сети):
HF_HUB_OFFLINE=1 python -m ml.detect path/to/video.mov --out dets.json
HF_HUB_OFFLINE=1 python -m ml.detect video.mov --extras tta,tiles,open_vocab
# тесты:
python -m pytest ml/detect/tests            # медленные (крошечные модели transformers): -m slow / -m "not slow"
```
При импорте и инференсе в сеть ничего не ходит: `from_pretrained(..., local_files_only=True)`.
Можно положить модель, сохранённую через `save_pretrained`, в `$SENTINEL_WEIGHTS/detect/<имя>/`.

### Переменные окружения
| env | смысл |
|---|---|
| `SENTINEL_WEIGHTS` | каталог весов (по умолчанию `<repo>/weights`) |
| `SENTINEL_MOCK=1` | только тестовый режим: цветовой mock вместо моделей |
| `SENTINEL_DETECT_EXTRAS` | `tta,tiles,open_vocab,finetuned` или `all` |
| `SENTINEL_DETECT_CONFIG` | YAML, который сливается поверх `config.yaml` |
| `SENTINEL_DETECT_BUDGET_S` | мягкий бюджет detect() в секундах (после него опции отключаются) |
| `HF_HUB_OFFLINE=1` | рекомендуется в рантайме (код и так не скачивает) |

## Конфиг (`config.yaml`)
Все пороги вынесены сюда: пороги по меткам после слияния (для person снижен до 0.25, потому что вид сверху), сырой `min_score`, TTA, тайлы, запросы и пороги OWLv2, `max_area`, гейтинг по движению, бюджет и `on_unavailable: raise|empty`, палитра mock.
Пиксельных координат зон и объектов нет: конфиг не зависит от ракурса и разрешения.

## Модели и лицензии (`models.json`)
| Модель | Заявленная лицензия | Статус проверки | Роль |
|---|---|---|---|
| `PekingU/rtdetr_v2_r18vd` | apache-2.0 | `declared_unverified` | baseline |
| `ustc-community/dfine-small-coco` | apache-2.0 | `declared_unverified` | опционально (не подключена) |
| `google/owlv2-base-patch16-ensemble` | apache-2.0 | `declared_unverified` | опционально |
| дообученная ONNX | наследуется от базовой | `derived` | опционально |

Лицензия указана по карточкам моделей и репозиториям кода. Сам **вес** в этой сессии не проверен: HF Hub отсюда недоступен.
Название семейства не заменяет проверку конкретного чекпойнта. На этапе setup запустите `python -m ml.detect.licenses --check`; оно сверяет `card_data.license` и показывает файлы LICENSE/README в снапшоте. `license_status` меняет вручную человек, который их просмотрел.
Лицензии кода зависимостей указаны в `requirements.txt` (BSD-3/Apache-2.0/MIT).
Permissive-модели выбраны, чтобы упростить соблюдение условий и следовать правилу команды. Сами по себе AGPL/GPL коммерческое использование не запрещают.

## Дообучение (опционально, только на проверенной разметке)
1. Кадры из роликов извлекаются локально (A04 или `synth.ArrayClip.from_file`) в `data/frames/<видео>/*.jpg`. В git они не попадают.
2. `python -m ml.detect.autolabel propose --frames data/frames --out data/autolabel/proposals.json` — OWLv2 выдаёт **предложения** в формате COCO (`verified=false`, `score`, `source`). Пороги строже рантайма, плюс проверка согласованности на отражённом кадре.
3. Человек импортирует их в CVAT или Label Studio, исправляет, удаляет и добавляет боксы, экспортирует COCO и подтверждает:
   `python -m ml.detect.autolabel mark-verified reviewed.json data/autolabel/verified.json --reviewer "<имя>"`.
4. `python -m ml.detect.finetune train --coco data/autolabel/verified.json --frames data/frames --out runs/ft`. На непроверенной разметке — `UnverifiedLabelsError`. Train/val делится по видео; аугментации (перспектива, аффинные, цвет, смаз) имитируют другой угол камеры.
5. `python -m ml.detect.finetune export --model runs/ft/hf --out "$SENTINEL_WEIGHTS/detect/finetuned"`, затем `SENTINEL_DETECT_EXTRAS=finetuned`.
6. `python -m ml.detect.eval_det --gt <проверенный val> --run --frames data/frames`. AP считается качеством только при `quality_claim=true`. Согласие с псевдоразметкой качеством не является и в отчётах так не подаётся.

Данные локальные: кадры и описания сцен не передаются в облачные сервисы, включая облачные LLM-ассистенты.

## Ограничения
- COCO-модели хуже работают на виде сверху. Пороги для person снижены, но их нужно подбирать на реальных роликах (A16).
- Baseline не выдаёт cash, safe, safe_open, door, door_open, cash_bag, money_counter и keys: таких классов нет в COCO. Нужны open-vocab (опция) или дообучение на проверенной разметке.
- Open-vocab работает только на ключевых кадрах (около 1 fps), поэтому эти метки разреженные по времени.
- Бюджет: после `budget_s` опции отключаются; если прогноз превышает `hard_budget_s`, шаг удваивается (максимум ×4). Причины пишутся в `last_stats()["warnings"]`, статус становится `degraded`.

## Для интегратора
- **A03 (pipeline):** вызвать `ml.detect.warmup()` из `backend.pipeline.warmup()`; `detect(clip)` оборачивать в `try/except DetectorUnavailable`. Если детектор недоступен, это надо явно сообщить в `meta.warnings`; тихо отдавать `no_violation` нельзя. `last_stats()` класть в `meta["detect"]`.
- **A04 (ingest):** если у клипа есть `clip.quality = {"motion_energy": [...]}` той же длины, что `clip.times`, гейтинг использует его. Иначе считает разницу кадров сам.
- **A06/A08:** исходный класс лежит в `Detection.source` (`<backend>:<class>[~proxy]`). Роли из предметов не выводить. При включённом open-vocab его метки приходят около 1 раза в секунду.
- **A01 (contracts/testkit/root):** в корневой `.gitignore` добавить `__pycache__/`, `weights/`, `data/`, `runs/` (сейчас стоит локальный `ml/detect/.gitignore`); зарегистрировать pytest-маркеры `slow`/`gpu` в корне (сейчас — в `ml/detect/tests/conftest.py`). Synth-видео для mock: палитра BGR в `config.yaml: mock.palette`, генератор `ml.detect.synth.make_synth_video()`. Fallback-копию `_contract_fallback.py` удалить после мерджа canonical contracts.
- **A17 (deploy):** в setup-скрипт добавить `python -m ml.detect.weights --download` и `python -m ml.detect.licenses --check`; в рантайме выставить `HF_HUB_OFFLINE=1`. Для `finetune.py` нужны `accelerate` и `albumentations` (в рантайме не нужны).
- **Предложение к контракту v2 (на решение A01):** добавить в `Detection` поле `attrs: dict` для `orig_class`, `proxy` и `query`, вместо того чтобы кодировать их в `source`.

## Решения, принятые автономно
- Baseline: только RT-DETRv2-S. Он быстрый и с заявленной Apache-2.0; остальное — флагами, чтобы не раздувать критический путь.
- При отсутствии весов detect() бросает исключение, а не возвращает пустой список. Иначе молчаливый «no_violation» замаскировал бы пропуск нарушения. Режим `empty` оставлен для интегратора.
- Mock включается только явным `SENTINEL_MOCK=1`.
- Тайлы квадратные, со стороной = короткая сторона кадра. Для портрета 464×848 это 2 тайла, а не 6–8: дешевле, и вертикальное разрешение при этом растёт ×1.8.
- WBF для TTA: бокс, найденный только в одном виде, не исчезает, а получает штраф ×0.85. Полнота важнее.
