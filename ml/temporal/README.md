# ml/temporal — обучаемая временная модель (A10)

Замороженные признаки SigLIP2 → **сначала простой обучаемый baseline**, затем temporal head для сравнения →
риск-кривая с шагом 0.5 с и события-кандидаты для fusion (A12). Это подсказка для проверяющего,
а не юридическое заключение.

> **Чего здесь нет.** У A10 не было ни видео, ни GPU. Код проверен только на синтетике и mock-признаках.
> Реальных метрик (AUC, recall, |Δt|) нет: их даст `python -m ml.temporal.cv` на машине команды.
> Никаких обещаний вида «100% recall» или «40 LOO-обучений за 10 минут».

## Что сделано

| Часть | Файл | Суть |
|---|---|---|
| Признаки | `features.py` | SigLIP2 `google/siglip2-base-patch16-224` (fp16 на CUDA). Два вида: `full` — весь кадр; `focus` — адаптивный квадратный кроп: зона `cashier_area`, если A06 передал зоны, иначе область движения. **Никаких захардкоженных координат.** Динамика: энергия и площадь движения, смена эмбеддинга. Zero-shot сходства с `prompts.yaml`. Кэш `cache/features/<sha>.npz`. Бэкенд `mock` для тестов |
| «Другой угол» | `augment.py` | Одна случайная перспектива, поворот, кроп, отражение и яркость на всё видео. Используется `ml.ingest.augment.make_view_augmenter(seed)`, если такая функция есть |
| Разметка | `labels.py` | Нормализация `X-1` / `Х-3` (кириллица), гибкий парсер `labels.csv` (`,` `;` или таб, рус/англ заголовки, «22 секунда», «0:22», «57 и 68»), группы дубликатов по md5 и по списку из конфига |
| Обучение | `dataset.py`, `predictors.py`, `model.py` | **baseline**: логрегрессия по кадрам (numpy, + скользящее среднее). **temporal**: TCN (опционально +BiGRU) + MIL. Скейлер и PCA подгоняются только на train |
| Валидация | `cv.py`, `splits.py`, `metrics.py` | Nested grouped CV и опциональный нетронутый holdout. Проверка утечек в `check_split` |
| Инференс | `infer.py`, `postproc.py` | `score(clip)`. Без чекпоинта возвращает `([], [])` |
| Бонус | `cluster.py` | k-means окон вокруг размеченных секунд + сетки кадров (локально, в `outputs/`) для быстрой разметки типов |

### Слабая разметка (как учим)
- **Позитивный ролик:** гауссиана (σ = 1.5 с) вокруг каждой размеченной секунды даёт локальную мягкую цель
  в пределах ±`far_radius_s` (3 с). **Остальная часть ролика — маска (неизвестно), а не негатив**
  (`far_weight: 0`).
- **Чистый ролик:** негатив на всех шагах.
- **temporal head:** дополнительно MIL уровня окна (top-k среднее). Если окно позитивного ролика не содержит
  метку, MIL-потеря для него маскируется.
- **Дубликаты** (Х-4≡Х-47, Х-5≡Х-48, а также любые совпадения md5) попадают в одну группу сплита.
  Признаки берутся один раз, **все исходные метки сохраняются** (`sources` в индексе и отчёте),
  обе секунды 57 и 68 остаются якорями. Расхождение «нарушение / без» помечается `conflict`
  и решается явной политикой `labels.conflict_policy: positive | exclude`. Произвольного выбора нет.
- **Голова типа** учится **только** при проверенном столбце `type_id` (V01…V99) в `labels.csv`.
  Без него `type_id = "V99"` и `extra.type_source = "none"`. Zero-shot сходство лежит отдельно
  в `extra.zero_shot_hint` с `trained: false`, и за классификацию оно не выдаётся.

### Протокол оценки (без утечек)
- Внешний сгруппированный сплит (`loo` или K) даёт оценку. Внутренний сплит (только train внешнего фолда)
  калибрует порог. Скейлер, PCA, модель и порог подгоняются на train каждого фолда.
- В валидацию идут **только оригиналы**. Аугментированные копии и дубликаты валидационной группы
  не попадают ни в валидацию, ни в train этого фолда (тесты `test_splits.py`).
- В отчёте `reports/temporal_cv.json`:
  - `threshold_free` — AUC и таймкод |Δt|, hit@1/2/3 с. От порога не зависят.
  - `recall_at_fp*` — помечены как «оракульные»: порог подобран по тем же OOF, оценка оптимистична.
  - `nested` — recall и FPR при пороге, подобранном внутренним CV. Это **независимая** оценка.
  - `deploy_threshold` — порог по всем OOF для прода. **Не** является оценкой качества.
  - `holdout` (если задан `--holdout`) — одна оценка выбранной модели на группах, которые не участвовали
    ни в подгонке, ни в выборе.
- `selected`: temporal выбирается только если его AUC строго выше baseline. Выбор между двумя моделями
  по OOF даёт лёгкий оптимистичный сдвиг — для честной цифры используйте `--holdout`.

## Публичные функции
```python
from ml.temporal import score, warmup, status
risk, events = score(clip)                      # контракт: score(clip) -> (risk_curve, events)
risk, events = score(clip, zones=zones, budget_s=20.0)   # необязательные расширения
warmup()                                        # загрузить чекпоинт и SigLIP2 заранее
status()                                        # диагностика последнего вызова: ok | no_checkpoint | backend_mismatch | error
```
- `risk`: `[(t, r)]`, шаг 0.5 с, `r ∈ [0, 1]` — сглаженная вероятность модели.
- `events`: `Event(source="temporal", type_id="V99" | тип из обученной головы, confidence=пик риска,
  t_start/t_end — где риск ≥ 0.5·пик)`. `facts` — нейтральная фраза про повышенный риск в интервале.
  `bboxes["temporal_focus"]` — кроп focus в момент пика. `extra`: kind, порог, `type_source`,
  `zero_shot_hint`.
- Нет чекпоинта, чекпоинт не совпадает с режимом (mock/siglip) или произошла ошибка → `([], [])`
  и предупреждение в `status()`. Пайплайн не падает.

## Как запустить (на машине команды, RTX 4060)
```bash
pip install -r ml/temporal/requirements.txt
# один раз скачать веса (дальше всё офлайн; по умолчанию модуль сам ставит HF_HUB_OFFLINE=1):
SENTINEL_OFFLINE=0 python -c "from transformers import AutoModel, AutoProcessor as P; m='google/siglip2-base-patch16-224'; AutoModel.from_pretrained(m); P.from_pretrained(m)"
python -m ml.temporal.extract --videos data/raw --labels data/labels.csv      # cache/features/*.npz + index.json
python -m ml.temporal.cv --kind both                                           # reports/temporal_cv.json
python -m ml.temporal.train --kind both                                        # weights/temporal/{baseline,temporal}/ + selected.json
python -m ml.temporal.cluster --k 5                                            # outputs/temporal/clusters/ (локально)
python -m pytest ml/temporal/tests -q                                          # без GPU и без весов
```
Опции: `cv --outer loo|5 --inner 3 --holdout 0.2 --epochs N --max-aug K`;
`train --select auto|baseline|temporal --threshold auto|0.6`.
Тип нарушения можно добавить в `labels.csv` столбцом `type_id`, после чего голова типа начнёт обучаться.

## Что реально проверено (контейнер без GPU, синтетика)
- `pytest ml/temporal/tests`: **30 passed** (Python 3.13, numpy 2.5, torch 2.14 CPU, transformers 5.20.1,
  opencv 4.x).
- Сквозной тест в mock-режиме: синтетические видео (X/Х вперемешку, md5-дубликат с конфликтом меток)
  → `extract` → `cv` → `train` → `score`.
- Путь SigLIP-бэкенда проверен на **крошечной случайной** SiglipModel (формы, нормировка). Настоящие
  веса SigLIP2 не скачивались и не запускались.
- Скорость temporal head на CPU контейнера (4 потока), синтетический батч 8×192 шага×137 признаков:
  TCN ~31 мс/шаг (полное обучение 720 шагов ≈ 22 с); TCN+BiGRU ~156 мс/шаг (≈ 113 с).
  На RTX 4060 не мерилось. Nested CV по умолчанию: baseline — LOO снаружи; temporal — 5×(3+1) = 20 обучений.

## Ограничения
- Данных мало (~40 видео, одна секунда на нарушение, тип не размечен). Ждите высокой дисперсии метрик
  и смотрите на `nested` и `holdout`, а не на оракульные цифры.
- Zero-shot подсказки (`prompts.yaml`) не калиброваны на CCTV сверху. Это только признаки для обучаемой
  модели и подсказка в `extra`.
- Режим CPU: 1 fps и только вид `full` (недостающий `focus` заполняется нулём — так же, как view-dropout
  на обучении). Качество ниже, чем на GPU.
- Длинные видео: не больше `max_frames` (1200) кадров признаков. При нехватке бюджета `score()` обрезает
  извлечение (`status()["truncated_at"]`).
- Focus по движению: прямоугольник вокруг области движения, без детектора. При статичной сцене
  совпадает с полным кадром.

## Для интегратора
- **Вызов в пайплайне (A03):** `ml.temporal.score(clip, zones=zones, budget_s=...)`. `zones` (от A06)
  необязательны, но улучшают кроп. По сигнатуре контракта `score(clip)` тоже работает.
  `ml.temporal.warmup()` вызывать из `backend.pipeline.warmup()`. В `meta.warnings` класть
  `ml.temporal.status()`, если там не `ok`.
- **Fusion (A12):** события temporal несут `type_id="V99"` и `extra.type_source="none"`, пока нет
  проверенных типов. Их стоит использовать как поддержку и таймкод для событий rules/vlm, а не как
  самостоятельный тип. `extra.zero_shot_hint` — только подсказка.
- **env:** `SENTINEL_MOCK=1` — mock-признаки (нужен mock-чекпоинт, иначе `([], [])`).
  `SENTINEL_CACHE_DIR`, `SENTINEL_WEIGHTS_DIR` (→ `<dir>/temporal`), `SENTINEL_TEMPORAL_WEIGHTS` (точный путь).
  `SENTINEL_TEMPORAL_CONFIG` — yaml поверх `config.yaml`. `SENTINEL_OFFLINE=0` — разрешить скачивание весов.
- **Деплой (A17):** веса SigLIP2 положить в HF-кэш. `weights/temporal/` — результат `train`.
  Ни то, ни другое не коммитить.
- **A04:** если появится `ml.ingest.augment.make_view_augmenter(seed) -> Callable[[frame], frame]`,
  extract подхватит его автоматически. `_video.SimpleClip` — fallback-загрузчик, пока нет `ml.ingest.load_video`.
- **A01:** `_contract_fallback.py` — копия контракта. Удалить после мерджа `contracts/`; импорт уже
  через `try: from contracts.sentinel_contracts import ...`. Пустой `ml/__init__.py` добавлен
  (разрешённое исключение).
- **TODO:** откалибровать `prompts.yaml` и `infer.*` на реальных данных. Сравнить baseline и temporal
  по `nested`/`holdout`. Если команда разметит типы — добавить `type_id` и переобучить.
  Опционально — ONNX-экспорт SigLIP2 для CPU.

## Решения, принятые автономно
- Шаг сетки модели 0.25 с (4 fps), выход 0.5 с.
- `full` подаётся сжатым до 224×224 (без паддинга), `focus` — квадратный кроп.
- PCA эмбеддингов до 32 на вид (подгонка только на train), чтобы не переобучиться на ~40 видео.
- Порог в проде — `deploy_threshold` из CV или `infer.event_threshold`; источник записывается в `meta.json`.
- По умолчанию `arch: tcn`, потому что GRU на CPU в 5 раз медленнее.
