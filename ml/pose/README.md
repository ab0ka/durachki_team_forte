# ml/pose — поза COCO-17 и поза-сигналы (агент A07)

Дополнительный источник сигналов для Kassa Sentinel. Поза — **подсказка**, не доказательство:
рука у уха не доказывает телефон, рука у рта — еду, пропавшие точки — сокрытие рук.
Предметные выводы делаются только при наличии предмета (трек `cell_phone` / `cup` / `bottle` / `food`
в аргументе `tracks`) и достаточной видимости. Низкая уверенность / окклюзия → **unknown**, а не ложный ноль
и не нарушение.

## Публичные функции (сигнатуры контракта v1 не менялись)

```python
from ml.pose import estimate, pose_signals, warmup, status

poses  = estimate(clip, tracks)               # -> list[PoseFrame] (фактически PoseFrameX, наследник)
states = pose_signals(poses, tracks, zones)   # -> list[SceneState] (фактически PoseSceneState, наследник)
warmup()   # -> bool: загрузить и прогреть бэкенд; False — бэкенд недоступен
status()   # -> dict: итог последнего estimate(): state, backend, error, warnings, n_crops, stride, elapsed_s
```

* `estimate` никогда не бросает исключение из-за позы: нет весов / сбой onnxruntime → `[]`,
  `status()["state"] == "failed"` и `error` с причиной, warning в логгер `ml.pose`.
  **Поддельные (mock) позы при сбое не подставляются** — mock только по явному `SENTINEL_MOCK=1`.
* Импорт пакета без побочных эффектов; numpy/cv2/onnxruntime импортируются лениво.

### Семантика `SceneState.signals`

| значение | смысл |
|---|---|
| ключа нет | **unknown**: окклюзия, низкий score точек, нет нужного предмета, нет кассира в кадре |
| `< 0.5` | наблюдаемое «нет» (нужные точки видны; для предметных сигналов — предмет есть, но не у руки/головы) |
| `≥ 0.5` | сигнал активен после гистерезиса и требований к длительности |

Список неопределимых в момент `t` сигналов — `state.extra["unknown"]`.

| сигнал | что считаем по позе | обязательный предмет / условие | unknown, когда |
|---|---|---|---|
| `phone_to_ear` | запястье у уха/головы ≥ 1.5 с (гистерезис) | трек `cell_phone` у головы/кисти хотя бы в одном кадре отрезка | голова/кисти не видны; жест есть, телефона нет |
| `phone_in_hand` | запястье рядом с bbox телефона ≥ 1 с | трек `cell_phone` в момент t | телефона в кадре нет |
| `eating_drinking` | повторяющееся движение кисти ко рту (кисть идёт к голове, а не голова к кистям) | трек `cup`/`bottle`/`food` у кисти/рта во время подхода | жест без предмета; рот/кисти не видны |
| `hands_hidden` | обе кисти видны → `0.0` («руки не скрыты») | позитив из позы **выключен** (`allow_positive_from_pose: false`) | кисти не видны (это окклюзия, а не умысел) |
| `cashier_facing_away` | нормаль к линии плеч направлена от зоны `desk` (или на `door`) ≥ 2 с | зона `desk` (или `cash_tray`) | нет зоны; плечи не видны/«схлопнуты» |
| `handover_motion` | вытянутые кисти двух людей сближаются (над `desk`/`cash_tray`, если зоны есть) | — (это движение, не утверждение о предмете) | у кого-то из пары не видно кистей |

«Кассирские» сигналы (`phone_*`, `eating_drinking`, `hands_hidden`, `cashier_facing_away`) берутся только
по трекам с `role == "cashier"` (`signals.staff_roles`). Значения по всем людям — в `extra["by_track"]`.

### `PoseSceneState.extra`

* `gestures` — подсказки без предмета, **не доказательства**: `hand_at_head`, `hand_to_mouth`,
  `hands_not_visible` (0..1, только для кассира);
* `by_track[tid]` — `role`, `signals`, `gestures`, `bending_to_safe`, `sitting`, `bbox` для каждого человека;
* `track_ids[signal]`, `boxes[signal]` — кто и где (bbox норм.) дал активный сигнал (для `Event.track_ids/bboxes`);
* `bending_to_safe` (нужна зона `safe`), `sitting` / `standing` (эвристика; `None` при низкой уверенности);
* `handover_pairs` — `[tid_a, tid_b, role_a, role_b, value]` для активных пар;
* `n_posed`, `staff_track_ids`, `pose_status` (из `status()`), `unknown`.

### `PoseFrameX.extra`

`bbox` (норм.), `source` (`rtmpose-onnx` | `mediapipe` | `mock`), `frame_wh`, `role`, `smoothed`,
`schema: "coco17"`, опционально `raw_keypoints`. `keypoints` — всегда 17 точек в порядке COCO-17,
`(x, y)` нормированы к кадру после поворота, `score ∈ 0..1`.

## Как это устроено

* **Бэкенд (один)**: RTMPose-m (COCO-17) на onnxruntime, CUDA → CPU. Свой батчинг кропов через кадры
  (`batch_size: 16`, если ONNX с динамическим batch; иначе цикл), пре/пост-обработка как в rtmlib
  (affine-кроп с padding 1.25, SimCC-декод). MediaPipe — opt-in с **явным** маппингом
  `MP33_TO_COCO17` (тест сверяет имена точек). Whole-body (RTMW) — не по умолчанию.
* **Выбор людей / лимиты**: только `label == "person"`, приоритет по роли (cashier > collector > visitor >
  unknown) + `role_conf` + размер; `max_persons_per_frame: 3`, `max_crops_total: 2500`, `min_box_area`.
* **Бюджет**: `budget_s: 10` — после `budget_probe_frames` кадров прогнозируем время и увеличиваем шаг
  по сэмплам (до `max_stride: 5`); `status()["stride"]`, `state == "partial"`.
* **Сглаживание**: One-Euro по времени (неравномерная сетка) с confidence gating: точка со score < 0.3
  не обновляет фильтр, её координаты держатся `hold_s`, score остаётся низким.
* **Геометрия**: изотропные координаты (x × W/H кадра), расстояния в единицах масштаба тела
  (75-й перцентиль ширины плеч трека → по ушам → по bbox). Никаких пиксельных координат зон в коде —
  только зоны из аргумента `zones` (A06).
* **Гистерезис**: офлайн (видео целиком): семена ≥ `thr_on`, рост по ≥ `thr_off`, склейка разрывов
  ≤ `max_gap_s`, отсев короче `min_duration_s`. Unknown-кадры — разрывы, а не нули.

Все пороги — в `ml/pose/config.yaml` (ключи `thr_on`/`thr_off`: в YAML 1.1 `on`/`off` — булевы).

## Запуск

```bash
pip install -r ml/pose/requirements.txt
python -m pytest ml/pose/tests -q                 # CPU, без весов и сети
SENTINEL_MOCK=1 python -c "import ml.pose as p; print(p.warmup(), p.status())"
```

Веса (один раз, дальше офлайн): скачать zip из `models.json`, распаковать `*.onnx` в
`$SENTINEL_WEIGHTS_DIR/pose/` (имя файла должно содержать `rtmpose-m`) или указать путь явно
`SENTINEL_POSE_ONNX=/path/end2end.onnx`. Также находится кэш rtmlib (`~/.cache/rtmlib/hub/checkpoints`).

| env | назначение |
|---|---|
| `SENTINEL_MOCK=1` / `SENTINEL_POSE_BACKEND=mock` | mock-позы (шаблон в bbox), без весов |
| `SENTINEL_POSE_MOCK_SCRIPT` | скрипт действий mock: `phone_to_ear:2-6@1;eating:7-12;hands_hidden:...;facing_away:...` |
| `SENTINEL_POSE_ONNX` | явный путь к ONNX RTMPose |
| `SENTINEL_WEIGHTS_DIR` | корень офлайн-весов (ищем `pose/**/*rtmpose-m*.onnx`) |
| `SENTINEL_POSE_CONFIG` | YAML поверх `config.yaml` (deep-merge) |

## Модели и лицензии

См. `models.json`: RTMPose-m body7 (OpenMMLab, Apache-2.0, разрешён организаторами), MediaPipe Pose
(Apache-2.0, opt-in). Ultralytics/AGPL, CC-BY-NC и облачные API не используются.

## Проверено / не проверено

* Проверено: 39 pytest-тестов на CPU (Python 3.11, numpy 2.4, opencv 4.14, PyYAML 6.0, pytest 9.1) —
  синтетические последовательности точек и синтетическое видео через `cv2.VideoWriter` с mock-бэкендом;
  декод SimCC + обратное affine-преобразование сверены на синтетическом пике.
* **Не проверено** (в сессии не было GPU, весов и данных): реальные веса RTMPose (download.openmmlab.com
  недоступен из песочницы), скорость на RTX 4060, качество точек с потолочной камеры, пороги на реальных
  роликах. Цифры скорости не приводятся, чтобы не выдумывать измерения.

## Ограничения

* Вид сверху: ноги и часть кистей закрыты столом — поэтому много unknown, это ожидаемо.
* «Рука у щеки» и «телефон у уха» по позе неразличимы — отсюда требование трека `cell_phone`.
* Ориентация по плечам требует корректной разметки L/R моделью; при «схлопнутых» плечах (боком к
  наклонной камере) надёжность падает → unknown.
* `sitting/standing` — грубая эвристика.

## Для интегратора

1. **A06 (`ml.track.track`)**: чтобы предметные сигналы были определимы, верните в `tracks` также
   объект-треки с `label` из LABELS: `cell_phone`, `cup`, `bottle`, `food` (и `cash`, `money_counter`,
   `cash_bag` для контекста). Сигнатуры не меняются. Если объект-треков нет — `phone_*` и
   `eating_drinking` останутся unknown (это не ошибка). Роль `cashier` нужна для «кассирских» сигналов.
2. **A08 (`scene_states(..., pose_states)`)**: отсутствие ключа = unknown, не подставлять `0.0`;
   `extra` доступен как `getattr(state, "extra", {})`.
3. **A09 (rules)**: активен ⇔ `≥ 0.5`; `extra["gestures"]` не использовать как доказательство нарушения
   (только как повод для VLM-верификации A11); для `Event.track_ids/bboxes` — `extra["track_ids"]`,
   `extra["boxes"]`.
4. **A03 (pipeline)**: `ml.pose.warmup()` в `backend.pipeline.warmup()`; после `estimate` положить
   `ml.pose.status()` в `AnalysisResult.meta` (warnings при `state in {"failed","partial"}`). Позу
   держать необязательной стадией: при `[]` остальной пайплайн работает без неё.
5. **A17 (deploy)**: офлайн-вес RTMPose-m в `$SENTINEL_WEIGHTS_DIR/pose/`; `onnxruntime-gpu` требует
   CUDA 12 + cuDNN 9 (на Windows — DLL в PATH), иначе автоматически CPU.
6. **A01 (контракт)** — предложение для v2, сейчас не реализовано: добавить в `PoseFrame` и `SceneState`
   поле `extra: dict = field(default_factory=dict)` (сейчас — наследники `PoseFrameX`/`PoseSceneState`)
   и зафиксировать в контракте «нет ключа в signals = unknown».
7. После мерджа A01 удалить `ml/pose/_contract_fallback.py` (импорт в `ml/pose/_contract.py` уже
   сначала пробует `contracts.sentinel_contracts`).
8. `ml/pose/tests/__init__.py` есть намеренно: тесты импортируются как пакет `ml.pose.tests.*`, чтобы
   одноимённые тест-файлы других модулей не конфликтовали (нужен пустой `ml/__init__.py`).

TODO: тюнинг порогов на реальных роликах (A16, LOO); сверить `bgr_to_rgb` с `pipeline.json` из zip
весов; опционально RTMW (кисти) для `phone_in_hand` без детектора.

## Журнал решений

* По дополнению к ТЗ: один реальный бэкенд (RTMPose ONNX), лимиты кропов, mock не используется как
  fallback, публичные сигнатуры ровно как в контракте (без дополнительных аргументов).
* Unknown = отсутствие ключа: контракт не меняется, а «нет данных» не путается с «нет нарушения».
* `hands_hidden` из позы позитивом не бывает (по умолчанию): пропавшие точки сверху — обычная окклюзия.
* Нос не используется для ориентации по умолчанию (`nose_weight: 0`): на наклонной камере его смещение
  от плеч определяется высотой головы, а не направлением.
* Ключи порогов `thr_on`/`thr_off`: `on`/`off` в YAML 1.1 превращаются в `True`/`False`.
