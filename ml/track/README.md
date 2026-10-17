# ml/track — трекинг, роли без лиц, авто-зоны (A06)

`track(clip, dets) -> (list[Track], list[Zone])` — сигнатура по замороженному контракту v1.
Моделей и весов нет: своя реализация ByteTrack на numpy и геометрические эвристики.
`models.json` поэтому пустой (`[]`).

## Что сделано
| Часть | Файл | Суть |
|---|---|---|
| ByteTrack | `bytetrack.py` | Своя реализация (идея ByteTrack: двухэтапное сопоставление high/low score). Фильтр Калмана, учитывающий dt, с затуханием скорости. Стоимость = 1 − [(1−w)·IoU + w·близость центров]: при 5 fps у идущего человека IoU между кадрами бывает 0. Время в секундах, пропуск кадров детектором (`every_n`) определяется автоматически. Отдельные трекеры по группам классов: `person`; `safe`+`safe_open` → трек `safe`; `door`+`door_open` → `door`; `cash_bag`+`bag` → `cash_bag`, если ≥30% голосов за неё, иначе `bag`; остальные метки — каждая своя группа. |
| Склейка | `stitch.py`, `appearance.py` | Короткий разрыв (≤2,5 с, окклюзия или пропуск детектора) склеивается по положению с растущим допуском, внешность может запретить склейку. Длинный разрыв (ушёл и вернулся) — **только** при совпадении HSV-гистограммы **одежды** (кроп без верхних 25% бокса и без краёв; лица не используются) и правдоподобном месте: вернулся на своё место или появился там же, где исчез. Без внешности длинные разрывы не склеиваются. Статичные предметы склеиваются по IoU медианных боксов (мерцание до 15 с). |
| Авто-зоны | `zones.py` | Без единой захардкоженной координаты; выпуклая оболочка и масштаб от центроида аффинно-эквивариантны. **Независимые опоры** (строятся из предметов, не из людей): `desk` — оболочка медианных боксов laptop/money_counter и статичных cell_phone/keys/document; `safe` — медианный бокс safe/safe_open; `door` — детекции door/door_open и кластер точек появления/исчезновения людей; `cash_tray` — кластер cash (fallback — окрестность money_counter, conf 0,2). **Производные**: `cashier_area` — места кассира ∪ «отпечаток» стола, сдвинутый на его сторону; `client_area` — люди по другую сторону стола, иначе зеркало `cashier_area` через центр стола (conf ≤ 0,3). Если опор нет, зоны нет или у неё низкий `conf`. Зона «весь кадр по умолчанию» не создаётся никогда. |
| Роли | `roles.py` | См. ниже: `cashier` / `collector` / `visitor` / `unknown` + `role_conf` + provenance. |
| Утилиты A08/A09 | `events.py` | `entries_exits`, `dwell`, `proximity`, `gaps`, `presence_state`, `person_signals`, `box_at`, `zone_by_name`. |
| Отладка | `viz.py`, `serialize.py` | `python -m ml.track.viz video.mov --cache cache/<sha>`; кэш стадии `cache/<sha>/track.json`. |

### Роли — принципы
1. **Кассир** выбирается по признакам, не зависящим от зон людей: близость к объектам стола и к их центру, присутствие с начала ролика, неподвижность (скорость по точкам с шагом ~1 с), сторона сейфа. `cashier_area` строится **после** выбора кассира и для повторного «назначения кассиром» сама по себе не используется. Цикла «кассир, потому что в cashier_area; cashier_area, потому что там кассир» нет.
2. Нет объектов стола → fallback: кассир — самый неподвижный человек, присутствующий с начала. `role_conf ≤ 0,45`, зоны `cashier_area`, `desk` и `client_area` получают `conf ≤ 0,3`, `visitor ≤ 0,6`, в причинах — `fallback_no_desk_objects`.
3. **non-cashier ≠ посторонний.** Второй человек в зоне кассира одновременно с кассиром получает `unknown` (`co_present_in_cashier_area`, плюс `entered_mid_clip`, если пришёл по ходу ролика): у него может быть допуск. Близкий по score второй кандидат в кассиры → `unknown` (`ambiguous_cashier_candidate`), а conf кассира снижается.
4. **Сумка сама по себе роль не подтверждает.** `collector` = контакт с `cash_bag`/`bag` **и** независимый признак прихода: вошёл по ходу ролика или сумка появилась вместе с ним. Только контакт с сумкой → `unknown` (`bag_contact_only`). Спутник инкассатора без своей сумки → `unknown` (`arrived_with_collector`).
5. **Уход и возврат.** Возврат с совпадающей одеждой — тот же трек (склейка) или `cashier` (`clothes_appearance_match`). Только «то же место» без внешности → `unknown` (`possible_cashier_return`, `position_only_no_appearance`). Другая одежда на месте кассира → `unknown` (`possible_shift_change`, если остался до конца ролика).
6. Короче `min_presence_s` (1 с) → `unknown` (`too_short`). У предметов всегда `role="unknown"`, `role_conf=0`.

### Пропал из кадра ≠ ушёл
- `entries_exits()` по умолчанию **не** создаёт `exit` из-за пропуска детекций. Выход фиксируется только по наблюдаемому пересечению границы зоны с гистерезисом 0,4 с. Режим «пропуск > N с = выход» включается явно: `gap_as_exit_s=N`.
- `gaps(track, zones, t_end=...)` классифицирует каждый пропуск:
  - `lost_in_view` — пропал в глубине кадра: окклюзия или пропуск модели, **не уход**;
  - `left_view_edge` — последний раз виден у края кадра: вышел из поля зрения, но не обязательно из помещения;
  - `left_via_door` — последний раз виден в зоне `door`.
- `person_signals()`: кассир не виден, но `lost_in_view` → `cashier_present = 0.5` («неизвестно»), а не 0.
- В `Track.boxes` только **реальные** наблюдения с реальными timestamps детекций (`fill_gaps_s: 0`). Интерполяция включается в конфиге, но выключена по умолчанию.

## Публичные функции
```python
from ml.track import track, track_full, entries_exits, dwell, proximity, gaps, presence_state, person_signals
tracks, zones = track(clip, dets)          # контракт; dets[i].track_id проставляется на месте
res = track_full(clip, dets, cfg=None)     # TrackResult(tracks, zones, info) — info = provenance
res.info["roles"][tid]   # {"role","conf","reasons":[...],"features":{...},"in_cashier_area",...}
res.info["zones"][name]  # {"method","independent","evidence"/"tracks",...}
res.info["stitch"]       # журнал склеек: kind short_gap|long_gap_appearance, gap_s, app_dist
res.info["cashier_selection"], res.info["appearance"], res.info["timings"], res.info["warnings"]

entries_exits(tracks, zones, *, anchor="center", min_dwell_s=0.4, enter_at_start=True, gap_as_exit_s=None)
    -> list[(t, track_id, zone_name, "enter"|"exit")]
dwell(track, zone, t0=None, t1=None, *, max_dt=1.0) -> float          # секунды наблюдённого присутствия
proximity(track_a, track_b, t, *, max_dt=0.6, aspect=1.0) -> float    # 0..1; 0, если кого-то нет в момент t
gaps(track, zones=None, *, min_gap_s=0.6, edge_margin=0.06, t_end=None) -> list[dict]
presence_state(track, t, zones=None) -> "observed"|"not_yet"|"lost_in_view"|"left_view_edge"|"left_via_door"
person_signals(tracks, zones, times, *, extended=False) -> list[SceneState]
    # cashier_present, cashier_in_workplace, n_persons, outsider_in_cashier_area, collector_present
    # outsider_in_cashier_area: visitor -> role_conf, unknown -> 0.5, collector/cashier -> 0
```
Вход принимает как `Detection`, так и dict с теми же ключами. Битые боксы пропускаются, в `info.warnings` пишется предупреждение. Пустые или `None` детекции дают `([], [])`.

## Запуск
```bash
pip install -r ml/track/requirements.txt
python -m pytest ml/track/tests -q                     # 35 тестов, без GPU/весов/данных (~8 с)
python -m ml.track.viz video.mov --cache cache/<sha> [--out out.mp4] [--recompute] [--blur-heads]
```
`viz` берёт детекции из `cache/<sha>/detections.json` (или `dets.json`, `detect.json`, `*.pkl`), а если их нет — из `ml.detect.detect(clip)`. Кадры читает через `ml.ingest.load_video`, иначе через встроенный `SimpleClip` на cv2 с автоповоротом. Пишет `track.json` и `track_viz.mp4`. Видео не коммитить: в `ml/track/.gitignore` они уже исключены.

## Проверки (реально выполнено)
- `pytest ml/track/tests` — 35 passed. Покрыто:
  - пустые и битые детекции;
  - стабильность ID при дрожании и выпадении 20% детекций;
  - детекция через кадр (`every_n=3`);
  - продление трека «слабыми» детекциями;
  - два пересекающихся человека без смены ID;
  - сценарии clean, V01, V02 (с внешностью и без), «на место кассира сел другой», V07, «сумка без прихода», «два сотрудника» (3 сида), отсутствие объектов стола;
  - зоны под аффинным «поворотом камеры» (±25…35°, масштаб, зеркало), поворот на 90° в альбомную ориентацию;
  - события, кэш, viz CLI.
- Скорость на синтетике (облачный CPU, без GPU): 145 с, 725 кадров, 4371 детекция → **1,43 с** на всю стадию. Реальные ролики не измерялись: данных здесь нет.

## Ограничения (честно)
- Пороги не тюнены на реальных данных: синтетика ≠ реальная сцена. Все пороги — в `config.yaml`.
- Смена ID при плотном контакте людей (передача сумки «из рук в руки») в ассоциации не исправляется: внешность используется только при склейке треклетов.
- Без кадров (`clip.frame` недоступен) или без cv2 внешности нет. Тогда длинные разрывы не склеиваются, возврат кассира → `unknown`.
- Гистограмма одежды не различает людей в одинаковой форме, например двух кассиров в одинаковой униформе.
- Сторону стола определяет позиция кассира относительно центра объектов стола. При сильно наклонном ракурсе (камера сбоку) «отпечаток» стола — грубое приближение.
- `door` без детекции двери строится по кластеру появлений и исчезновений; при одном входе получается низкий conf или зоны нет.
- Роли — подсказки для правил, не утверждения о допуске или личности.

## Для интегратора
- **Подключение** в `backend.pipeline`: `tracks, zones = ml.track.track(clip, dets)`. Для provenance вызывать `ml.track.track_full(...)` и класть `res.info` в `AnalysisResult.meta["track"]` или в кэш через `ml.track.serialize.save_track_cache(cache_dir, tracks, zones, info)`.
- `track()` **проставляет `track_id` в переданные `Detection`** (in-place) — A08 может связывать детекции с треками.
- **A08/A09**: `entries_exits`, `dwell`, `proximity`, `gaps`, `presence_state` и `person_signals` — чистые функции над контрактными `Track`/`Zone`. Рекомендация: не считать `unknown` автоматически посторонним (вес 0,5 в `person_signals`), а `lost_in_view` не считать уходом.
- **ENV**:
  - `SENTINEL_TRACK_CONFIG=/path.yaml` — свой конфиг;
  - `SENTINEL_TRACK_APPEARANCE=0` — выключить гистограммы одежды;
  - `SENTINEL_MOCK=1` — ни на что не влияет: моделей нет, модуль всегда работает одинаково.
- **Контракт**: импорт через `ml/track/contract.py`: `contracts.sentinel_contracts`, а если его нет — `_contract_fallback.py`. Fallback удалить после мерджа A01.
- **Предложение A01 (не реализовано самостоятельно)**: добавить в `Track` поле `extra: dict` (provenance роли: `reasons`, `features`) и в `Zone` поле `extra: dict` (`method`, `independent`). Сейчас это только в `track_full().info`.
- **TODO на реальных данных**:
  1. Подобрать `roles.*` и `zones.*` по ~40 роликам: где реально сидит кассир, как детектор видит дверь.
  2. Проверить `cash_bag` против `bag` у A05.
  3. Проверить, нужен ли якорь `torso` или `bottom` вместо `center` для другого ракурса.
