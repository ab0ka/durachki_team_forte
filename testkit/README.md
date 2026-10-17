# testkit/ — общий тест-кит Kassa Sentinel (владелец A01)

Тест-кит позволяет интегрировать и тестировать модули **без реальных видео, весов и GPU**. Источник истины — детерминированные сценарии `testkit/scenarios.py`. Из одного сценария строятся и синтетическое видео, и «идеальные» выходы каждой стадии, поэтому они согласованы между собой.

Синтетическая сцена: вид сверху, стол с ноутбуком, стационарным телефоном, счётчиком банкнот и лотком наличных; серый сейф на тумбе с открывающейся дверцей; стеклянная перегородка с окном и дверью. Люди нарисованы эллипсами «плеч» с головой и руками. Наличные — зелёные пачки, инкассаторская сумка — зелёная, с ремнём и пломбой.

| Сценарий | GT | Что происходит в секунду `t` (≈45% ролика) | Ключевой сигнал |
|---|---|---|---|
| `clean` | no_violation | клиент у окна, пересчёт, передача наличных через окно | — |
| `clean_collector` | no_violation | инкассатор передаёт сумку кассиру из рук в руки | — |
| `V01` | V01 | посторонний проходит через дверь за кассовый стол | `outsider_in_cashier_area` |
| `V02` | V02 | кассир ушёл из кадра, наличные на столе | `workplace_unattended_with_cash` |
| `V03` | V03 | сейф открыт, кассир отошёл, у окна клиент | `safe_open` |
| `V04` | V04 | дверь открыта, в проёме человек | `door_open` |
| `V05` | V05 | телефон у уха при наличных на столе | `phone_to_ear` |
| `V06` | V06 | стакан у рта, потом еда; наличные на столе | `eating_drinking` |
| `V07` | V07 | инкассаторская сумка оставлена на полу без присмотра | `cash_bag_visible` |
| `V08` | V08 | кассир отвернулся, руки с наличными скрыты | `hands_hidden` |

Ракурсы `view`: `default`, `mirror`, `rot180`, `landscape` (поворот на 90°), `perspective` (наклон камеры). Ими проверяется устойчивость к «другому углу». Пиксельные координаты зон хардкодить нельзя.

## API

```python
from testkit.synth_video import make_video, render_frames
gt = make_video(tmp_path / "v01.mov", "V01", duration=20, fps=30, size=(464, 848))   # + view=, seed=, osd=
# gt: scenario, verdict, type_id, t, t_start, t_end, duration, description, fps, size, n_frames, view, path, fourcc

from testkit.fake_clip import FakeClip, clip_from_video
clip = FakeClip.from_scenario("V03", duration=10, fps=10, size=(232, 424))   # VideoClip в памяти, кадры лениво
clip = clip_from_video("v01.mov", sample_fps=5.0)                             # VideoClip через OpenCV

from testkit.fixtures import make_bundle, make_analysis_result, make_catalog
b = make_bundle("V02", duration=20, sample_fps=5, view="perspective", noise=0.02)
b.detections, b.tracks, b.zones, b.poses, b.states, b.events, b.gt, b.catalog
result = make_analysis_result("V05")        # AnalysisResult «как от fusion» (для API/UI/eval)

from testkit.contract_checks import check_detections, check_scene_states, check_events, check_result
assert check_detections(ml.detect.detect(clip)) == []        # выход стадии соответствует контракту
```

Отдельные генераторы: `make_detections` (с `noise`, `dropout`, `tracked`), `make_tracks`, `make_zones`, `make_pose_frames` (COCO-17), `make_scene_states` (все 17 `SIGNALS`), `make_expected_events`, `make_evidence_b64`.

`make_catalog()` — заглушка той же формы, что `norms/catalog.yaml`, но **без норм** (`rules: []`). Настоящие ссылки на нормы берутся только из каталога A13.

## v2: кейсы спецификации команды (`testkit/spec.py`)

Синтетика проверяет **механизмы** (контракт, логику, тайминги, «неизвестно»), а **не качество модели**. Метрики считаются только на реальных данных у L2.

- **Кейсы — пары «норма / нарушение»** из спецификации, всего 21:

  | Группа | Норма | Нарушение |
  |---|---|---|
  | Уход кассира | `leave_clean` — убрала всё, сейф закрыт | `leave_cash`, `leave_bag` (тип V02, альтернатива V07), `leave_safe_open` (V03, альтернатива V02) |
  | Клиенты и посторонние | `client_opposite` | `outsider_cashier_side`, `second_outsider` (V01) |
  | Порядок появления | `cashier_enters_first`; `client_first` — роль по зоне, а не по порядку | — |
  | Свет выключили | `lights_off_clean` | `lights_off_cash` |
  | Еда, рука у лица | `hand_to_face` — не сигнал | `drinking` (V06) |
  | Телефон | `desk_phone` — стационарный, норма | `mobile` (V05) |
  | Личные вещи | — | `jacket_on_desk` (V09) |
  | Инкассаторская сумка | `bag_handover_proper` — из рук в руки и сразу в сейф | `bag_left` (V07) |
  | Сейф | — | `safe_open_working` (V03) |
  | Несколько событий | — | `multi_event` (V05, затем V02), `debounce_phone` (одно событие) |
- **Планировки:** `A` (горизонтальная перегородка) и `B` (вертикальная перегородка, стол вдоль неё, сейф внизу); ориентации `portrait` и `landscape`. В логике нет пиксельных констант: стороны кассира и клиента — полигоны планировки.
- **Тайминги:** длительность и момент события — из одного распределения для всех классов. `sample_timing(rng, duration_range)` класса не знает, поэтому длительность и позиция события не становятся признаками.
- **GT:**
  - `events` — список событий, таймкод по началу действия (переход: отошла, вошёл, поднесла);
  - `alt_type_ids` — допустимые альтернативные типы;
  - `unknown_spans` — интервалы, где сцена не видна (свет выключили).
- **Неизвестно ≠ нет:** в темноте в `signals` остаётся только `scene_visibility`. Если кассира нет в кадре, его сигналы (телефон, еда, руки, взгляд) отсутствуют, а не равны 0.

```python
from testkit.spec import make_suite, build_case, case_bundle, make_case_video, CaseSpec
for spec in make_suite(seed=0):                  # 21 кейс × 2 планировки × 2 ориентации
    gt = build_case(spec).gt.as_dict()           # verdict, events[...], unknown_spans
    b = case_bundle(spec)                        # детекции/треки/зоны/позы/сигналы/ожидаемые события
make_case_video("x.mov", CaseSpec("leave_cash", "B", "landscape", duration=20, onset=9), fps=10)
```

```bash
python -m testkit.e2e_stub --suite spec --strict-stages --layouts A,B --orientations portrait,landscape
```

Сценарии v1 (`testkit/scenarios.py`) сохранены для совместимости. Событие в них всегда на 45% ролика, поэтому для оценки модели они не подходят — только для проверки механики.

## Сквозной прогон

```bash
SENTINEL_MOCK=1 python -m testkit.e2e_stub                   # все сценарии, mock модулей включён явно
python -m testkit.e2e_stub --scenarios clean,V01 --view perspective --strict-stages
python -m testkit.e2e_stub --fixtures-only --require-accuracy   # самопроверка тест-кита (10/10)
python -m testkit.synth_video --out data/synth --all          # ролики в data/synth (в .gitignore)

# полная сборка (как job e2e-full в CI): оркестратор + канонический сериализатор жюри
SENTINEL_MOCK=1 python -m testkit.e2e_stub --require-full --expect-status done --scenarios clean,V01,V03,V07 --duration 8 --fps 30
python -m testkit.e2e_stub --require-full --expect-status failed --scenarios clean --duration 8 --fps 30   # без mock
```

`e2e_stub` не меняет `os.environ`: mock модулей включает вызывающий (`SENTINEL_MOCK=1` в команде или в шаге CI). Mock-результат оркестратора без явного режима — ошибка: production сам на mock не переключается.

* `--require-full` — обязателен `backend.pipeline.run`, фикстуры запрещены, ответ — только через `contracts.serialize.result_to_jury_api`. Нет оркестратора → «NOT RUN», код 3 («ПОЛНАЯ СБОРКА НЕ ПРОВЕРЕНА»), а не зелёный результат.
* `--expect-status done|failed|processing` — статус каждого ответа. `done` ловит регрессию «нет `meta.coverage` → `failed`».
* `--deadline` (по умолчанию 240 с, как в ТЗ): прогон дольше дедлайна +10% — ошибка. Меньший дедлайн меняет адаптивный `sample_fps` оркестратора, поэтому для проверки сборки его не уменьшают.
* Коды возврата: 0 — ок; 1 — ошибка (схема жюри, падение, дедлайн, статус, точность); 3 — NOT RUN.

Как выбирается реализация:

1. Если в ветке есть `backend.pipeline.run` (A03), гоняется он.
2. Иначе пайплайн собирается по стадиям контракта:
   * если модуль уже в ветке, берётся его функция;
   * иначе — фикстура.
3. Выход каждой стадии проверяется `contract_checks`. Если модуль упал или вернул невалидный выход, он заменяется фикстурой. С флагом `--strict-stages` такой прогон завершается кодом 1.

Ответ всегда проверяется строгой схемой жюри (`contracts.validate.jury_response_errors`).

## Тесты

`pytest testkit -q` — 228 тестов (v1 и v2) примерно за 8 с, без GPU и без весов. Для видео нужен `opencv-python-headless`; без него видео-тесты пропускаются (`importorskip`).

## Ограничения

* Сцена схематичная. Она проверяет форму данных, тайминги и логику, но **не качество детекторов**: метрики на реальных роликах считает A16.
* В `.mov` пишется кодек `mp4v` (OpenCV). Тег поворота в метаданных добавляет `add_rotation_metadata()`, только если в `PATH` есть `ffmpeg`.
* Таймкоды GT — «секунда события», как в разметке датасета, примерно 45% длительности ролика.
