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

## Сквозной прогон

```bash
python -m testkit.e2e_stub                                   # все сценарии, mock-режим
python -m testkit.e2e_stub --scenarios clean,V01 --view perspective --strict-stages
python -m testkit.e2e_stub --fixtures-only --require-accuracy   # самопроверка тест-кита (10/10)
python -m testkit.synth_video --out data/synth --all          # ролики в data/synth (в .gitignore)
```

Как выбирается реализация:

1. Если в ветке есть `backend.pipeline.run` (A03), гоняется он.
2. Иначе пайплайн собирается по стадиям контракта:
   * если модуль уже в ветке, берётся его функция;
   * иначе — фикстура.
3. Выход каждой стадии проверяется `contract_checks`. Если модуль упал или вернул невалидный выход, он заменяется фикстурой. С флагом `--strict-stages` такой прогон завершается кодом 1.

Ответ всегда проверяется строгой схемой жюри (`contracts.validate.jury_response_errors`).

## Тесты

`pytest testkit -q` — 109 тестов примерно за 4 с, без GPU и без весов. Для видео нужен `opencv-python-headless`; без него видео-тесты пропускаются (`importorskip`).

## Ограничения

* Сцена схематичная. Она проверяет форму данных, тайминги и логику, но **не качество детекторов**: метрики на реальных роликах считает A16.
* В `.mov` пишется кодек `mp4v` (OpenCV). Тег поворота в метаданных добавляет `add_rotation_metadata()`, только если в `PATH` есть `ffmpeg`.
* Таймкоды GT — «секунда события», как в разметке датасета, примерно 45% длительности ролика.
