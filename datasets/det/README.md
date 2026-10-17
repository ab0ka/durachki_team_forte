# datasets/det — разметка боксов для дообучения детектора (вид сверху)

Детекторы, обученные на COCO, плохо видят с потолочной камеры купюры, открытый сейф и инкассаторскую сумку.
Здесь — пайплайн ручной разметки поверх пред-разметки OWLv2 и результат в формате COCO.

* **В git:** код, `annotations*.json` (только боксы и относительные пути), `split.json`, этот README.
* **Не в git:** видео (`data/raw/`), кадры (`data/frames/`), веса (`weights/`), база Label Studio (`datasets/det/.ls-data/`).
  Кадры воспроизводятся из видео командой `extract_frames.py`: имя кадра однозначно задаёт видео и время.
* **Лицензии:** Label Studio Community — Apache-2.0, OWLv2 (`google/owlv2-base-patch16-ensemble`) — Apache-2.0,
  torch/numpy — BSD, transformers — Apache-2.0. Всё работает локально, облачных API нет.
  ffmpeg (LGPL/GPL) — только внешний инструмент нарезки, в продукт не входит.

## Классы

`category_id` — порядок в `common.py:CLASSES`, менять нельзя (на него завязаны веса). `contract` — имя в
`contracts.LABELS`: у контракта закрытый сейф называется `safe`, при обучении маппинг брать из
`categories[].contract_label`.

| id | класс | contract | что размечаем |
|---|---|---|---|
| 1 | `cash` | `cash` | купюры: каждая пачка/веер отдельным боксом, в руке тоже; монеты — нет |
| 2 | `cash_bag` | `cash_bag` | инкассаторская сумка, пломбированный мешок/пакет с деньгами |
| 3 | `safe_closed` | `safe` | корпус закрытого сейфа |
| 4 | `safe_open` | `safe_open` | сейф с приоткрытой хоть немного дверцей; бокс вместе с дверцей |
| 5 | `cell_phone` | `cell_phone` | телефон в руке или на столе |
| 6 | `cup` | `cup` | чашка, кружка, стакан (бутылки **не** размечаем: у контракта для них отдельный `bottle`) |
| 7 | `food` | `food` | еда и упаковки с едой |
| 8 | `money_counter` | `money_counter` | счётная машинка для купюр |
| 9 | `door_open` | `door_open` | открытая дверь: створка + проём; закрытую не размечаем |
| 10 | `person` | `person` | каждый человек (видимая сверху часть), обрезанный краем кадра — тоже |

Общие правила: бокс плотно по видимой части. Объект, видимый меньше чем на ~20% или неопознаваемый, не
размечаем. Кадр без объектов — **Submit** (негатив, тоже нужен). Непонятный кадр — **Skip** (в COCO не попадёт).
Та же инструкция показывается в Label Studio.

## Статистика и сплит

Таблица ниже перезаписывается `export_coco.py` при каждом экспорте.

<!-- STATS:BEGIN -->
Ещё не размечено: видео лежат локально у владельца данных, `annotations.json` появится после первого экспорта.
<!-- STATS:END -->

**Сплит train/val — по видео, а не по кадрам.** Соседние кадры одного ролика почти одинаковые, поэтому сплит по
кадрам дал бы утечку и завышенный mAP. Правила:

* Одинаковые по md5 файлы (`X-4≡X-47`, `X-5≡X-48`) — одна группа: кадры режутся один раз, группа целиком в одном сплите.
* Распределение хранится в `split.json` и при доразметке не меняется. Новые видео (например, из
  `datasets/extra`) распределяются так:
  * каждый класс, который встречается хотя бы в 2 группах, попадает и в val, и в train;
  * доля кадров val ≈ 20% (`--val-frac`).
* `--val X-3 X-17 ...` задаёт val вручную, `--resplit` распределяет заново.
* `images[].split`, `images[].video_id`, `images[].t_sec` и `images[].dense` (кадр из окна около таймкода)
  есть в `annotations.json`; готовые части — `annotations_train.json` и `annotations_val.json`.

> **Осторожно с eval (A16):** детектор, дообученный на кадрах всех 42 роликов, «видел» видео, на которых
> затем считается LOO по вердиктам. Для честной оценки пайплайна обучайте детектор на train-видео, а метрики
> жюри смотрите на val-видео (или делайте сплит детектора по тем же фолдам).

## Как воспроизвести (Windows, из корня репо)

### Установка (один раз)

```powershell
py -3.13 -m venv datasets/det/.venv-owl
datasets/det/.venv-owl/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
datasets/det/.venv-owl/Scripts/python -m pip install -r datasets/det/requirements-owl.txt
py -3.13 -m venv datasets/det/.venv-ls
datasets/det/.venv-ls/Scripts/python -m pip install -r datasets/det/requirements-ls.txt
```

ffmpeg берётся из PATH, переменной `FFMPEG` или пакета `imageio-ffmpeg`.

### Шаги (дальше `$PY = "datasets/det/.venv-owl/Scripts/python"`)

1. **Кадры.** Видео — `data/raw/*.mov` и `datasets/extra/**` (mov/mp4/avi/mkv/m4v). Таймкоды берутся из
   `data/labels.csv` (схема `eval.labels`, строится `tools/dataprep/build_labels.py` из `open_labels.xlsx`) и
   `datasets/extra/**/labels.csv` (столбцы `video_id` или `file`, `t_sec`/`t_all`, несколько — через `;`).
   ```powershell
   & $PY datasets/det/extract_frames.py            # шаг 1 с + 4 кадра/с в окне ±3 с около таймкодов
   ```
   Таймкоды в разметке — целые секунды и неточные (у baseline L1 медианная ошибка 3.8 с), отсюда окно ±3 с
   (`--window`). Время кадра — PTS (видео VFR). Выход: `data/frames/<video>/<video>_t0012.250.jpg`,
   `data/frames/index.csv`, `data/frames/videos.csv`. Повторный запуск дорезает только новое.
2. **Пред-разметка OWLv2** на GPU (на CPU — `--device cpu`, медленно). Первый запуск качает ~0.6 ГБ в `weights/hf/hub`.
   ```powershell
   & $PY datasets/det/prelabel_owlv2.py
   ```
   Промпты и пороги — `common.py:CLASSES`. Пороги занижены: удалить лишний бокс быстрее, чем дорисовать.
   На один объект остаётся один бокс (межклассовое подавление при IoU > 0.7): поменять класс клавишей
   быстрее, чем удалять дубль. Скорость на RTX 4050 Laptop — ~3.5 кадра/с, весь пул (~2000 кадров) — ~10 мин.
   Повторный запуск досчитывает только новые кадры.
3. **Отбор 500–800 кадров** и задачи для Label Studio:
   ```powershell
   & $PY datasets/det/build_tasks.py --target 700
   ```
   Почти одинаковые кадры статичной камеры отбрасываются: изменилось меньше 0.2% пикселей (`--dup-frac`).
   С каждого видео берётся ≥6 кадров. Остальное
   отбирается по приоритету: окно около таймкода, редкие классы в пред-разметке (cash, cash_bag, сейф,
   money_counter, дверь), разнесённость по времени.
4. **Label Studio** (два окна PowerShell):
   ```powershell
   & $PY datasets/det/ls_project.py start          # окно 1: http://localhost:8080, логин/пароль печатаются
   & $PY datasets/det/ls_project.py setup          # окно 2: проект, Local Storage на data/frames, импорт задач
   ```
   В Label Studio: открыть задачу → боксы OWLv2 уже стоят → поправить → **Submit**. Горячие клавиши
   классов — `1`…`0` в порядке таблицы.
   Скрипт работает с API по legacy-токену. С версии 1.2x он включается только при создании базы
   (`--enable-legacy-api-token` в `start`), поэтому если `.ls-data` создана раньше без него, удалите её
   до начала разметки.
5. **Экспорт в COCO** (можно запускать сколько угодно раз по ходу разметки):
   ```powershell
   & $PY datasets/det/ls_project.py export         # → annotations.json, annotations_{train,val}.json, split.json, README
   ```

Новые видео в `datasets/extra`: повторить шаги 1–3 (старый отбор сохраняется, в `tasks_new.json` попадают только
новые кадры), затем `ls_project.py setup` (импортирует новое в тот же проект) и `export`.

## Как читать COCO

```python
import json, pathlib
coco = json.load(open("datasets/det/annotations_train.json", encoding="utf-8"))
root = pathlib.Path(coco["info"]["images_root"])            # data/frames, от корня репо
img = coco["images"][0]; path = root / img["file_name"]     # data/frames/X-1/X-1_t0012.250.jpg
# bbox — [x, y, w, h] в пикселях; category_id 1..10 — как в таблице классов
```

## Файлы

| файл | что |
|---|---|
| `common.py` | классы (единый источник), пути, поиск видео, таймкоды, группы дубликатов |
| `extract_frames.py` | видео → кадры (ffmpeg) |
| `prelabel_owlv2.py` | OWLv2 → `data/frames/_owlv2/predictions.jsonl` |
| `build_tasks.py` | отбор кадров → `data/frames/_ls/tasks_new.json` |
| `ls_project.py` | запуск Label Studio, проект, импорт, экспорт |
| `export_coco.py` | JSON Label Studio → COCO + сплит + статистика в README |
| `models.json` | модели и лицензии (проверяет `scripts/check_licenses.py`) |
| `tests/` | `datasets/det/.venv-owl/Scripts/python -m pytest datasets/det/tests -q` |
