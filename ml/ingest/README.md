# ml/ingest — декодирование видео, сэмплинг, качество, синтетические искажения (A04)

Модуль превращает `.mov`/`.mp4` в `Clip` — реализацию контрактного `VideoClip`: сэмплы
(по умолчанию 5 fps) с **фактическими PTS**, точный доступ к кадрам исходника для
доказательств, метрики качества и `motion_energy`. Моделей нет (`models.json = []`).

## Публичный API

```python
from ml.ingest import load_video, IngestError

clip = load_video("X-1.mov")                 # контракт: load_video(path, sample_fps=5.0)
clip.path, clip.duration, clip.fps, clip.width, clip.height   # width/height — ПОСЛЕ поворота
clip.times          # фактические PTS выбранных кадров, сек от первого кадра (первый = 0.0)
clip.frame(i)       # BGR uint8 сэмпла i (из LRU — копия)
clip.frame_at(t)    # контракт: точный кадр исходника, ближайший к t, полный размер (только картинка)

# расширения (совместимы с контрактом, ничего не ломают)
img, ts = clip.frame_at_ts(t)   # + ФАКТИЧЕСКИЙ timestamp кадра -> EvidenceFrame.timestamp_sec
ref = clip.read_exact(t)        # FrameRef(image, timestamp_sec, requested_sec)
refs = clip.frames_at([t1, t2]) # пачкой, один проход декодера
clip.snap_time(t)               # PTS ближайшего кадра исходника (по таймлайну загрузки)
clip.index_at(t)                # ближайший сэмпл
clip.source_times               # PTS всех кадров потока
clip.quality                    # list[dict] на каждый сэмпл (см. ниже)
clip.motion_curve()             # [(t, motion_energy)] — формат timeline
clip.activity_segments()        # [(t0, t1)] участки движения
clip.gated_indices()            # индексы сэмплов для тяжёлых детекторов (чаще в движении)
clip.meta                       # probe, backend, orientation, coverage, complete, vfr, timings, warnings
clip.warnings                   # предупреждения по-русски (тёмное видео, неполный файл, ...)
clip.close()                    # или `with load_video(...) as clip:`

load_video(path, sample_fps, *, config=None, backend=None, progress=None)  # keyword-only расширения
load_video("synthetic://10")    # Clip в памяти без файла (для тестов других модулей)
```

Прочее: `ml.ingest.probe.probe/validate` (+ CLI `python -m ml.ingest.probe video.mov`),
`ml.ingest.augment`, `ml.ingest.contact_sheet`, `ml.ingest.names` (`normalize_video_id`,
`find_duplicates`), `ml.ingest.synthetic` (тестовые видео со штрих-кодом номера кадра).

## Решения

* **Время = PTS.** `t = pts * time_base − t0`, t0 — PTS первого выданного декодером кадра.
  `frame_index / fps` временем не считается нигде. VFR, пропуски кадров, ненулевой start_time
  и edit lists обрабатываются одинаково. Для узлов сетки `k / sample_fps` берётся **ближайший
  по PTS** кадр (без повторов); `clip.times` — PTS выбранных кадров, а не узлы сетки.
* **Поворот.** Display matrix (с кадра PyAV / из ffprobe) или тег `rotate`; логика повторяет
  autorotate ffmpeg, включая варианты с зеркалом. Размер берётся из декодированного кадра —
  никаких зашитых 464×848. Тест сверяет наши кадры с выводом `ffmpeg` (autorotate) для
  −90/90/180/−90+hflip.
* **Бэкенды.** PyAV (основной) → OpenCV (фолбэк, авто-поворот OpenCV выключается, поворот
  применяем сами). Проба: ffprobe → PyAV → OpenCV.
* **Память.** Сэмплы — JPEG q=92 в RAM до `storage.max_ram_mb` (512), сверх — во временный
  файл (удаляется при `close()`/сборке мусора). LRU декодированных сэмплов ограничен и по
  числу (96), и по МБ (256); кэш точных кадров — 32 шт / 128 МБ. Сэмплы ужимаются до
  `max_side=1280` (для 464×848 — без изменений); `frame_at` — всегда полный размер.
* **Ошибки — не «пустой успех».** Нет кадров / ошибок декодера больше `max_decode_errors` /
  декодировано < `min_coverage` (50%) заявленной длительности → `DecodeError`
  (`code="decode_failed"|"incomplete_video"`). 50–98% → Clip с `meta["complete"]=False`,
  `meta["coverage"]` и предупреждением. Файл без `moov`, мусор, 0 байт → `VideoProbeError`;
  нет видеопотока → `NoVideoStreamError`; длительность вне 1–900 с → `InvalidDurationError`.
  У всех: `.code`, русское `.message`, `.to_dict()` (A02 может отдавать 422).
* **sample_fps.** Default контракта (5.0) не менялся. Можно передать любое значение
  (`<= 0` — каждый кадр); `sampling.default_fps` в конфиге — для пайплайна/CLI;
  `sampling.max_samples` (6000) снижает частоту с предупреждением, чтобы не съесть RAM.
* **Скорость.** Декод в потоках FFmpeg (`thread_type=AUTO`), конвертация кадров + JPEG + метрики —
  в пуле потоков. `decode.adaptive_skip`: если прогноз декода > `budget_s` (20 с), декодер
  переключается на опорные кадры (I/P) с предупреждением; `decode.skip_nonref: true` — сразу.

### `clip.quality[i]`
`t, sharpness (var Laplacian на 160 px), brightness, contrast, dark_frac, bright_frac,
motion_raw, motion_frac, motion_energy (= min(1, motion_frac / 0.06)), is_blurry, is_dark,
is_bright, is_frozen`. Все пороги — `config.yaml` (`quality.*`, `activity.*`).

## Синтетические искажения (`augment.py`)

Это **синтетическое искажение** кадра (гомография, поворот, кроп-зум, зеркало, фотометрия,
blur, шум, JPEG, смена fps), а **не** съёмка другой камерой: нет параллакса, новых окклюзий и
сторон объектов. Устойчивость на пресетах — необходимое, но не достаточное условие для
критерия «другой угол»; финальная проверка — только на реальном видео.

* numpy: `apply(img, aug)`, `homography(aug, w, h)`, `transform_points`, `transform_bbox`
  (кадр и разметка преобразуются одной матрицей), `ClipAugmenter(seed).sample()` — одни
  параметры на весь клип, `random_aug(rng, strength)`, `resample_indices(times, 25)`.
* CLI: `python -m ml.ingest.augment in.mov out.mov --preset angle_left|angle_right|high|low|zoom|mirror|dark|rot_cw|rot_ccw|bright|blur|noise|jpeg|fps25|all [--engine auto|ffmpeg|python]`.
  `all` пишет `out_<preset>.mov`. ffmpeg-движок: геометрия через `perspective` (та же
  гомография); на тесте разница с numpy-версией < 2 уровней яркости. `dark/bright` в ffmpeg
  (`eq` по яркости в YUV) совпадают с numpy только приблизительно. Кодек вывода — `mpeg4`.

## Контакт-листы (`contact_sheet.py`)

`python -m ml.ingest.contact_sheet video.mov --t 22 --window 4 --out sheet.jpg`
(`--step 0.5`, `--cols 4`; без `--t` — обзор всего ролика `--every N`). Подписи — фактические
PTS, кадр ближе всего к `t` обведён красным.
**Приватность:** лист пишется только в локальный файл для человека или ЛОКАЛЬНОЙ VLM; CLI
печатает лишь путь и таймкоды (без картинок/base64/описаний). Не передавайте листы и кадры
реальных роликов облачным ассистентам (включая расширения IDE).

## Запуск и проверки

```bash
pip install -r ml/ingest/requirements.txt
python -m pytest ml/ingest/tests -q          # без GPU, без весов, без реальных данных
python -m pytest ml/ingest/tests -q -m "not slow"
```
Тесты генерируют синтетику сами (`cv2.VideoWriter`, ffmpeg lavfi) со штрих-кодом номера
кадра в каждом кадре — так проверяются PTS сэмплов, `frame_at` и seek вперёд/назад.
Покрыто: VFR, rotation (−90/90/180/hflip; PyAV и OpenCV против ffmpeg), seek, фактические PTS,
битые MOV (без moov, обрезанный faststart 30%/85%, испорченный payload, мусор, 0 байт,
только аудио), лимиты RAM/LRU, pickle, аугментации, контакт-листы, имена.
Тесты с ffmpeg помечены skip, если ffmpeg нет в PATH.

**Фактически выполнено** (Linux-контейнер, 4 vCPU, Python 3.11.17, av 18.1.0,
opencv-python-headless 5.0.0.93, ffmpeg 6.1.1): `88 passed` (~21 с).

**Замеры — только синтетика в этом контейнере** (testsrc2, 145 с, 30 fps, 464×848,
сэмплинг 5 fps → 725 кадров; на целевом ноутбуке и реальных iPhone-роликах не мерялось):

| файл | бэкенд | загрузка, с | RAM сэмплов |
|---|---|---|---|
| H.264, без поворота | PyAV | 5.1 (со `skip_nonref` 4.8) | 33 МБ |
| HEVC 848×464 + display matrix −90 | PyAV | 6.8 (со `skip_nonref` 5.3) | 33 МБ |
| H.264 / HEVC | OpenCV-фолбэк | 5.2 / 6.7 | 33 МБ |

`frame(i)` из LRU ≈ 0.4 мс (копия), промах (декод JPEG) ≈ 2–4 мс; 4 точных кадра
`frames_at` — 0.07 с (H.264) / 0.5 с (HEVC). Только декод (без конвертации) H.264 занимает
~2.7 с при ~10 с CPU — контейнер медленный; цель «< 5 с на CPU» на прод-ноутбуке нужно
подтвердить замером, обещать не могу.

## Зависимости и лицензии

| пакет | лицензия | зачем |
|---|---|---|
| PyAV (`av`) | BSD-3-Clause; колёса содержат FFmpeg (LGPL-2.1) | основной декодер |
| opencv-python-headless | Apache-2.0 (FFmpeg внутри — LGPL) | фолбэк, JPEG, метрики, warp |
| numpy | BSD-3-Clause | |
| PyYAML | MIT | config.yaml |
| ffmpeg/ffprobe (бинарь, опционально) | LGPL/GPL в зависимости от сборки | probe, CLI augment, тесты; отдельный процесс, не линкуется |

Моделей и весов нет; сеть не нужна.

## Ограничения

* Повороты не кратные 90° округляются до 90°; анаморфный SAR ≠ 1 не учитывается.
* HDR/10-бит (iPhone HLG) конвертируется в BGR без тон-маппинга (цвета бледнее).
* При `skip_nonref` (ручном или адаптивном) `source_times` неполный, сэмплы чуть неравномернее;
  `frame_at_ts` всё равно возвращает фактический PTS.
* OpenCV-фолбэк выбирает кадр правилом «первый с t ≥ узел − полкадра» (точно для CFR, для VFR —
  в пределах шага кадра); PyAV — честный ближайший.

## Для интегратора

* **Импорт контракта:** `video.py` делает `from contracts.sentinel_contracts import *` с фолбэком на
  `ml/ingest/_contract_fallback.py` (точная копия; удалить после мерджа A01).
* **Предложение A01 (не меняю контракт сам):** добавить в `VideoClip` необязательный метод
  `frame_at_ts(t) -> tuple[np.ndarray, float]` и атрибут `source_times: list[float]`.
  Обратно совместимо: `frame_at(t)` остаётся как есть.
* **A12 (fusion/доказательства):** `EvidenceFrame.timestamp_sec` брать из `clip.frame_at_ts(t)[1]`
  (или `clip.read_exact(t).timestamp_sec`), а не из запрошенного t. Если клип чужой реализации
  без метода — `getattr(clip, "frame_at_ts", None)`, иначе `clip.snap_time(t)` / t.
* **A03 (pipeline):** `load_video(path, sample_fps, progress=lambda f: progress("ingest", f))`;
  ловить `ml.ingest.IngestError` (`.code`, `.message`) → `status="failed"` с понятной ошибкой;
  `clip.meta["warnings"]`, `["complete"]`, `["coverage"]`, `["timings"]` → `AnalysisResult.meta`.
  `clip.motion_curve()` — дешёвый приор, `clip.gated_indices()` — для activity gating.
* **A02 (API):** коды ошибок → HTTP 422: `file_not_found`, `probe_failed`, `no_video_stream`,
  `duration_out_of_range`, `decode_failed`, `incomplete_video`.
* **A08:** сигнал `motion_energy` уже посчитан: `clip.quality[i]["motion_energy"]` (0..1).
* **A10/A05:** `ClipAugmenter` для обучения (одни параметры на клип), `transform_bbox` для разметки.
* **A16 (eval):** `names.normalize_video_id` (Х-3 → X-3); `names.find_duplicates` — по md5
  Х-4≡Х-47 (таймкоды 57 и 68 с) и Х-5≡Х-48: не считать дубликаты отдельными примерами в LOO.
  Пресеты `augment` — синтетический стресс-тест, не замена реальному видео под другим углом.
* **A17 (deploy):** PyAV-колёса для Windows/Python 3.11 есть на PyPI; ffmpeg/ffprobe не обязательны
  (`$SENTINEL_FFPROBE`, `$SENTINEL_FFMPEG`, `$SENTINEL_INGEST_CONFIG`). Временные файлы
  при переполнении RAM — в системный temp.
* **A01:** в корневой `pytest`-конфиг можно добавить маркеры `slow`, `gpu` (сейчас регистрирую
  в `ml/ingest/tests/conftest.py`). `ml/ingest/.gitignore` не пускает медиа из моей папки.
* Env: `SENTINEL_INGEST_CONFIG`, `SENTINEL_FFPROBE`/`FFPROBE_BIN`, `SENTINEL_FFMPEG`/`FFMPEG_BIN`.
