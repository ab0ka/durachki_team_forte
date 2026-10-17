# L1 — локальные данные, окружение, веса (технический статус)

Машина: Windows 11, RTX 4060 Laptop 8 ГБ (драйвер 591.74, CUDA 13.1). Репо: `C:\Users\LEGION\Desktop\durachki_team_forte`.

**Политика.** В git нет и не будет видео, кадров, контакт-листов, меток, описаний сцен и весов. Всё это лежит
только локально в `data/`, `cache/` и `weights/`. В этом документе есть только техника и агрегаты.
Подробные результаты скриптов пишутся на диск, а в stdout выводятся только статус и метрики.

## 1. Данные (локально, не в git)

| путь | что |
|---|---|
| `data/raw/X-<n>.mov` | 42 исходных MOV; имена нормализованы (кириллическая `Х` → латинская `X`) |
| `data/videos` | NTFS junction на `data/raw`: путь по умолчанию в `eval.labels` (`--videos data/videos`) |
| `data/open_labels.xlsx` | исходная разметка (binary + таймкод); заново **не** размечается |
| `data/labels.csv` | схема `eval.labels`: `video_id,file,label,t_sec,md5,group,t_all,conflict,raw` |
| `data/manifest.csv` | sha256, md5, ffprobe (w/h/rotation/codec/duration/nb_frames/avg_fps), `dup_of`, `needs_review` |
| `data/review_queue.csv` | md5-группы с расходящимися таймкодами (проверяет человек) |
| `data/labels_types.csv` | типы по событиям: сейчас предложения VLM (`source=vlm, confirmed=0`), после проверки — человек |
| `cache/types_review/index.html` | локальная страница подтверждения типов (открыть в Chrome/Edge; **не публиковать**) |
| `cache/DATA_FINDINGS.md` | черновик находок: дописывает человек, в git не идёт |
| `data/labels_types_TEMPLATE.csv` | пустой шаблон типов (устарел, заменён HTML) |
| `cache/sheets/` | контакт-листы 4×3 (окна t−5…t+6 с и обзорные) **для просмотра человеком** |
| `cache/emb/siglip2_2fps/`, `cache/baseline/` | кэш эмбеддингов и OOF-предсказания baseline |

Агрегаты: 42 файла, 40 уникальных по sha256 (2 пары дубликатов, у одной расходятся таймкоды →
`conflict=1`). Классы: 22 violation / 20 no_violation.
Формат: H.264, **464×848 (портрет), rotation=0**, avg fps 29.76–30.0, **VFR**: время кадра брать из PTS, а не
из `i/fps`. Длительность 11.6–142.5 с (медиана 22.7, всего 24.1 мин).

Скрипты в `tools/dataprep/` (запускать из корня репо: `.venv/Scripts/python.exe -I -X utf8 ...`):
`build_labels.py data`, `build_manifest.py data`, `contact_sheets.py <video> <out.jpg> <times|overview:N>`.

### Разметка типов: локальная VLM → подтверждение человеком

1. `vlm_types.py --root .` — Qwen3-VL-4B (nf4) получает 4 обзорных кадра и окно t−5…t+5 с, при сомнении
   (conf < 0.6, NEW или нераспознанный ответ) ещё окно t−10…t+10 с. Дополнительно разбираются пары md5-дубликатов
   с разными таймкодами и описывается процедура в 8 чистых видео. Сырые ответы пишутся в
   `cache/types_review/vlm_raw.jsonl`, контакт-листы — в `cache/types_review/img/`. Повторный запуск продолжает с места остановки.
1a. `vlm_probe.py --root .` — **нейтральный опросник**: 14 наблюдаемых признаков (yes/no/unclear + секунда),
   модели не сообщают, что в окне нарушение. Окна t±5 с вокруг меток нарушений и окна чистых видео (1/3 и 2/3).
   Разница частот «да» (нарушения против чистых) показывает, какие сигналы правил реально отличают нарушение.
   Результат пишется в `cache/types_review/vlm_probe.jsonl`.
2. `types_review.py --root .` собирает `cache/types_review/index.html` (видео с перемоткой к t, контакт-листы,
   предложение VLM, форма подтверждения), `data/labels_types.csv` (`source=vlm, confirmed=0`) и черновик
   `cache/DATA_FINDINGS.md`.
3. Человек открывает HTML в Chrome/Edge и подтверждает типы, потом выгружает `labels_types.csv`, `clean_procedures.csv` и
   `review_state.json`. Затем `types_review.py --root . --confirmed <csv> --clean-notes <csv> --state <json>`, после чего
   в `data/labels_types.csv` попадают `source=human, confirmed=1`.

Колонки `labels_types.csv`: `video_id,t_sec,type_id,confidence,what_is_visible,actors,objects,onset_or_peak,
event_time_sec,new_type,alt_type_id,source,confirmed,notes` (надмножество для `eval.read_types_csv`).
**Пока `confirmed=0`, типы — только гипотезы VLM, обучать на них типовую голову нельзя.**

Наблюдения о самой VLM-классификации (технические, без данных):
- `confidence` от VLM почти константна (0.95 во всех ответах). Калибровки нет, поэтому порог «сомнения» не срабатывает.
- «Секунда события» и onset/peak якорятся на подсказанной в промпте метке: у всех событий |Δ| ≤ 1 с.
  Для таймкода это не измерение.
- Принудительный выбор типа склонен цепляться за признаки, которые есть и в норме (например, вход клиента через дверь).
  Поэтому рядом показан нейтральный опросник, а решение остаётся за человеком.

## 2. Окружение

- `uv` 0.13.1 установлен через `pip --user` в Python 3.11: `%APPDATA%\Python\Python311\Scripts\uv.exe` (**не в PATH**).
- `.venv`: Python 3.11 (системный 3.14 не используется). Ключевые версии: torch 2.11.0+cu128, torchvision 0.26.0+cu128,
  transformers 5.20.1, accelerate 1.15.0, bitsandbytes 0.50.2, huggingface-hub 2.2.0, opencv-python-headless 4.14
  (пин `<5`, как в `pyproject` A01), numpy 2.4.6, scikit-learn 1.9.1, pytest, ruff, pyyaml, jsonschema.
- `torch.cuda.is_available() == True`. Полный lock: `tools/dataprep/requirements-l1.lock.txt`.

## 3. Веса: HF hub-кэш `weights/hf/hub` (раскладка `deploy/weights.py`)

Загрузка без сети: `from_pretrained(repo_id, cache_dir="weights/hf/hub", local_files_only=True)` или
`HF_HUB_CACHE=weights/hf/hub HF_HUB_OFFLINE=1`. Все модели ниже проверены на загрузку офлайн.

| repo | commit | лицензия | размер | статус |
|---|---|---|---|---|
| google/siglip2-base-patch16-224 | 75de2d55 | Apache-2.0 | 1.5 ГБ | OK, baseline (§4) |
| google/owlv2-base-patch16-ensemble | cfd3195b | Apache-2.0 | 0.6 ГБ | OK (загрузка) |
| PekingU/rtdetr_v2_r18vd | 56509617 | Apache-2.0 | 78 МБ | OK (загрузка) |
| ustc-community/dfine-medium-obj2coco | e8284966 | Apache-2.0 | 76 МБ | OK (загрузка) |
| intfloat/multilingual-e5-small | 614241f6 | MIT | 0.47 ГБ | OK (загрузка) |
| Qwen/Qwen3-VL-4B-Instruct | ebb281ec | Apache-2.0 | 8.3 ГБ | OK, **выбранная VLM** (§5) |

Вне hub-кэша (пути, которые ищут модули):

| что | путь | лицензия | статус |
|---|---|---|---|
| Qwen/Qwen2.5-7B-Instruct-GGUF Q4_K_M (2 шарда, 4.4 ГиБ, commit bb5d59e0) | `weights/copilot/qwen2.5-7b-instruct-q4_k_m-0000{1,2}-of-00002.gguf` | Apache-2.0 | скачано; copilot ждёт один файл `Qwen2.5-7B-Instruct-Q4_K_M.gguf`, поэтому `COPILOT_MODEL_PATH=` на шард `-00001-of-00002` (llama.cpp сам подхватит второй) |
| RTMPose-m body7 256×192 ONNX (openmmlab) | `weights/pose/20230831/rtmpose_onnx/rtmpose-m_simcc-body7_…/end2end.onnx` | Apache-2.0 | OK: `CUDAExecutionProvider`, 11.7 мс на батч из 8 кропов; находится через `ml.pose…find_weights` (cwd = корень репо) |

**onnxruntime-gpu закреплён на 1.23.2.** Версии ≥ 1.24 на PyPI собраны под CUDA 13 и не грузят CUDA-провайдер
рядом с torch cu128. Пакет rtmlib не ставился: у `ml/pose` свой ONNX-бэкенд, а модель та же.

Сознательно **не** скачано:
- Qwen2.5-VL-7B-AWQ: 6.9 ГБ весов, на 8 ГБ при работающем рабочем столе не помещается.

`weights/Qwen__Qwen3-VL-4B-Instruct-GGUF/` — **недокачанный** GGUF, загрузка прервана; не использовать.

## 4. Baseline (binary по open_labels): SigLIP2 linear probe — **не годится**

`tools/dataprep/siglip2_binary_baseline.py` и `siglip2_window_baseline.py`: кадры 2 fps, 2911 кадров.
Скорость эмбеддинга 135 кадров/с на GPU (fp16), пик VRAM 0.84 ГиБ; декодирование 52 с на CPU.
Валидация StratifiedGroupKFold(5) по md5-группам × 5 сидов.

| вариант | video ROC-AUC | таймкод \|argmax−t\| (позитивы) |
|---|---|---|
| весь ролик: mean+max+std → LogReg | 0.34 ± 0.04 | — |
| кадры ±2 с вокруг таймкода vs кадры чистых, max | 0.30 ± 0.03 | медиана 3.8 с; ≤1 с 18%, ≤2 с 45% |

Оба варианта стабильно хуже случайного: глобальный эмбеддинг кадра не отделяет нарушение от нормы.
Нужны объектные и позовые сигналы (детектор + трекинг + зоны + правила) и/или VLM-верификация окна.
**Ловушка:** длительность сама по себе даёт AUC 0.67 (позитивы короче: медиана 20.6 против 29.1 с). Это артефакт
датасета, использовать длительность как признак нельзя.

## 5. Локальная VLM: Qwen3-VL-4B-Instruct, nf4 (LM) + bf16 (vision)

`tools/dataprep/vlm_smoke.py` гоняет модель на синтетических кадрах 464×848 (реальные сцены не используются).
До загрузки свободно 6.93 / 8.00 ГиБ, загрузка занимает 12 с.

| кадров | входных токенов | латентность (2 новых токена) | пик VRAM |
|---|---|---|---|
| 1 | 386 | 1.4 с | 2.88 ГиБ |
| 3 | 1118 | 1.1 с | 3.39 ГиБ |
| 6 | 2216 | 3.4 с | 4.72 ГиБ |

После загрузки занято 2.71 ГиБ. Конфиг: `BitsAndBytesConfig(load_in_4bit, nf4, compute bf16, double_quant,
llm_int8_skip_modules=["visual","lm_head"])`. ≤6 кадров на запрос помещаются вместе с детектором.

Реальный прогон `vlm_types.py` на 31 запрос: окно 11 кадров ×0.6 и обзор 4 кадра ×0.45, 2300–2600 входных
токенов. Медиана 26 с на запрос (генерация JSON ~300 токенов), пик VRAM 5.26 ГиБ, весь прогон занял 24 мин.

**Грабли WDDM.** Без лимита (11 кадров ×0.75 + 6 обзорных, около 3000 токенов, пик 6.5 ГиБ) VRAM забилась до 7.8/8 ГБ.
Драйвер ушёл в sysmem fallback, и за 4 минуты не было ни одного ответа. Решение:
`torch.cuda.set_per_process_memory_fraction((free − 2 ГиБ) / total)` (CUDA-контекст, буферы bnb и рост памяти других приложений — вне лимита) и `generate(max_time=90)`: зависший из-за спилла вызов обрывается и повторяется с уменьшенными кадрами до загрузки модели. Тогда вместо спилла
будет OOM, и `ask_safe` повторит запрос с уменьшенными кадрами. Отдельные выбросы до 435 с всё равно случаются.
Для бюджета 5 мин/видео в рантайме VLM — только на 1–3 коротких окнах-кандидатах.

## 6. Грабли

- В `main` нет `.gitignore`: он лежит в ветке A01 (`claude/gifted-bardeen-i404eh`). Пока его не влили, локально
  стоит защита в `.git/info/exclude` (data/weights/cache/видео/кадры/веса).
- Шаблон `data/` в `.gitignore` без якоря и игнорирует **любой** каталог `data` (например, `tools/data/`).
  Каталоги с кодом так не называть.
- `python -I` игнорирует `PYTHONIOENCODING`, кириллицу в выводе включать через `-X utf8`.
  Консоль Git Bash/cmd по-прежнему показывает кракозябры, поэтому файлы писать с `encoding="utf-8"`.
- Длинные пути и кириллица ломают нативные exe: видео лежат под латинскими именами в коротком пути.
- Типы нарушений пока **не размечены**. Предложения делает локальная VLM, подтверждает человек;
  до подтверждения типовую голову обученной не считать.
