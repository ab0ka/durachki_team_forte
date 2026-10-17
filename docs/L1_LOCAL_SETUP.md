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
| `data/labels_types_TEMPLATE.csv` | пустой шаблон типов для человека; колонки совместимы с `eval.read_types_csv` |
| `cache/sheets/` | контакт-листы 4×3 (окна t−5…t+6 с и обзорные) **для просмотра человеком** |
| `cache/emb/siglip2_2fps/`, `cache/baseline/` | кэш эмбеддингов и OOF-предсказания baseline |

Агрегаты: 42 файла, 40 уникальных по sha256 (2 пары дубликатов, у одной расходятся таймкоды →
`conflict=1`). Классы: 22 violation / 20 no_violation.
Формат: H.264, **464×848 (портрет), rotation=0**, avg fps 29.76–30.0, **VFR**: время кадра брать из PTS, а не
из `i/fps`. Длительность 11.6–142.5 с (медиана 22.7, всего 24.1 мин).

Скрипты в `tools/dataprep/` (запускать из корня репо: `.venv/Scripts/python.exe -I -X utf8 ...`):
`build_labels.py data`, `build_manifest.py data`, `contact_sheets.py <video> <out.jpg> <times|overview:N>`.

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

Сознательно **не** скачано:
- Qwen2.5-VL-7B-AWQ: 6.9 ГБ весов, на 8 ГБ при работающем рабочем столе не помещается.
- Qwen2.5-7B-Instruct GGUF: текстовый LLM; при необходимости его подтянет `deploy/weights.py prepare` по `copilot/models.json`.
- RTMPose (rtmlib).

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
