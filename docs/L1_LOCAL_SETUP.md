# L1 — локальные данные, окружение, веса (технический статус)

Машина: Windows 11, RTX 4060 Laptop 8 ГБ (драйвер 591.74, CUDA 13.1). Репо: `C:\Users\LEGION\Desktop\durachki_team_forte`.

**Политика.** Репозиторий публичный. В git идут только исходники и техническая документация. Видео, кадры,
контакт-листы, CSV/XLSX с метками, описания сцен, DATA_FINDINGS, веса, кэш и секреты лежат только локально
в `data/`, `cache/` и `weights/`. В этом документе есть только техника и агрегаты.
Скрипты пишут подробности на диск, а в stdout выводят только статус и метрики.

## 1. Данные (локально, не в git)

| путь | что |
|---|---|
| `data/raw/X-<n>.mov` | 42 исходных MOV; имена нормализованы (кириллическая `Х` → латинская `X`) |
| `data/videos` | NTFS junction на `data/raw`: путь по умолчанию в `eval.labels` и `backend.pipeline compare` |
| `data/open_labels.xlsx` | исходная разметка (binary + таймкод); заново **не** размечается |
| `data/labels.csv` | схема `eval.labels`: `video_id,file,label,t_sec,md5,group,t_all,conflict,raw` |
| `data/manifest.csv` | sha256, md5, ffprobe (w/h/rotation/codec/duration/nb_frames/avg_fps), `dup_of`, `needs_review` |
| `data/review_queue.csv` | md5-группы с расходящимися таймкодами (решает человек) |
| `data/labels_types.csv` | строка на каждое размеченное событие; `type_id` **только** у подтверждённых человеком (см. §1.2) |
| `cache/types_review/index.html` | локальная страница подтверждения типов (открыть в Chrome/Edge; **не публиковать**) |
| `cache/types_review/vlm_raw.jsonl`, `vlm_probe.jsonl`, `img/` | сырые ответы VLM и контакт-листы для человека |
| `cache/DATA_FINDINGS.md` | черновик находок: дописывает человек, в git не идёт |
| `cache/emb/siglip2_2fps/`, `cache/baseline/` | кэш эмбеддингов и OOF-предсказания baseline |
| `weights/L1_weights_manifest.json` | эталон sha256 всех весов (для `check_local.py --hash`) |

Агрегаты: 42 файла, 40 уникальных по sha256 (2 пары дубликатов, у одной расходятся таймкоды →
`conflict=1`, **не решено**). Классы: 22 violation / 20 no_violation.
Формат: H.264, **464×848 (портрет), rotation=0**, avg fps 29.76–30.0, **VFR**: время кадра брать из PTS, а не из
`i/fps`. Длительность 11.6–142.5 с (медиана 22.7, всего 24.1 мин).

### 1.1 Инструменты `tools/dataprep/`

Запускать из корня репо: `.venv/Scripts/python.exe -I -X utf8 tools/dataprep/<скрипт>`.

| скрипт | что делает |
|---|---|
| `build_labels.py data [--force]` | xlsx → `labels.csv` (существующий файл без `--force` не перезаписывает) |
| `build_manifest.py data` | sha256 + ffprobe → `manifest.csv`, `review_queue.csv` |
| `contact_sheets.py <video> <out.jpg> <t1,t2,…\|overview:N> [scale] [crop=…]` | контакт-лист для человека; держит в памяти только нужные кадры |
| `vlm_types.py --root .` | предложения типов от VLM (GPU), см. §1.2 |
| `vlm_probe.py --root .` | нейтральный опросник признаков (GPU, возобновляемый) |
| `types_review.py --root . [--confirmed …]` | HTML для человека, `labels_types.csv`, черновик DATA_FINDINGS (CPU) |
| `check_local.py --root . [--hash] [--eval-src …]` | проверка окружения, весов, данных, меток и совместимости с `eval` (CPU) |
| `siglip2_*_baseline.py`, `vlm_smoke.py` | baseline и smoke-тест VLM (GPU) |

Тесты (только синтетика, CPU): `.venv/Scripts/python.exe -m pytest tools/dataprep/tests -q`.

### 1.2 Типы: VLM предлагает → человек подтверждает

1. `vlm_types.py` — Qwen3-VL-4B (nf4) получает 4 обзорных кадра и окно t−5…t+5 с, при сомнении ещё окно t−10…t+10 с.
   Дополнительно разбирается пара md5-дубликатов с разными таймкодами и описывается процедура в 8 чистых видео.
2. `vlm_probe.py` — **нейтральный опросник**: 14 наблюдаемых признаков (yes/no/unclear + секунда); модели не говорят,
   что в окне нарушение. Окна t±5 с вокруг меток и окна чистых видео (1/3 и 2/3). Частоты «да» в нарушениях против
   чистых показывают, какие сигналы правил реально отличают нарушение. **Статус: 11/38 окон, пауза** (GPU отдан L2).
3. `types_review.py` собирает HTML (видео с перемоткой к t, контакт-листы, предложение VLM, чипы опросника, форма
   подтверждения), `data/labels_types.csv` и черновик `cache/DATA_FINDINGS.md`.
4. Человек подтверждает в HTML и выгружает `labels_types.csv`, `clean_procedures.csv`, `review_state.json`;
   затем импорт (§7).

Семантика `data/labels_types.csv` (колонки `video_id,t_sec,type_id,confidence,what_is_visible,actors,objects,
onset_or_peak,event_time_sec,new_type,draft_type_id,vlm_type_id,alt_type_id,source,confirmed,conflict,notes`):
- `type_id` заполнен **только** при `confirmed=1, source=human`. `eval.read_types_csv` пропускает строки с пустым
  `type_id`, поэтому предложения VLM в эталон не попадают.
- `vlm_type_id` — предложение VLM; `draft_type_id` — неподтверждённый выбор человека (в выгрузке из HTML).
- NEW → `V99` (по контракту) + описание в `new_type`. «Нарушения не видно» и «неясно» → `type_id` пуст, `notes=HUMAN:NONE/UNSURE`.
- md5-конфликт: `conflict=<id дубликата>`, `notes=UNRESOLVED…`, пока человек не примет решение в карточке пары
  (тогда `PAIR_DECISION:<решение>`). `labels.csv` скрипты не меняют.

Наблюдения о VLM-классификации (технические):
- `confidence` от VLM константна (0.95 во всех ответах), калибровки нет.
- «Секунда события» и onset/peak якорятся на подсказанной метке (|Δ| ≤ 1 с у всех) — для таймкода это не измерение.
- Принудительный выбор типа цепляется за признаки, которые есть и в норме (например, вход клиента через дверь).

## 2. Окружение

- `uv` 0.13.1 установлен через `pip --user` в Python 3.11: `%APPDATA%\Python\Python311\Scripts\uv.exe` (**не в PATH**).
- `.venv`: Python 3.11.6 (системный 3.14 не используется). Ключевые версии: torch 2.11.0+cu128,
  torchvision 0.26.0+cu128, transformers 5.20.1, accelerate 1.15.0, bitsandbytes 0.50.2, huggingface-hub 2.2.0,
  onnxruntime-gpu **1.23.2**, opencv-python-headless 4.14 (`<5`), numpy 2.4.6, pillow 13.0.0, scikit-learn 1.9.1.
- Полный lock (92 пина): `tools/dataprep/requirements-l1.lock.txt`; `check_local.py` сверяет установленное с lock.

## 3. Веса: HF hub-кэш `weights/hf/hub` (раскладка `deploy/weights.py`)

Загрузка без сети: `from_pretrained(repo_id, cache_dir="weights/hf/hub", local_files_only=True)` или
`HF_HUB_CACHE=<repo>/weights/hf/hub HF_HUB_OFFLINE=1`. Все модели проверены на загрузку офлайн.

| repo | commit | лицензия | размер | статус |
|---|---|---|---|---|
| google/siglip2-base-patch16-224 | 75de2d55 | Apache-2.0 | 1.43 ГиБ | OK, baseline (§4) |
| google/owlv2-base-patch16-ensemble | cfd3195b | Apache-2.0 | 0.58 ГиБ | OK (загрузка) |
| PekingU/rtdetr_v2_r18vd | 56509617 | Apache-2.0 | 0.08 ГиБ | OK (загрузка) |
| ustc-community/dfine-medium-obj2coco | e8284966 | Apache-2.0 | 0.07 ГиБ | OK (загрузка) |
| intfloat/multilingual-e5-small | 614241f6 | MIT | 0.46 ГиБ | OK (загрузка) |
| Qwen/Qwen3-VL-4B-Instruct | ebb281ec | Apache-2.0 | 8.28 ГиБ | OK, **выбранная VLM** (§5) |

Вне hub-кэша (пути, которые ищут модули):

| что | путь | лицензия | статус |
|---|---|---|---|
| Qwen/Qwen2.5-7B-Instruct-GGUF Q4_K_M (2 шарда, 4.36 ГиБ, commit bb5d59e0) | `weights/copilot/qwen2.5-7b-instruct-q4_k_m-0000{1,2}-of-00002.gguf` | Apache-2.0 | copilot ждёт один файл, поэтому `COPILOT_MODEL_PATH=` на шард `-00001-of-00002` (llama.cpp подхватит второй) |
| RTMPose-m body7 256×192 ONNX (openmmlab) | `weights/pose/20230831/rtmpose_onnx/rtmpose-m_simcc-body7_…/end2end.onnx` | Apache-2.0 | OK: `CUDAExecutionProvider`, 11.7 мс на батч из 8 кропов; `ml.pose` находит при cwd = корень репо |

**onnxruntime-gpu закреплён на 1.23.2**: версии ≥ 1.24 на PyPI собраны под CUDA 13 и не грузят CUDA-провайдер
рядом с torch cu128. rtmlib не ставился: у `ml/pose` свой ONNX-бэкенд, модель та же.
Не скачано сознательно: Qwen2.5-VL-7B-AWQ (6.9 ГБ весов, на 8 ГБ с рабочим столом не помещается).
`weights/Qwen__Qwen3-VL-4B-Instruct-GGUF/` — **недокачанный** GGUF, не использовать.

## 4. Baseline (binary по open_labels): SigLIP2 linear probe — **не годится**

`siglip2_binary_baseline.py` и `siglip2_window_baseline.py`: кадры 2 fps (2911 кадров), эмбеддинг 135 кадров/с
на GPU (fp16), пик VRAM 0.84 ГиБ. Валидация StratifiedGroupKFold(5) по md5-группам × 5 сидов.

| вариант | video ROC-AUC | таймкод \|argmax−t\| (позитивы) |
|---|---|---|
| весь ролик: mean+max+std → LogReg | 0.34 ± 0.04 | — |
| кадры ±2 с вокруг таймкода vs кадры чистых, max | 0.30 ± 0.03 | медиана 3.8 с; ≤1 с 18%, ≤2 с 45% |

Глобальный эмбеддинг кадра не отделяет нарушение от нормы; нужны объектные и позовые сигналы и/или VLM-проверка окна.
**Ловушка:** одна длительность даёт AUC 0.67 (позитивы короче: медиана 20.6 против 29.1 с). Это артефакт датасета.

## 5. Локальная VLM: Qwen3-VL-4B-Instruct, nf4 (LM) + bf16 (vision)

`vlm_smoke.py` (синтетические кадры 464×848): после загрузки занято 2.71 ГиБ, загрузка 12 с.

| кадров | входных токенов | латентность (2 новых токена) | пик VRAM |
|---|---|---|---|
| 1 | 386 | 1.4 с | 2.88 ГиБ |
| 3 | 1118 | 1.1 с | 3.39 ГиБ |
| 6 | 2216 | 3.4 с | 4.72 ГиБ |

Конфиг: `BitsAndBytesConfig(load_in_4bit, nf4, compute bf16, double_quant, llm_int8_skip_modules=["visual","lm_head"])`.
Прогон `vlm_types.py` (31 запрос, окно 11 кадров ×0.6 + обзор 4 кадра ×0.45, 2300–2600 токенов):
медиана 26 с на запрос, пик VRAM 5.26 ГиБ, всего 24 мин.

**Грабли WDDM.** На 8 ГБ с рабочим столом (1–2.4 ГБ VRAM, плавает) драйвер при заполнении VRAM уходит в sysmem
fallback: вызов «висит» минутами при 100% загрузке и ~27 Вт. Что помогает:
1. до загрузки модели `torch.cuda.set_per_process_memory_fraction((free − 2 ГиБ) / total)`, тогда вместо спилла
   будет OOM (CUDA-контекст, буферы bnb и чужие приложения в лимит не входят);
2. `generate(max_time=90)`; `ask_safe` при OOM или упоре в `max_time` повторяет с уменьшенными кадрами и пишет `shrink`.
Для бюджета 5 мин/видео в рантайме VLM — только на 1–3 коротких окнах-кандидатах.

## 6. Грабли

- `.git/info/exclude` **общий для всех worktree** (в том числе `durachki_L2`). В нём только якорные `/data/ /weights/
  /cache/ /.venv/` и бинарные форматы данных и весов. Глобальные `*.jpg`/`*.png`/`*.npy` и неякорный `data/` туда не
  добавлять: они прячут законные файлы кода в соседнем worktree. В `integration-cloud` `.gitignore` тоже якорный.
- `python -I` игнорирует `PYTHONIOENCODING` и не кладёт каталог скрипта в `sys.path`: кириллица через `-X utf8`,
  соседние модули `tools/dataprep` грузятся через `importlib` по пути.
- Консоль Windows по умолчанию cp1251: subprocess, печатающий `≠`/кириллицу без `-X utf8`, падает с `UnicodeEncodeError`.
- Длинные пути и кириллица ломают нативные exe: видео лежат под латинскими именами в коротком пути.

## 7. Для L2: проверки и импорт подтверждённой разметки

```powershell
# окружение, веса (офлайн-резолв), данные, метки; CUDA не трогается
.venv\Scripts\python.exe -I -X utf8 tools\dataprep\check_local.py --root . --no-cuda
# + sha256 видео (vs data/manifest.csv) и весов (vs weights/L1_weights_manifest.json), ~15 с
.venv\Scripts\python.exe -I -X utf8 tools\dataprep\check_local.py --root . --hash --no-cuda
# + совместимость с eval.labels из ветки интеграции (корень worktree с пакетом eval/)
.venv\Scripts\python.exe -I -X utf8 tools\dataprep\check_local.py --root . --only labels --eval-src C:\Users\LEGION\Desktop\durachki_L2
# тесты инструментов (синтетика)
.venv\Scripts\python.exe -m pytest tools\dataprep\tests -q

# импорт после подтверждения человеком (файлы, выгруженные из cache\types_review\index.html):
.venv\Scripts\python.exe -I -X utf8 tools\dataprep\types_review.py --root . `
    --confirmed $HOME\Downloads\labels_types.csv --clean-notes $HOME\Downloads\clean_procedures.csv `
    --state $HOME\Downloads\review_state.json
.venv\Scripts\python.exe -I -X utf8 tools\dataprep\check_local.py --root . --only labels
```

Импорт берёт только строки `confirmed=1`; повторный запуск идемпотентен. Данные и веса лежат только в основной папке
(`C:\Users\LEGION\Desktop\durachki_team_forte\data`, `…\weights`). Из другого worktree на них указывают через
`SENTINEL_WEIGHTS` / `HF_HUB_CACHE` / `--root` или junction.

## 8. Совместимость с `claude/integration-cloud` @ 2055fc4 (проверено на CPU)

- `eval.labels` читает `data/labels.csv` и `data/labels_types.csv`: 42 строки, 40 групп, conflict-группа одна
  (дубликат с двумя таймкодами), типов в эталоне 0 (подтверждений пока нет).
- `tools/dataprep/tests` проходят и под корневым `conftest`/`pyproject` интеграции (30/30).
- `deploy` (тесты без gpu/slow) на Windows: 56 passed, 5 failed, 1 skipped. Падения **одинаковые со старым и новым**
  lock L1:
  - 3 — cp1251 в subprocess и ассертах (`test_offline` ×2, `test_supervisor::test_foreign_process_on_port_is_not_touched`);
  - 1 — WinError 193 (`test_lock_calls_real_resolver_with_platform` запускает как процесс не-exe);
  - 1 — `test_repo_merge_is_clean` (Pillow, см. ниже).
- `ml/fusion` с Pillow 13.0.0: 129/130. Единственное падение — предупреждение о каталоге норм, не Pillow.
  Пин `Pillow<13` в `ml/fusion/requirements.txt`, судя по тестам, не нужен.
