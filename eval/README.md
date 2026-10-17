# eval/ — оценка «как у жюри» (A16)

Инструменты оценки Kassa Sentinel: разметка, групповой split, протокол прогона, метрики, scorecard, санити-полы,
абляции, журнал итераций, black-box приёмка API, наши съёмки с замороженной камерой, устойчивость, отчёты.
**Измерений на реальных данных здесь нет** (видео и GPU только у локального L2): код проверен синтетическими
тестами и явно помеченным mock-пайплайном. Mock-тесты проверяют код, а не качество CV. Цифр без замера не обещаем.

**Все 42 видео датасета уже использованы в разведке → любая оценка на них — `exploratory`, не независимый тест.**
Наш 100-балльный балл — явно помеченная аппроксимация (proxy), не официальная формула жюри.

## Одна команда для L2 (абляция «одно изменение»)

```bash
python -m eval.ablate --name H07-<гипотеза> --candidate-env SENTINEL_<FLAG>=1 \
    --videos data/raw --labels data/labels.csv --manifest data/manifest.csv --out runs/ablate/H07
#   варианты: --candidate-config cand.yaml (конфиг пайплайна целиком), --reuse-baseline runs/ablate/H06/baseline,
#             --api https://<tunnel> --key $SENTINEL_API_KEY (замер через сервис), --subset train,calib
```

**Что сравниваем:** baseline и кандидат одной процедурой (`eval.run`, каждая рука — отдельный процесс, последовательно,
один GPU) на одинаковых входах (`inputs_sha` и `labels_sha256` совпадают, иначе вердикт INVALID). Отличаться должен
ровно один флаг/конфиг (иначе предупреждение).

**Выход** (`runs/ablate/H07/`):
* `baseline/`, `candidate/` — `preds/` (кадры, описания — **только локально**), `metrics.json`, `protocol.json`,
  `proxy_score.json`, `summary_public.json`;
* `compare.json` (локально, с ID видео) и `compare_public.json` (только агрегаты — можно передать в облако);
* строки в `reports/runs.jsonl` (агрегаты) и график `reports/runs.html`.

**Метрики:** FN/FP, recall/precision/specificity с 95% CI, failed/partial/timeouts, hit@1/2/3 с (one-to-one),
кадр в ±1 с, тип (только при настоящих GT типов), latency p50/p90/max, пик VRAM, санити-полы обеих рук,
парное сравнение по видео (исправлено / сломано, точный тест Макнемара).

**Вердикт** (`config.yaml → ablate`):
* **FAIL** — FN вырос; FP вырос больше допуска `max(1, 5%·N)` (рост recall при росте FP — **не PASS**);
  выросли failed/partial/timeouts; p90 > SLA 60 с; VRAM > 7500 МБ; латентность +20% без выигрыша в качестве;
* **PASS** — нет причин FAIL и есть измеримый выигрыш: FN↓, FP↓ при FN не хуже, hit@2 ≥ +0.05, p90 ≥ 20% быстрее;
* **INCONCLUSIVE** — эффекта нет; **INVALID** — разные входы или разная процедура.

PASS на 42 разведочных видео — тоже разведочный. Новый режим остаётся выключенным в config пайплайна, пока L2 не
замерил выигрыш. Включение по умолчанию — решение команды после подтверждения на замороженном hold-out (`eval.extra`).

## Быстрый старт (локально, на машине с данными)

```bash
# L1 уже строит data/labels.csv (tools/dataprep/build_labels.py) в схеме eval.labels; или:
python -m eval.labels --xlsx data/open_labels.xlsx --videos data/raw --out data/labels.csv
python -m eval.split  --labels data/labels.csv --out data/splits.csv          # один раз, потом не менять
python -m eval.run    --videos data/raw --label baseline --out runs/eval/base1 --raw
python -m eval.sanity --labels data/labels.csv --manifest data/manifest.csv --run runs/eval/base1 \
                      --scores cache/baseline/siglip2_window_oof.csv:score_oof
python -m eval.jury_sim --url https://<tunnel> --key $SENTINEL_API_KEY --videos data/raw --n 6 --concurrency 3
python -m eval.report --run runs/eval/base1           # runs/eval/base1/report.html — только локально
python -m eval.runs html                              # reports/runs.html — журнал итераций
pytest eval/tests -q                                  # без GPU и весов
```

Все выходы — в `runs/` (gitignored) и `reports/` (только агрегаты). `preds/*.json` и HTML-отчёты с кадрами
**остаются локально**. Наружу (облачным агентам/LLM, в чат, в PR) — только `summary_public.json`,
`compare_public.json`, `reports/runs.jsonl`: агрегаты без кадров, описаний и ID видео.

## Модули и публичные функции

| Файл | Что делает | Главное API |
|---|---|---|
| `labels.py` | xlsx → `data/labels.csv` (`video_id,file,label,t_sec,md5,group,t_all,conflict,raw,sha256`) | `parse_timecodes`, `build_labels`, `read_labels_csv`, `read_types_csv`, `load_ground_truth` |
| `split.py` | групповой train/calib/holdout + фолды → `data/splits.csv` | `make_splits`, `write_splits`, `load_splits`, `ids_for`, `log_holdout_access` |
| `protocol.py` | протокол прогона: SHA, хэши конфигов и входов, версии пакетов, GPU, флаги; сэмплер VRAM | `build_protocol`, `same_procedure`, `VramSampler`, `inputs_fingerprint` |
| `run.py` | прогон папки (inproc / cli / api) → `preds/`, `protocol.json`, метрики, журнал | `run_folder`, `select_videos`, `apply_env`, `write_scores_raw` |
| `metrics.py` | метрики, scorecard, публичная сводка | `evaluate`, `evaluate_dirs`, `scorecard`, `public_summary`, `match_one_to_one`, `wilson` |
| `sanity.py` | санити-полы и детекторы утечек длительности, позиции и имени файла | `sanity`, `auc`, `spearman`, `stratified_auc`, `load_durations`, `run_scores` |
| `ablate.py` | абляция baseline vs кандидат → вердикт | `compare`, `mcnemar_exact`, `public` |
| `runs.py` | журнал итераций `reports/runs.jsonl` + HTML-график | `record_from_run`, `append`, `load`, `render_html` |
| `jury_sim.py` | black-box приёмка по публичному URL | `run_jury_sim`, `JurySim`, `inspect_result`, `jpeg_size` |
| `extra.py` | наши съёмки: схема, заморозка C3, split по `scene_id`, отчёты по стратам | `read_extra`, `freeze`, `make_extra_splits`, `stratified_report` |
| `schema.py` | проверка ответа по схеме ТЗ (+ `contracts/api_schema.json`) | `check_api_response`, `decode_jpeg_b64` |
| `score_like_jury.py` | **наш proxy-балл** 0..100 (не формула жюри) | `score`, `format_score` |
| `robustness.py` | синтетические возмущения ракурса и качества | `make_variant`, `VARIANTS`, `compare` |
| `report.py` | локальный HTML-отчёт (содержит кадры! не выносить) | `build_report` |
| `loo.py` | LOGO (**exploratory**) и nested CV порога | `logo`, `nested_cv`, `fit_threshold` |

Все пороги — в `eval/config.yaml` (`--config my.yaml` переопределяет).

## Решения фазы 2 (записано по требованию «решения — в README»)

1. **Семантика «нет ответа ≠ чисто».** failed, missing, невалидный вердикт и `no_violation` при `meta.coverage < 0.95`
   (partial) никогда не считаются TN. На позитиве это пропуск (FN), на негативе — `unresolved_neg`, что снижает
   specificity. `coverage.require: false`: пока A03 не отдаёт coverage, отсутствие coverage не считается partial,
   но считается в scorecard (`coverage_unknown`). Когда coverage появится, включить `require: true`.
2. **unknown ≠ false в разметке.** Пустая или непонятная метка в labels.csv даёт `unknown`: группа исключается
   из метрик и выводится в `label_unknown_ids`, а не превращается в «без нарушения».
3. **Конфликт 57/68 — отдельно** (`conflict_policy: separate`). Группа дубликатов с разными таймкодами считается
   в бинарных метриках (метка у обоих — нарушение), но не входит в основные метрики локализации и кадра.
   Её Δt до каждой альтернативы — в блоке `conflict_groups`. Решает человек. Альтернативный расчёт
   (одно событие с альтернативами) — в `localization_alt`. Исходные строки сохраняются в `raw`.
4. **Вид оценки.** `dataset.evaluation_kind: exploratory` по умолчанию. `--kind holdout_frozen` разрешён только для
   `--subset holdout --final` по split'у, созданному `eval.extra freeze`; иначе метка автоматически понижается до exploratory.
5. **Политика таймкода — замером.** `summary.timecode_policy` считает hit@k и медиану |Δt| для `timecode_sec`,
   `t_start` и середины интервала на тех же предсказаниях (GT — момент начала нарушения). Выбирать по цифрам L2.
6. **Длительность, позиция, имя файла — не признаки.** `eval.sanity` показывает, обгоняет ли скор baseline по длительности.
   Корреляция с длительностью (|ρ| ≥ 0.3) или номером файла — отдельная строка с ⚠. AUC внутри терцилей длительности
   показывает сигнал сверх длины ролика. Пулинги кривой риска (max / mean / top3 / p90 / prominence) — для выбора без артефакта max.
7. **C3 замораживается до настройки.** Сцены, общие с другими камерами, по умолчанию остаются в tune: это пересечение
   содержания, а не ракурса. Их список — в `scene_overlap_not_frozen`. Строгий режим — `--strict-scenes`.
8. **Журнал `reports/runs.jsonl`** — только агрегаты, без ID видео. `reports/` в .gitignore A01 не входит: коммитить
   журнал можно (там нет кадров и описаний) или добавить `reports/` в .gitignore — решение A01/интегратора.

## Протокол (`protocol.json`) — baseline и кандидат одной процедурой

`git_sha` (+dirty), `eval_version`, хэш и содержимое eval-конфига, путь и хэш конфига пайплайна (`--pipeline-config`
→ env `SENTINEL_PIPELINE_CONFIG`), флаги `SENTINEL_*` (`--env K=V`; ключи маскируются), версии пакетов через
`importlib.metadata` (torch, transformers, bitsandbytes, opencv… — без импорта), GPU (nvidia-smi), список входов и
`inputs_sha` (sha256 из labels.csv), `labels_sha256`, вид оценки, режим, subset. VRAM: пик `memory.used` всей карты
по nvidia-smi на каждое видео (`VramSampler`) и `torch.cuda.max_memory_allocated`, если torch уже импортирован пайплайном.

## Санити-полы (`sanity.py`)

* «всё violation» (recall 1, FP = N), «всё clean» (FN = P) — по всем размеченным видео;
* baseline по длительности (короче → нарушение; по данным L1 AUC ≈ 0.67): AUC, порог in-sample (оптимистично) и LOGO-порог;
* пол таймкода: t = медианная относительная позиция GT × длительность → hit@k;
* наши скоры (scores_raw, confidence, пулинги timeline, OOF-CSV L1): AUC, ρ с длительностью и номером файла,
  AUC в терцилях длительности. Предупреждения: |ρ| ≥ 0.3, AUC ≤ baseline по длительности, AUC < 0.5.

## Метрики (`metrics.py`)

* **Scorecard** (главное): вид оценки, n (+/−), recall / precision / specificity с 95% CI Вильсона, FN, FP, failed,
  partial, unresolved_neg, timeouts, degraded, coverage_unknown, hit событий @0.5/1/2/3, кадр ±1 с, лишние кандидаты,
  тип (только при GT), latency median/p90/max, превышения SLA, пик VRAM, версии и хэши.
* **Локализация — one-to-one** в окне `match_window_s` (5 с); hit@k по событиям. «Рано» / «поздно» — по знаку Δt.
  Лишние кандидаты считаются отдельно и в FP по видео не входят: разметка событий неполная (одна секунда на видео).
* **Типы** — только по настоящей разметке (`labels_types.csv`, для наших съёмок — колонка `type_id`), иначе n/a.
* **Нормы** — полнота и формат ссылок (rule есть, из каталога, verified). **Наличие quote/URL ≠ правильная трактовка
  нормы**; корректность — только по ручной экспертизе `data/norms_review.csv` (`type_id,url,clause,correct`).
* Схема ТЗ, язык описаний (hedged, без домыслов), согласованность вердиктов на дубликатах.

## Black-box приёмка (`jury_sim.py`)

Порядок: auth 401 (POST и GET, без ключа и с неверным), 404 на неизвестный job (мягко), `.txt` → 415; затем
**последовательно ≥ 5 видео** (меньше — пункт NOT RUN); затем **ограниченная пачка** (`--concurrency`, по умолчанию 3);
в конце большой файл. По каждой фазе — отдельные пункты:
загрузка < 100 с (таймаут Cloudflare; ≥ 60% — WARN; HTTP 52x — FAIL), все файлы приняты, MOV, все задачи завершены
(незавершённые — FAIL), ≤ 300 с от приёма **включая очередь**, схема ТЗ, таймкоды и кадры в пределах длительности
по PTS (ffprobe локального файла), кадры — JPEG в ориентации видео (горизонтальное «другой угол» тоже проверяется),
настоящие модели (`meta.mock` / `meta.models` без mock; не сообщены — WARN), failed — с причиной,
partial не выдан за no_violation.
Итог: **FAIL** при любом FAIL; **INCOMPLETE**, если критичный пункт NOT RUN; иначе **PASS**.
В отчёт пишутся только статусы и тайминги.

## Наши съёмки (`extra.py`)

`data/extra/labels.csv`: `video_id,file,label,t_all,type_id,camera_id,scene_id,layout,lighting[,orientation]`.
`check` проверяет схему и пишет `labels_eval.csv` и `labels_types.csv`. `freeze --camera C3` замораживает
hold-out (необратимо без `--force`) и пишет `splits.csv`: C3 → holdout, остальные → train/calib группами по `scene_id`.
`report --run` строит отчёт по стратам: камера, планировка, свет, ориентация. Подбор порогов — только на `--subset train,calib`.
Финал — один раз: `--subset holdout --final --kind holdout_frozen`, каждое обращение журналируется.

## Proxy-балл (`score_like_jury.py`) — наша аппроксимация, НЕ формула жюри

| Компонент | Формула (доля 0..1 × вес) |
|---|---|
| detection 25 | `1 − (2·FN + 1·(FP + негативы без ответа)) / (2·P + N)` |
| timecode 15 | среднее по позитивам: `0.667·ступень(|Δt|: ≤1→1, ≤2→.75, ≤3→.5, ≤5→.25) + 0.333·[кадр в ±1 с]`, FN → 0 |
| type 15 | точность по настоящим GT типов; без них n/a |
| norms 15 | proxy полноты ссылок `0.4·[≥1 rule] + 0.3·[из каталога] + 0.3·verified` (не корректность!) |
| robustness 10 | proxy на синтетике `0.7·(1 − смена вердикта) + 0.3·стабильный таймкод`; без данных n/a |
| product 10 | proxy `0.4·схема + 0.2·JPEG + 0.2·1–3 кадра + 0.2·hedged·(1−forbidden)` |
| architecture 10 | proxy `0.5·доля в SLA + 0.3·(1 − failed) + 0.2·jury_sim pass_rate` |

n/a-компоненты исключаются, остальное перенормируется на 100.

## Устойчивость (`robustness.py`), LOO (`loo.py`)

Перспектива, поворот, кроп, зеркало, яркость, blur, h264, 25 fps — **синтетические возмущения, не доказательство
работы на реальном новом ракурсе**. Варианты наследуют split исходника. LOGO по данным, на ошибках которых меняли
пайплайн, — **exploratory**; nested CV оценивает процедуру подбора порога.

## Зависимости и модели

`eval/requirements.txt` совместим с lock L1 (Python 3.11, opencv < 5, pytest 9, PyYAML 6.0.3). Внешние бинарники:
ffmpeg/ffprobe, nvidia-smi (опционально, для VRAM). torch eval не импортирует. Нейросетевых моделей нет: `models.json` = `[]`.

## Для интегратора

* Контракт: `eval/_contracts.py` → `contracts.sentinel_contracts`, при отсутствии — `eval/_contract_fallback.py`.
* **A03 (`backend.pipeline`)**: inproc зовёт `run(path, job_id, progress, deadline_s)` / `warmup()`. Конфиг пайплайна
  передаётся через env `SENTINEL_PIPELINE_CONFIG` (имя — `runner.pipeline_config_env`), флаги — через `SENTINEL_*`.
  Их нужно читать при вызове `run()`, а не только при импорте: тогда работает и `--inproc-arms`.
  **Просьба положить в `meta`:** `version`; `duration` (по PTS); `coverage` (0..1 или `{stage: доля}`; без неё partial не отличить
  от «чисто»); `models` (список hf_id); `mock` (true только в тестах/демо); `vram_peak_mb`; `timings.total_s`; `warnings`
  (строки с deadline/timeout при деградации); `error` при failed; `scores_raw` при `SENTINEL_RAW_SCORES=1`.
* **A02 (API)**: jury_sim ожидает 401 / 415 / 404 / 413, `error` у failed. Загрузка ~29 МБ должна укладываться в таймаут
  Cloudflare (~100 с) — замер в пункте «загрузка < 100 с».
* **A12**: `scores_raw.jsonl` для `ml.fusion.tune`; пороги — только на train/calib; выбор пулинга — по `eval.sanity`.
* **A10**: train — по `data/splits.csv` (не holdout), аугментации — после split.
* **A01**: решить про `reports/` в .gitignore (журнал — только агрегаты).
* Ограничения: CLI-режим не знает латентности по каждому видео; сопоставление жадное (на 1–2 события результат тот же);
  VRAM по nvidia-smi — пик всей карты с шагом опроса 0.5 с.
