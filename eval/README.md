# eval/ — оценка «как у жюри» (A16)

Инструменты оценки Kassa Sentinel: разметка, групповой split, прогон пайплайна, метрики, scorecard,
наш proxy-балл, устойчивость к синтетическим возмущениям, имитация жюри по публичному URL.
**Измерений на реальных данных здесь нет** (данные и GPU только локально у команды): код проверен
синтетическими тестами и mock-пайплайном. Mock-тесты проверяют код, а не качество CV.

## Быстрый старт (локально, на машине с данными)

```bash
python -m eval.labels --xlsx data/open_labels.xlsx --videos data/videos --out data/labels.csv
python -m eval.split  --labels data/labels.csv --out data/splits.csv          # один раз, потом не менять
python -m eval.run    --videos data/videos --subset train,calib --out runs/eval/dev1 [--raw]
python -m eval.run    --api https://<tunnel> --key $SENTINEL_API_KEY --subset calib --out runs/eval/api1
python -m eval.run    --subset holdout --final --out runs/eval/final      # только для итоговых цифр (логируется)
python -m eval.robustness --subset calib --variants persp_left,rot_p10,mirror,zoom80 --out runs/robust
python -m eval.jury_sim --url https://<tunnel> --key $SENTINEL_API_KEY --videos data/videos --n 6
python -m eval.score_like_jury --metrics runs/eval/dev1/metrics.json --robustness runs/robust/robustness.json
python -m eval.report --run runs/eval/dev1          # runs/eval/dev1/report.html — только локально
python -m eval.loo --scores runs/eval/dev1/scores_raw.jsonl   # exploratory LOGO + nested CV порога
pytest eval/tests -q            # без GPU и весов; ffmpeg-тесты помечены slow
```

Все выходы пишутся в `runs/` (в .gitignore). `preds/*.json` и HTML-отчёты содержат кадры и описания сцен:
**они остаются только локально**. Наружу (облачным агентам/LLM, в чат, в PR) — только
`summary_public.json`: агрегаты без кадров, описаний и ID видео (`eval.metrics.public_summary`).

## Модули и публичные функции

| Файл | Что делает | Главное API |
|---|---|---|
| `labels.py` | xlsx → `data/labels.csv` (`video_id,file,label,t_sec,md5,group,t_all,conflict,raw,sha256`) | `parse_timecodes`, `parse_label`, `read_labels_xlsx`, `build_labels`, `write_labels_csv`, `read_labels_csv`, `read_types_csv`, `load_ground_truth` |
| `split.py` | групповой train/calib/holdout + фолды → `data/splits.csv` | `make_splits`, `write_splits`, `load_splits`, `ids_for`, `log_holdout_access` |
| `run.py` | прогон папки (inproc / cli / api) → `preds/*.json`, `_runinfo.json`, метрики | `run_folder`, `select_videos`, `write_scores_raw`, `InprocRunner`, `ApiRunner`, `run_cli_batch` |
| `metrics.py` | метрики, scorecard, публичная сводка | `evaluate`, `evaluate_dirs`, `scorecard`, `public_summary`, `match_one_to_one`, `wilson`, `load_predictions` |
| `schema.py` | проверка ответа по схеме ТЗ (+ `contracts/api_schema.json`, если есть) | `check_api_response`, `decode_jpeg_b64`, `TZ_JSON_SCHEMA` |
| `score_like_jury.py` | **наш proxy-балл** 0..100 | `score`, `format_score` |
| `robustness.py` | синтетические возмущения ракурса/качества, смена вердикта, сдвиг таймкода | `make_variant`, `VARIANTS`, `compare` |
| `jury_sim.py` | имитация жюри по URL | `run_jury_sim`, `JurySim`, `format_table` |
| `report.py` | локальный HTML-отчёт (кадры! не выносить) | `build_report` |
| `loo.py` | LOGO (**exploratory**) и nested CV порога по `scores_raw.jsonl` | `logo`, `nested_cv`, `fit_threshold`, `video_score` |

Все пороги вынесены в `eval/config.yaml` (`--config my.yaml` переопределяет).

## Разметка (`labels.py`)

* Имена: латинская X / кириллическая Х / `x_03` → `X-<n>`. Если разные файлы дают один ID, второй получает `X-n#2`.
* Таймкоды: «22 секунда» → 22, «1минута 8 секунда» / «1 мин. 8 сек.» / «1:08» → 68, «без нарушения» → пусто.
  Excel-время «1:08», прочитанное как 01:08:00, превращается в 68 с.
* Дубликаты по содержимому (md5 + sha256): Х-4≡Х-47, Х-5≡Х-48 → одна `group`, флаг `conflict`.
  **Исходные строки сохраняются** (`raw`), конфликт **не решается выбором**:
  * разные таймкоды (57 / 68 с) → одно GT-событие с альтернативами `[57, 68]`; локализация считается ещё и
    без конфликтных групп (`localization_no_conflict`);
  * разные метки (нарушение / без) → `label=conflict`: группа исключается из бинарных метрик и выводится в `label_conflict_ids`.
* `data/labels_types.csv` (`video_id,t_sec,type_id`) поддерживается, **но для бинарной оценки не нужен.
  Без настоящей разметки типов точность типа = n/a** (никаких proxy и выдуманных меток).

## Split и честность цифр (`split.py`)

* Группа = дубликаты по содержимому ∪ одна сессия съёмки (`--sessions video_id,session_id`, необязательно).
  Члены группы всегда попадают в один split. Стратификация по метке, фиксированный seed. Split делается
  **до аугментаций**: варианты из `robustness.py` и обучающие аугментации A10 наследуют split исходника.
* **train/calib** — выбор моделей и порогов. **holdout** — итоговые цифры. Открыть holdout можно только
  через `eval.run --subset holdout --final`; каждое обращение пишется в `data/holdout_access.jsonl`,
  при повторном появляется предупреждение. Перезапись `splits.csv` — только с `--force`.
* LOO / leave-one-group-out по тем же данным, на ошибках которых много раз меняли пайплайн, — это
  **exploratory**, а не независимый тест. Для итоговых цифр: нетронутый групповой holdout или nested CV
  (подбор порогов только на outer-train).

## Метрики (`metrics.py`)

* **Scorecard (главное):** n (+/−), recall / precision / specificity с 95% интервалами Вильсона
  (при n≈20–40 интервалы широкие, так и должно быть), FN, FP, failed, timeouts, degraded, hit событий, лишние кандидаты,
  латентность (median / p90 / max, превышения SLA 60 с) и версии: `eval_version`, `git_sha`, `meta.version` пайплайна,
  sha256 labels.csv, subset, режим.
* По видео: confusion; failed/missing в P/R считаются как «нет нарушения» (пропуск), в accuracy — как ошибка.
  Timeout распознаётся по статусу failed и тексту ошибки/предупреждения (deadline, timeout, таймаут).
* **Локализация — one-to-one:** жадное сопоставление нарушений и GT-событий по |Δt| в окне `match_window_s`
  (5 с). hit@0.5/1/2/3 считается по событиям. Знак Δt даёт категории «рано» / «поздно». Справочно — Δt без окна.
  **Лишние кандидаты** (несопоставленные нарушения) считаются отдельно и в FP по видео не входят:
  GT размечена одной секундой на видео, разметка событий неполная.
* Кадр-доказательство: валидный JPEG в ±1 с от сопоставленного GT-таймкода; 1–3 кадра на нарушение.
* Нормы: у каждого нарушения ≥1 rule, rules из `norms/catalog.yaml` для этого `type_id`, URL adilet, доля `verified`.
  **Наличие quote/URL ≠ правильная трактовка нормы.** Корректность — только по ручной экспертизе:
  `data/norms_review.csv` (`type_id,url,clause,correct`) → `reviewed_correct_rate`.
* Схема ТЗ (обязательные поля, enum, base64 JPEG), язык описаний (hedged «признаки возможного нарушения»,
  запрещённые домыслы: «украл», «умысел», «подделка»…), согласованность вердиктов на md5-дубликатах.

## Proxy-балл (`score_like_jury.py`) — НЕ оценка жюри

Формула жюри неизвестна. Это наша свёртка scorecard в одно число для сравнения итераций; решения принимаются по scorecard.
Веса — критерии ТЗ (25/15/15/15/10/10/10). Компонент без настоящих данных = n/a: он исключается, остальное
перенормируется на 100 (`missing` показывает, что исключено).

| Компонент | Формула (доля 0..1 × вес) |
|---|---|
| detection 25 | `1 − (2·FN + 1·(FP + failed⁻)) / (2·P + N)`: пропуск вдвое дороже ложного срабатывания |
| timecode 15 | среднее по позитивам: `0.667·ступень(|Δt|: ≤1→1, ≤2→.75, ≤3→.5, ≤5→.25) + 0.333·[кадр в ±1 с]`, FN → 0 |
| type 15 | точность по `labels_types.csv`; **без неё n/a** |
| norms 15 | proxy полноты ссылок: `0.4·[≥1 rule] + 0.3·[rules из каталога] + 0.3·verified`, FN → 0 (не корректность!) |
| robustness 10 | proxy на синтетике: `0.7·(1 − смена вердикта) + 0.3·стабильный таймкод`; без `robustness.json` n/a |
| product 10 | proxy: `0.4·схема + 0.2·JPEG + 0.2·1–3 кадра + 0.2·hedged·(1−forbidden)` (UI жюри оценит само) |
| architecture 10 | proxy: `0.5·доля в SLA + 0.3·(1 − failed) + 0.2·jury_sim pass_rate` |

## Устойчивость (`robustness.py`)

Варианты: `persp_left/right`, `rot_±10/±20`, `zoom80`, `mirror`, `bright_±30`, `blur`, `h264_crf32`, `fps25`
(ffmpeg, размер кадра сохраняется). Если `ml.ingest.augment` предоставляет `make_variant(src, dst, name)`
(или `apply_variant` / `augment_video`), используется он. Выход: доля смены вердикта, сдвиг таймкода и
точность по GT для каждого варианта, плюс `variants_manifest.csv` с наследованием split.
**Синтетика не доказывает качество на реальном новом ракурсе.** Holdout в стресс-тест не брать.

## Имитация жюри (`jury_sim.py`)

Порядок проверок: auth 401 (без ключа и с неверным; POST и GET), `.txt` → 415, неизвестный job → 404 (мягкая проверка),
затем **sequential** (видео по одному), затем **bounded stress** (≤ `--concurrency`, по умолчанию 5), в конце большой файл
(→ 413/4xx или job=failed, но не 5xx). Лимит 300 с считается **от приёма каждого job, включая очередь**.
Одновременная загрузка 5+ видео не обязательна, но в stress-фазе возможна. Каждый ответ проходит проверку схемы ТЗ.
В `runs/jury_sim.json` пишутся только статусы и тайминги. Код возврата 1 при любом FAIL.

## Зависимости и модели

`eval/requirements.txt`: openpyxl, PyYAML, numpy, requests, jsonschema (опц.), opencv-python-headless (только тесты
и миниатюры), pytest. Внешние бинарники: ffmpeg/ffprobe (robustness). Нейросетевых моделей нет: `models.json` = `[]`.

## Для интегратора

* Контракт: `eval/_contracts.py` импортирует `contracts.sentinel_contracts`, при отсутствии — `eval/_contract_fallback.py`
  (копия замороженного v1). После мерджа A01 fallback можно удалить.
* `backend.pipeline` (A03): inproc-режим зовёт `run(path, job_id, progress, deadline_s)` и `warmup()`, сериализует через
  `contracts.serialize.result_to_api` (fallback: `asdict(AnalysisResult)` — это надмножество схемы ТЗ) и дописывает
  `meta` целиком. CLI-шаблон: `config.yaml → runner.cli_cmd` (`python -m backend.pipeline batch {videos} --out {out}`);
  ожидается по одному `<stem>.json` на видео. **Предложение A03:** класть в `meta` поля `version`, `duration`,
  `timings.total_s`, `warnings` (строки с «deadline» / «timeout» при деградации), а при `SENTINEL_RAW_SCORES=1` —
  `scores_raw` (сырые скоры кандидатов для `ml.fusion.tune`).
* API (A02): `--api` шлёт `?raw=1` при `--raw` (можно игнорировать). jury_sim ожидает 401 / 415 / 404 / 413.
* A12 `ml.fusion.tune`: вход — `runs/.../scores_raw.jsonl` (`video_id, group, label, gt_times, verdict, scores_raw, violations`).
  Подбирать пороги только на train/calib (`--subset train,calib`).
* A10 (temporal): для train брать `data/splits.csv` (split ≠ holdout); аугментации делать после split.
* A13: `norms.load_catalog()` используется для проверки, что rules взяты из каталога; иначе читается `norms/catalog.yaml`.
* Env: `SENTINEL_API_KEY` (по умолчанию для `--key`), `SENTINEL_MOCK=1` (в тестах), `SENTINEL_RAW_SCORES=1` (raw-режим).
* Ограничения/TODO: CLI-режим не знает латентности по каждому видео (`latency_s_estimated` = общее время / n);
  сопоставление жадное, не венгерское (на 1–2 события на видео результат тот же).
