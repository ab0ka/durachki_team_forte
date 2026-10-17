# A14 — `copilot/` — Sentinel Copilot (опциональный модуль)

> **Подключение, форматы, env, отключение, тесты, что проверено:** см. [`INTEGRATION.md`](INTEGRATION.md).

Локальный ассистент проверяющего: отвечает на вопросы по **результату анализа видео** (`job_result`)
и **нормам из `norms/catalog.yaml`** (RAG), ссылается только на проверенные источники, при нехватке
данных честно говорит «по имеющимся данным это установить нельзя». Плюс детерминированное
«заключение для проверяющего» в markdown.

**Статус: опциональный, по умолчанию выключен в проде.** На 8 ГБ VRAM ресурсы делятся с VLM. Если
copilot выключен или локальной модели нет, `ask` возвращает **HTTP 501** с пустым `answer`.
Выдуманного ответа нет, внешних AI API нет. Запуск API и пайплайна от copilot не зависит:
импорт ничего не загружает.

> Честно о том, чего НЕТ: модель **не дообучалась** (LoRA/QLoRA, массовая генерация датасета и
> конвертация весов вынесены за пределы критического пути, скриптов в ветке нет). Реальный
> инференс на GGUF в этой сессии **не запускался** (нет GPU/весов/данных). Измерений качества и
> скорости нет. Всё проверено на моках и синтетике (см. «Проверки»).

## Что сделано

| Файл | Что внутри |
|---|---|
| `service.py` | `answer()`, `status()`, `warmup()`, `release()`, `set_gpu_lock()`; таймаут, коды ошибок, SSE-поток событий |
| `router.py` | FastAPI `APIRouter`: `POST /api/v1/jobs/{id}/ask` (+SSE), `GET /api/v1/copilot/status`, `GET /api/v1/jobs/{id}/report` |
| `report.py` | `report(job_result) -> markdown`, без LLM, нормы перепроверяются по каталогу |
| `corpus.py` | загрузка `norms/catalog.yaml` + `norms/sources/*.md`, чанкинг (front matter, заголовки, «N.», «п. N», «Статья N»), проверка цитат |
| `rag.py`, `embed.py` | гибридный поиск: косинус (e5 или hashing без весов) + BM25, numpy-индекс, кэш эмбеддингов, бонус нормам по `type_id` |
| `llm.py` | бэкенды: `llama_cpp` (in-process, ленивая загрузка, частичный offload по свободной VRAM, выгрузка по простою), `llama_server` (только loopback/частные адреса, прокси отключены), `mock`, `extractive`, `scripted` (тесты) |
| `prompts.py` | системный промпт (рус.), контекст: факты результата + нумерованные нормы `[N1]…`, нарушения `[V1]…` |
| `guard.py` | пост-проверка ответа (ниже) |
| `extractive.py` | классификатор намерений + детерминированный ответ (mock/extractive) + обязательные отказы |
| `fixtures/norms/` | **ТЕСТОВАЯ** мини-фикстура норм (вымышленные документы, URL `*.invalid`), только для тестов/mock |

### Пост-проверка (`guard.check_answer`)
- `[Nk]` допустимы только из выданного контекста; превращаются в сноски `[1]`, `[2]`, которые
  соответствуют `citations[i].ref`. Выдуманные маркеры удаляются.
- «п. 15», «пункт 15», «статья 5» в тексте: пункт должен быть в контексте или в каталоге/корпусе,
  иначе предложение вырезается.
- Названия актов («Закон «…»», «Правила «…»») сверяются с документами корпуса. URL — только известные корпусу.
- Таймкоды допускаются только из результата (± `guard.timecode_tolerance_s`) или из вопроса/истории.
- Денежные суммы допускаются, только если они явно есть в результате или в вопросе.
- Предложение без лексической опоры на результат/нормы (доля < `guard.min_support_ratio`) считается
  домыслом и вырезается. Если не осталось ничего, ответ: «По имеющимся данным это установить нельзя:
  ответ модели не подтверждён…».
- Предложения режутся с учётом сокращений («п. 9», «ст. 5», «22 с.»), поэтому «п. 999» не проскакивает.
- `<think>…</think>` (Qwen3) вырезается. В SSE уходят только проверенные предложения.
- `grounded = true`, только если в сыром ответе не было неподтверждённых ссылок и ответ на что-то
  опирается: есть цитата, отказ или ссылка на факты результата.
- Нормы, приложенные к результату (`violations[].rules`), тоже перепроверяются по каталогу.
  Неподтверждённые не цитируются, в `warnings` появляется предупреждение.

### Обязательные отказы (без вызова LLM)
Вопросы о сумме, подлинности денег, умысле, личности (кто это, ФИО, распознавание лица), допуске,
проводках и содержании разговоров получают ответ «По имеющимся данным это установить нельзя: по видео
это не определяется…». Вопросы не по теме получают «Я отвечаю только на вопросы по результату анализа
видео и нормам». Штрафы и прочее, чего нет в источниках, получают «По имеющимся данным это установить
нельзя: в источниках нет информации об этом». Вопрос о моменте, где в результате нет событий, тоже
получает «По имеющимся данным…».

## Публичные функции

```python
from copilot import answer, report, status, warmup, release, make_router

answer(job_result: dict | AnalysisResult | str, question: str, history: list[{"role","content"|"text"}]) -> dict
# -> {"answer": str, "citations": [{"document","clause","url","ref","quote","verified"}], "sources": [str],
#     "grounded": bool, "refusal": bool, "intent": str, "backend": str, "warnings": [...], "removed_citations": [...],
#     "latency_ms": int, "disclaimer": str,
#     "error": None | "disabled" | "unavailable" | "timeout" | "busy" | "not_ready" | "empty_question" | "llm_error" | "internal_error"}
report(job_result) -> str            # markdown, работает и без LLM
status() -> {"enabled","available","backend","reason","message","llm","warnings",...}
warmup() -> bool / release()         # прогреть / выгрузить LLM (освободить VRAM); исключений не бросают
copilot.service.set_gpu_lock(lock)   # общий замок GPU от A03 (acquire(timeout=)/release())
```
`job_result` — dict по схеме API (`result_to_api`) или `AnalysisResult`. Base64-кадры в LLM не передаются.

## HTTP API (`copilot/router.py`)

| Метод | Путь | Ответ |
|---|---|---|
| GET | `/api/v1/copilot/status` | `{"enabled","available","backend","reason",...}` |
| POST | `/api/v1/jobs/{job_id}/ask` | тело `{"question": str(1..2000), "history": [{"role":"user"\|"assistant","content" или "text"}] (≤40)}` → `answer()` |
| POST | `/api/v1/jobs/{job_id}/ask?stream=1` (или `Accept: text/event-stream`) | SSE: `event: token` `{"text"}` (только проверенные предложения) …, затем `event: done` (полный ответ) или `event: error` |
| GET | `/api/v1/jobs/{job_id}/report` | `text/markdown` |

Коды: 404 нет задачи · 409 анализ не завершён · 422 пустой вопрос / неверная история ·
**501 copilot выключен или модели нет** · 503 GPU/модель заняты, провайдер задач не задан или упал ·
504 таймаут · 500 ошибка модели / внутренняя. Тело ошибки: `{"detail": {"error", "message", "answer": "", ...}}`.

## Режимы и конфиг (`copilot/config.yaml`, все пороги там)

| `enabled` | Поведение |
|---|---|
| `mock` (по умолчанию) | включён только при `SENTINEL_MOCK=1` (mock-ответы помечены `backend: "mock"`); в проде выключен |
| `true` | включён: `llama_server` → `llama_cpp`; модели нет — `unavailable` (501) |
| `false` | выключен (501), `report` и `status` работают |

`llm.backend`: `auto` | `llama_cpp` | `llama_server` | `mock` | `extractive`. Режим `extractive`
(детерминированные ответы без генерации) включается **только явно** и никогда не подставляется
автоматически.

Env-переменные: `COPILOT_ENABLED`, `COPILOT_LLM_BACKEND`, `COPILOT_MODEL_PATH` (GGUF),
`COPILOT_LORA_PATH`, `COPILOT_GPU_LAYERS` (`auto`/`-1`/`0`/`N`), `COPILOT_SERVER_URL`,
`COPILOT_TIMEOUT_S`, `COPILOT_N_CTX`, `COPILOT_IDLE_UNLOAD_S`, `COPILOT_EMBED_BACKEND` (`auto`/`e5`/`hashing`),
`COPILOT_EMBED_MODEL`, `COPILOT_EMBED_DEVICE`, `COPILOT_CACHE_DIR`, `COPILOT_TOP_K`,
`COPILOT_NORMS_DIR` (папка с `catalog.yaml` и `sources/`), `COPILOT_ALLOW_FIXTURE`, `COPILOT_CONFIG`
(доп. YAML), `SENTINEL_MOCK`, `HF_HUB_OFFLINE=1`.

## Как запустить

```bash
pip install -r copilot/requirements.txt
python -m pytest copilot/tests -q                 # без GPU, весов и сети

# mock-демо (фикстура норм, без модели)
SENTINEL_MOCK=1 python -c "from copilot import answer; print(answer({'job_id':'1','status':'done','verdict':'no_violation','violations':[]}, 'Есть нарушения?'))"

# реальная локальная модель (на этой сессии не запускалось)
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
COPILOT_ENABLED=1 COPILOT_MODEL_PATH=weights/copilot/Qwen3-4B-Instruct-2507-Q4_K_M.gguf COPILOT_GPU_LAYERS=0 ...
# или отдельный процесс: llama-server -m <gguf> --port 8081 -ngl 0 ; COPILOT_SERVER_URL=http://127.0.0.1:8081
```

## Модели и лицензии (`copilot/models.json`, всё OPTIONAL)
- `intfloat/multilingual-e5-small` — MIT — эмбеддинги RAG (без весов работает hashing).
- Qwen3-4B-Instruct-2507 GGUF Q4_K_M — Apache-2.0 — рекомендуемая LLM (~2.5 ГБ).
- Qwen2.5-7B-Instruct GGUF Q4_K_M — Apache-2.0 — альтернатива (~4.7 ГБ).
- Библиотеки: llama-cpp-python (MIT), sentence-transformers (Apache-2.0), numpy (BSD), FastAPI (MIT).
  Qwen2.5-3B, ultralytics, облачные SDK не используются. Тест `test_no_cloud_ai_sdk_imports` это проверяет.

Размеры приблизительные: HF API из этой среды был недоступен.

## Проверки (реально выполнены в этой сессии)
`python -m pytest copilot/tests -q` даёт **101 passed** (Python 3.11, CPU, без весов и сети). Полный
список проверенного и того, что требует локального GPU, — в [`INTEGRATION.md`](INTEGRATION.md#6-что-реально-проверено-и-что-требует-локального-gpu). Кратко:
- выдуманные `[Nk]`, «п. 999», названия актов, URL, таймкоды, суммы и предложения без опоры
  вырезаются и в ответе, и в SSE;
- нет облачных вызовов (AST-скан, блокировка сокетов, только локальный `server_url`), нет
  автоскачивания (e5 только `local_files_only`, llama.cpp только `model_path`);
- нет тяжёлых импортов и загрузки моделей при import. Без весов: `status.available=false`, `ask` → 501;
- ошибки, таймауты и занятость copilot дают JSON 5xx и не затрагивают другие эндпоинты API;
  `warmup`/`release`/`status` не бросают исключений;
- совместимость с текущим кодом A02 (двойной монтаж `/api/v1` + `/ui/v1`) и A15
  (`history[{role, text}]`, `sources`).

## Для интегратора

**A02 (API).** В ветке A02 подключение уже сделано (`_mount_copilot`, двойной монтаж), сигнатура
совпадает. Общий шаблон: подключить лениво и защитить тем же API-ключом / той же сессией, что и
остальные эндпоинты:
```python
try:
    from copilot.router import make_router
    app.include_router(make_router(get_job=lambda job_id: jobs.get_result_dict(job_id)),  # dict | AnalysisResult | None, можно async
                       dependencies=[Depends(require_api_key)])
except Exception:  # copilot опционален — API стартует без него
    pass
```
`get_job` должен возвращать результат именно этой задачи. Задачи нет → `None` (404). Пока задача
считается → `status="processing"` (409). Эндпоинты не публикуются без ключа: ключ проверяет
dependency A02, роутер сам ключи не знает.

**A03 (пайплайн, GPU).** Copilot по умолчанию выключен. Если его включают на той же GPU:
`copilot.release()` перед стадиями VLM/детекции освобождает VRAM, `copilot.service.set_gpu_lock(lock)`
даёт общий замок (пока GPU занята, ответ `busy` → 503). `llm.gpu_layers: auto` оставляет
`vram_reserve_mb` другим и при нехватке уходит на CPU. После простоя `idle_unload_s` модель
выгружается. Для `warmup()` в общем `backend.pipeline.warmup()` copilot не нужен.

**A15 (UI).** Текущий клиент A15 совместим: `history[{role, text}]` принимается, `sources`
отдаются, при 501 панель скрывается. Опционально: перед показом панели вызвать
`GET …/copilot/status` (`available=false` → скрыть). При SSE по `event: done` заменять текст на
`done.answer`. Показывать `disclaimer`, при `grounded=false` пометить ответ «источники не
подтверждены». Кнопка «Заключение (md)» ведёт на `GET /jobs/{id}/report`.

**A13 (нормы).** Ожидается `norms/catalog.yaml` формата из ТЗ (`types.Vxx.rules[] = {document, clause, url,
quote, verified}`, `not_observable`). `norms/sources/*.md` лучше с front matter `title:` и `url:`,
пункты — заголовками «### п. 12» или абзацами «12. …». Подпункты «1)» остаются внутри пункта.
Если файла нет, а `SENTINEL_MOCK` не задан, copilot работает без норм («в источниках нет»).

**A17 (deploy).** Веса в `weights/copilot/` (в .gitignore), кэш эмбеддингов в `cache/copilot/`.
Для офлайна задать `HF_HUB_OFFLINE=1`. Для GPU нужен llama-cpp-python, собранный с CUDA, иначе
только CPU. llama.cpp server может работать отдельным контейнером. Допустимы только адреса
loopback, RFC1918 или docker-имена без точки.

**A16/A18.** Метрик и измерений по copilot нет, дообученной модели нет. В питче не заявлять
«обученный локальный Claude». Корректная формулировка: «опциональный локальный ассистент на
открытой LLM (Apache-2.0) с проверкой источников».

**Удалить при интеграции:** `copilot/_contract_fallback.py`, когда `contracts/` будет в main.
`copilot/_contract.py` сначала берёт `contracts.sentinel_contracts`. В рантайме copilot работает с
dict/dataclass без жёсткой привязки к типам, сам контракт используется в тестах.

### Известные ограничения / TODO
- LoRA/QLoRA, генерация датасета 1–3 тыс. примеров, оценка grounding до/после и конвертация GGUF
  отложены (вне критического пути). Под них уже готово: промпт-билдер `prompts.build_messages`
  (один формат для обучения и инференса), `ScriptedLLM` и `guard` для метрики «доля валидных цитат/отказов».
- Пороги RAG (`rag.min_score*`, веса гибрида) не тюнились на реальном каталоге A13.
- Классификатор отказов regex-based: возможны ложные отказы на необычные формулировки.
- FAISS не нужен (корпус в сотни чанков), используется numpy.

### Решения, принятые автономно
- По умолчанию `enabled: mock`: в проде copilot выключен, при `SENTINEL_MOCK=1` работает mock для демо UI.
- Детерминированный `extractive` не подставляется автоматически вместо LLM (честный 501), его можно включить только явно.
- `report` не требует LLM и доступен даже при выключенном ассистенте.
- Ссылки в ответе только маркерами `[Nk]` из контекста. Это надёжнее, чем просить модель писать названия документов.
