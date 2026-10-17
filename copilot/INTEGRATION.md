# Sentinel Copilot — подключение (A14 → интегратор L2), фаза 2

Copilot — **опциональный** модуль: объясняет проверяющему результат и нормы, на вердикт, таймкод и тип
**не влияет** и ресурсы у анализа не забирает. По умолчанию в проде выключен (`enabled: mock`). Если его
не подключить, удалить или выключить, остальная система работает как прежде.

| Ситуация | Поведение |
|---|---|
| Импорт `copilot`, `copilot.router` | без весов, без torch/transformers/llama_cpp, без сети, без чтения корпуса |
| Copilot выключен (по умолчанию) | `POST …/ask` → **501** `error="disabled"`; `status.available=false` |
| Включён, модели нет (`ml.vlm.service` нет/недоступен) | **200**, режим `retrieval`: ответ из фактов задачи и найденных норм, первая строка ответа «(Генеративный режим недоступен…)»; `status`: `available=true, generative=false, mode=retrieval` |
| Включён, модель есть | режим `generative`: общая Qwen3-VL-4B через `ml.vlm.service.chat`, `priority="low"` |
| Идёт анализ видео (busy-probe) или GPU не освободился за `wait_s` | модель не вызывается, ответ `retrieval` с пометкой «временно недоступен (GPU занят анализом)» |
| Ошибка модели / таймаут | 500 `llm_error` / 504 `timeout`, JSON, `answer=""`; другие эндпоинты не затронуты |
| Неподтверждённое в ответе модели | вырезается до выдачи (и в SSE); не осталось ничего → «По имеющимся данным это установить нельзя…» |

Mock (`backend: "mock"`) — только при `SENTINEL_MOCK=1` (тесты, демо), помечен в ответе и статусе.
Production сам на mock не переключается.

## 0. Состояние соседей (проверено чтением веток, 2026-10-17)
- **A02** (`backend/api/main.py::_mount_copilot`) монтирует `make_router(get_job, prefix="/api/v1"|"/ui/v1",
  dependencies=[…])`. Сигнатура совпадает, менять ничего не нужно.
- **A15** (`client.ts`) вызывает `GET …/copilot/status` (`available=false` → панель скрыта), `POST …/ask`
  с `{question, history:[{role, text}]}` → `{answer, sources}`, `GET …/report` (.md). Всё совместимо.
  Новое и необязательное: `GET …/jobs/{id}/suggestions` вместо захардкоженных 3 вопросов,
  `GET …/report?format=html`.
- **A11** (`ml/vlm`): `ml.vlm.service.chat` в ветках **пока нет** (есть `runtime.get_backend/gpu_guard`).
  Copilot привязан к `ml.vlm.service.chat` по спецификации, а до его появления работает в режиме retrieval.
  Временный адаптер к `ml.vlm.runtime` есть, но выключен (`vlm_service.runtime_adapter: false`).
- **A13** (`norms/`): каталог v2 (`legal_status`, `legal_note`, `exclusions`, `documents`) и источники
  `retrieval: search_snippets` разбираются честно: разделы «Не цитаты»/«Реквизиты»/«TODO» не
  цитируются, все фрагменты помечены «не сверено», типы `disabled` дают «без пункта».

## 1. Подключение

```python
# A02 (уже сделано в ветке A02): лениво, тем же ключом/сессией, что и остальные эндпоинты
from copilot.router import make_router
app.include_router(make_router(get_job=provider, prefix="/api/v1"), dependencies=[Depends(require_api_key)])
app.include_router(make_router(get_job=provider, prefix="/ui/v1"), dependencies=[Depends(ui_auth)])

# A02/A03: низкий приоритет — пока идёт анализ, копилот модель не трогает
import copilot.service as cs
cs.set_busy_probe(lambda: job_manager.has_running_jobs())   # любая функция () -> bool
cs.set_gpu_lock(shared_gpu_lock)                            # только для явного llama_cpp; VLM — замок A11
```

`provider(job_id)` возвращает результат именно этой задачи (dict по схеме API/UI-report,
`AnalysisResult` или JSON-строку) или `None` (404). `status="processing"` → 409, исключение → 503.

### Ожидаемый контракт `ml.vlm.service.chat` (A11)
Подходит любая из подписей, лишние параметры не передаются:
```python
chat(messages=[{"role": "system"|"user"|"assistant", "content": str}], *, max_new_tokens=512,
     timeout_s=45.0, priority="low", temperature=0.2, wait_s=10.0) -> str | GenResult | {"text": ...}
chat(system: str, prompt: str, images=[], max_new_tokens=..., timeout_s=...) -> ...
available() -> bool | (bool, reason)        # необязательно; False — генеративный режим выключен
```
Ожидаемое поведение: `priority="low"` уступает анализу видео. При занятости — исключение с именем
`Busy/GPUBusy/QueueFull/BackendBusy` или `GenResult(error="busy")`, таймаут — `TimeoutError` или
`timed_out=True`. Модель держит и выгружает A11/A03. Copilot её сам не грузит и не выгружает.

### Авторизация и общая GPU-очередь
- `/ask`, `/report`, `/suggestions`, `/copilot/status` подключать только с той же dependency, что
  `GET /jobs/{id}`. Роутер ключей не знает. Провайдер отдаёт лишь задачи, доступные вызывающему.
- GPU одна: copilot идёт через очередь A11 с `priority="low"`. `set_busy_probe` гарантирует, что во
  время анализа модель не вызывается вообще (ответ без генерации). VRAM copilot сверх VLM не
  занимает: e5 и retrieval работают на CPU.

## 2. Вход и выход

`POST …/jobs/{id}/ask` (тело `{"question": str(1..2000), "history": [{"role", "content"|"text"}] (≤40)}`) → 200:
```json
{"answer": "…[1]…", "citations": [{"document": "...", "clause": "п. 20", "url": "https://adilet.zan.kz/...",
  "fragment": "Ф3. Пункт 20 (приём наличных)", "ref": 1, "quote": "...", "verified": false,
  "relation": "catalog", "status": "candidate", "note": "норма каталога для этого типа; статус в каталоге: кандидат, требует сверки; …",
  "events": [1], "short": "Правила кассовых операций и инкассации (НБРК № 231)"}],
 "sources": ["[1] Правила … (НБРК № 231), п. 20 — https://… (норма каталога; кандидат; не сверено)"],
 "grounded": true, "refusal": false, "generative": true, "mode": "generative", "notice": null,
 "intent": "why", "backend": "vlm_service", "removed_citations": [], "warnings": [], "latency_ms": 1234,
 "disclaimer": "Подсказка для проверяющего, не юридическое заключение.", "error": null}
```
- `relation`: `catalog` (норма каталога A13 для типа события; `events` — номера нарушений) или
  `retrieved` (найдено поиском по тексту, связь не подтверждена). `verified=false` или
  `status=candidate` означает «требует сверки». Если в каталоге нет пункта, ответ говорит «без пункта».
- `mode`: `generative` | `retrieval` | `mock`. При `retrieval` в `notice` и первой строке `answer`
  стоит пометка режима.
- Ошибки: `{"detail": {"error", "message", "answer": "", …}}`.

| `error` | HTTP |
|---|---|
| `disabled` (и `unavailable` при `no_model_mode: unavailable`) | 501 |
| `not_ready` | 409 |
| `empty_question` | 422 |
| `busy` (только `busy_policy: error`) | 503 |
| `timeout` | 504 |
| `llm_error`, `internal_error` | 500 |

`GET …/jobs/{id}/suggestions?limit=4` → `{"job_id", "suggestions": [...], "by_type": {"V05": [...]}}`.
Вопросы по 3–4 на тип (V01–V09, V99) из `copilot/suggestions.yaml`. Учитываются чистые и неполные
результаты. Вопросов о личности, сумме и умысле нет.

`GET …/jobs/{id}/report` → markdown. `?format=html` → автономный HTML: экранирован, CSP
`default-src 'none'`, кадры-доказательства только из валидного base64 JPEG. Работает без модели и при
выключенном copilot. В отчёте: полнота анализа, «почему отмечено» (факты, детекторы, VLM — из
`meta.ui.evidence_overlays`/`meta.violation_extra`), нормы со статусом или «без пункта», исключения
каталога, спорные места, чек-лист, дисклеймер.

SSE (`?stream=1`): события `token` (только проверенные предложения) и затем `done` или `error`. В
режимах retrieval/mock ответ приходит одним `token`.

### Что copilot читает из результата
`violations[].{type, type_id, timecode_sec, t_start, t_end, confidence, description, evidence_frames[].timestamp_sec, rules}`.
Из `meta`: `completeness` (A01), `complete`/`partial` (A03), `coverage`, `analysis_complete`,
`ui.evidence_overlays[].{facts, sources, vlm}` и `ui.review_hints` (A12), `violation_extra` (A01).
Плюс `candidates`, `verdict_state`. Кадры, фигуры, `scores_raw` и `timeline` в контекст не попадают.

**Полнота:** неполный анализ (failed, partial, `complete=false`, required stage failed,
coverage < 1) никогда не превращается в «нарушений нет». Нет сведений о полноте — состояние
«полнота не подтверждена» (unknown ≠ полный).

## 3. Зависимости, переменные, веса
`copilot/requirements.txt`: numpy, PyYAML, pydantic, fastapi. Опционально sentence-transformers (e5).
Модель — общая с A11, своих весов LLM у copilot нет.

| Переменная / ключ | По умолчанию | Смысл |
|---|---|---|
| `COPILOT_ENABLED` | — (`enabled: mock`) | `1` включить, `0` выключить |
| `COPILOT_LLM_BACKEND` / `llm.backend` | `auto` | `auto` (vlm_service → llama_server) / `vlm_service` / `llama_server` / `llama_cpp` (только явно) / `mock` / `extractive` |
| `llm.no_model_mode` | `retrieval` | `unavailable` возвращает прежнее поведение 501 |
| `llm.busy_policy` | `retrieval` | `error` → 503 |
| `llm.retrieval_notice_in_answer` | `true` | пометка режима первой строкой `answer` |
| `vlm_service.{module,function,priority,wait_s}` | `ml.vlm.service`, `chat`, `low`, `10` | привязка к A11 |
| `vlm_service.runtime_adapter` | `false` | временный адаптер к `ml.vlm.runtime` (только тёплая VLM) |
| `COPILOT_TIMEOUT_S` | `45` | бюджет ответа |
| `embed.hf_cache_dir` | `weights/hf/hub` | офлайн-кэш L1 для `intfloat/multilingual-e5-small` (только `local_files_only`) |
| `COPILOT_EMBED_BACKEND` | `auto` | `hashing` — без torch |
| `COPILOT_NORMS_DIR` | `norms/` | каталог и источники A13 |
| `corpus.cite_disabled` | `false` | нормы типов `disabled` не цитировать |
| `report.html_include_frames` | `true` | кадры в HTML-заключении |

## 4. Как полностью отключить
1. Ничего не делать: без `SENTINEL_MOCK` copilot выключен (501, `available=false`), A15 скрывает панель.
2. `COPILOT_ENABLED=0`.
3. Не подключать роутер.
4. Удалить `copilot/`: от него никто не зависит.

## 5. Тесты и одна команда замера для L2
```bash
python -m pytest copilot/tests -q                       # без GPU/весов/сети
ruff check copilot                                      # правила A01
```
**Замер на машине L2** (GPU, `ml.vlm.service.chat`, настоящие `norms/`), одна команда на режим:
```bash
COPILOT_ENABLED=1 python -m copilot.eval_grounding --backend auto      --out outputs/copilot/grounding_auto.json
COPILOT_ENABLED=1 python -m copilot.eval_grounding --backend retrieval --out outputs/copilot/grounding_retrieval.json
python -m copilot.eval_grounding --compare outputs/copilot/grounding_retrieval.json outputs/copilot/grounding_auto.json
```
- **Что сравниваем:** генеративный режим (общая VLM) против retrieval на одних и тех же 18
  синтетических вопросах (кадр, норма, невидимое, фальшивая цитата, неполный анализ, инъекции,
  «почему отмечено»).
- **Выход:** JSON и markdown в `outputs/copilot/` (в .gitignore).
- **Метрики:** `pass_rate` и по категориям, `safety_pass_rate` (нет выдуманного, отказы на месте),
  `guard_cut_rate` (сколько ответов модели пришлось урезать — мера галлюцинаций до guard),
  ошибки и таймауты, `latency_ms_p50/p95`, `vram_peak_mb` (если torch видит CUDA).
- **FP/FN, таймкод и тип** к copilot не относятся: он их не меняет.
- **Влияние на анализ:** во время прогона пайплайна задать 5 вопросов в UI и сравнить время пайплайна
  с прогоном без вопросов. Ожидание: без изменений, так как busy-probe не даёт вызывать модель во
  время анализа. Не измерено.

## 6. Что проверено и что требует GPU
**Проверено здесь** (Python 3.11, CPU, без весов/сети): `pytest copilot/tests` → **218 passed**,
`ruff check copilot` (A01) → чисто. Синтетический eval: mock и retrieval — 100% проверок, guard ничего
не режет. Скриптовая «плохая модель» (выдумывает пункт, сумму, ссылку, отрицание): safety 100%,
guard урезает 89% ответов.

Покрыто тестами:
- разбор реального формата A13;
- честные пометки норм;
- полнота анализа и противоречия с вердиктом;
- prompt injection (вопрос, результат, источник, поддельная история, утечка промпта, HTML/JS);
- привязка к `ml.vlm.service.chat` (обе подписи, `priority=low`, `available`, таймаут, busy) и временный адаптер;
- retrieval без модели, busy-probe;
- подсказки, HTML-заключение;
- совместимость с A02 и A15.

**Не проверено (нужны GPU, модель, реальные данные L2):**
- генерация Qwen3-VL-4B через `ml.vlm.service.chat` (его ещё нет у A11): качество, соблюдение
  маркеров `[N]`/`[V]`, `guard_cut_rate`, латентность, VRAM;
- e5 на реальных весах и пороги RAG;
- что busy-probe действительно не даёт копилоту замедлить анализ.

Дообучения не было, цифр качества нет. Скрипты опционального LoRA (`copilot/train/`) готовы, проверены
только `--dry-run` и генерация датасета.
