# Sentinel Copilot — подключение (A14 → интегратор L2)

Copilot — **опциональный** модуль. Он ничего не меняет в анализе видео и по умолчанию выключен в
проде. Подключение — один `include_router` в API (A02). Если его не подключить, удалить или
выключить, остальная система работает как прежде: никто из других модулей copilot не импортирует.

| Что | Поведение |
|---|---|
| Импорт `copilot`, `copilot.router` | без весов, без torch/llama_cpp/transformers, без сети, без чтения корпуса |
| Copilot выключен (по умолчанию) | `POST …/ask` → **501** `error="disabled"`, `answer=""` |
| Включён, но локальной модели нет | **501** `error="unavailable"`, `answer=""`; облачных API нет, автоскачивания нет |
| Ошибка модели / таймаут / GPU занят | 500 `llm_error` / 504 `timeout` / 503 `busy`, JSON-тело; другие эндпоинты API не затронуты |
| Неподтверждённые цитаты, суммы, таймкоды, домыслы | вырезаются до выдачи (и в SSE); если не осталось ничего — «По имеющимся данным это установить нельзя…» |

## 0. Состояние соседних веток (проверено чтением кода, 2026-10-17)
- **A02** (`claude/laughing-faraday-xe89m9`, `backend/api/main.py::_mount_copilot`) уже вызывает
  `make_router(get_job=job_for_copilot, prefix="/api/v1"|"/ui/v1", dependencies=[...])`. Сигнатура
  совпадает, тест `test_a02_style_double_mount_api_and_ui` воспроизводит двойной монтаж. Провайдер A02
  возвращает dict ТЗ, `{"status": "processing"}`, failed-dict или `None`, всё это поддержано.
  Маршрута `/jobs/{id}/report` у A02 нет, конфликтов путей нет. **Менять у A02 ничего не нужно.**
- **A15** (`claude/busy-pasteur-y15o46`, `frontend/src/api/client.ts::ask`) шлёт
  `{question, history: [{role, text}]}` и читает `answer` + `sources` (строки), при 501 скрывает панель.
  Copilot принимает `text` как синоним `content` и отдаёт `sources` (подписи источников) рядом с
  `citations`. Тест `test_a15_request_shape_history_with_text_and_sources` это проверяет.
  **Менять у A15 ничего не нужно.**

## 1. Импорты и подключение роутера (A02)

```python
# backend/api/... (файл A02) — подключение ленивое и необязательное
from fastapi import Depends

def _include_copilot(app, get_result, require_api_key) -> None:
    try:
        from copilot.router import make_router
    except Exception:            # нет папки/зависимостей — API работает без copilot
        return
    app.include_router(
        make_router(get_job=get_result),            # get_result(job_id) -> dict | AnalysisResult | JSON-str | None (может быть async)
        dependencies=[Depends(require_api_key)],    # тот же X-API-Key, что у /api/v1/analyze и /api/v1/jobs/{id}
    )
```

Альтернатива без замыкания: модульный `copilot.router.router` + `copilot.router.set_job_provider(get_result)`.
Пока провайдер не задан, роутер отвечает 503.

Публичные функции (Python):
```python
from copilot import answer, report, status, warmup, release   # всё лениво, исключений не бросает
from copilot.router import make_router, set_job_provider, router
from copilot.service import set_gpu_lock, set_service, CopilotService
```

### Требования к авторизации (делает A02, роутер ключей не знает)
- Все три эндпоинта copilot подключать **только** с той же dependency проверки ключа/сессии, что и
  `GET /api/v1/jobs/{job_id}`. Без ключа → 401 (это проверено тестом с dependency).
- `get_job(job_id)` обязан возвращать результат именно этой задачи и только если вызывающий имеет к
  ней доступ (та же логика, что для `GET /api/v1/jobs/{id}`). Нет задачи или нет доступа → `None` (404).
- Copilot не хранит историю диалога: её присылает клиент в теле запроса, и она обрезается до 8 реплик.

### Требования к общей GPU-очереди (A03)
- По умолчанию copilot выключен и VRAM не занимает. Включать на ноутбуке 8 ГБ — только осознанно.
- Если включён на той же GPU:
  - `copilot.service.set_gpu_lock(lock)` — общий замок или очередь GPU. Нужен объект с
    `acquire(timeout=...)` и `release()`, подойдёт `threading.Lock/Semaphore`. Copilot ждёт его не
    дольше своего таймаута, иначе ответ `busy` (503). Анализ видео copilot не блокирует.
  - `copilot.release()` перед стадиями VLM/детектора выгружает LLM и освобождает VRAM. После простоя
    `llm.idle_unload_s` (300 с) модель выгружается сама.
  - `llm.gpu_layers: auto` отдаёт GPU столько слоёв, чтобы осталось `llm.vram_reserve_mb` (1500 МБ).
    `COPILOT_GPU_LAYERS=0` держит LLM только на CPU (рекомендуется при работающей VLM).
- `copilot.warmup()` в `backend.pipeline.warmup()` не нужен. Если его вызвать, он вернёт `False`
  при недоступности и не бросит исключение.

## 2. Формат входа и выхода (синтетический пример)

Вход `get_job` — результат по схеме API (`contracts.serialize.result_to_api`) или `AnalysisResult`:
```json
{"job_id": "demo-1", "status": "done", "verdict": "violation",
 "violations": [{"type": "Использование мобильного телефона при работе с ценностями", "type_id": "V05",
                 "timecode_sec": 22.0, "t_start": 20.0, "t_end": 25.0, "confidence": 0.81,
                 "description": "Кассир держит телефон у уха, на столе видны наличные",
                 "evidence_frames": [{"timestamp_sec": 22.0, "image_base64": "<JPEG base64>"}],
                 "rules": [{"document": "...", "clause": "п. ...", "url": "...", "quote": "..."}]}]}
```
`image_base64` в LLM не передаётся. `rules` необязательны и перепроверяются по каталогу.

`POST /api/v1/jobs/demo-1/ask` с `X-API-Key`, тело
`{"question": "Почему это нарушение?", "history": [{"role": "user"|"assistant", "content": "..."}]}`
(вопрос 1..2000 символов, история ≤ 40 элементов, вместо `content` можно писать `text`) → **200**.
Ниже реальный вывод в mock-режиме на тестовой фикстуре норм:
```json
{
  "answer": "Основания (только наблюдаемые факты):\n- (№1, 00:22) Использование мобильного телефона при работе с ценностями, уверенность 0.81. Что видно: Кассир держит телефон у уха, на столе видны наличные. Норма: [1].\nЭто признаки возможного нарушения, решение принимает проверяющий.",
  "citations": [{"document": "ФИКСТУРА: Тестовый регламент кассы", "clause": "п. 9",
                 "url": "https://example.invalid/fixture/reglament#p9", "ref": 1,
                 "quote": "Пользоваться мобильным телефоном при работе с ценностями запрещено.", "verified": false}],
  "sources": ["[1] ФИКСТУРА: Тестовый регламент кассы, п. 9 — https://example.invalid/fixture/reglament#p9"],
  "grounded": true, "refusal": false, "intent": "why", "backend": "mock", "job_id": "demo-1",
  "removed_citations": [], "warnings": ["norms/catalog.yaml не найден — используется ТЕСТОВАЯ фикстура (не нормы РК)"],
  "latency_ms": 61, "disclaimer": "Подсказка для проверяющего, не юридическое заключение.", "error": null
}
```
- Обязательные поля ответа: `answer`, `citations[{document, clause, url}]`, `grounded`. Остальные поля
  дополнительные. `sources` — те же источники готовыми строками для UI.
- Сноска `[k]` в `answer` соответствует `citations[i].ref == k`.
- `grounded=false` означает, что из сырого ответа модели что-то вырезано или ответ ни на что не опирается.
- Нехватка данных: `answer` начинается с «По имеющимся данным это установить нельзя…», `refusal=true`.

Ошибка (copilot выключен) → **501**:
```json
{"detail": {"answer": "", "citations": [], "sources": [], "grounded": false, "refusal": false, "error": "disabled",
            "message": "Ассистент отключён в этой установке (включается COPILOT_ENABLED=1).",
            "job_id": "demo-1", "disclaimer": "Подсказка для проверяющего, не юридическое заключение.", "warnings": []}}
```
Коды `error`:

| `error` | HTTP |
|---|---|
| `disabled`, `unavailable` | 501 |
| `not_ready` (задача ещё считается) | 409 |
| `empty_question` | 422 |
| `busy` | 503 |
| `timeout` | 504 |
| `llm_error`, `internal_error` | 500 |

Тело ошибки всегда `{"detail": {...}}`. 404 означает, что задачи нет. 503 с
`detail.code="job_provider_error"` означает, что упал `get_job`.

SSE: `POST …/ask?stream=1` (или `Accept: text/event-stream`). В событиях `token` приходят
**только проверенные** предложения, в конце приходит `done` с полным ответом (как выше) или
`error`. Ошибка до первого токена отдаётся обычным HTTP-кодом, а не 200.
```
event: token
data: {"text": "Таймкоды возможных нарушений:"}

event: token
data: {"text": "\n- (№1, 00:22) Использование мобильного телефона при работе с ценностями (интервал 00:20–00:25)."}

event: done
data: {"answer": "...", "citations": [], "grounded": true, ...}
```

Остальные эндпоинты:
- `GET /api/v1/copilot/status` → `{"enabled", "available", "backend", "mock", "reason": null|"disabled"|"unavailable", "message", "llm", "warnings", "disclaimer"}`.
- `GET /api/v1/jobs/{job_id}/report` → `text/markdown`, заключение для проверяющего. Работает без LLM, даже когда copilot выключен.

**A15 (UI):** показывать панель, только если `status.available == true`. По `done` заменять
накопленный текст на `done.answer` и показывать `citations` и `disclaimer`.

## 3. Зависимости, переменные окружения, пути весов

`copilot/requirements.txt`. Обязательные лёгкие пакеты: `numpy`, `PyYAML`, `pydantic`, `fastapi`.
Опциональные:
- `llama-cpp-python` (MIT) — локальная LLM. Для GPU собирать с `CMAKE_ARGS="-DGGML_CUDA=on"`.
- `sentence-transformers` (Apache-2.0) — эмбеддинги e5. Без них работает встроенный hashing-эмбеддер.

| Переменная | По умолчанию | Смысл |
|---|---|---|
| `COPILOT_ENABLED` | не задана → `enabled: mock` (вкл. только при `SENTINEL_MOCK=1`) | `1` включить, `0` выключить |
| `COPILOT_MODEL_PATH` | кандидаты из `config.yaml` | путь к GGUF |
| `COPILOT_SERVER_URL` | — | локальный `llama-server` (`http://127.0.0.1:8081`); публичные адреса отклоняются |
| `COPILOT_GPU_LAYERS` | `auto` | `0` CPU · `-1` всё на GPU · `N` частично |
| `COPILOT_TIMEOUT_S` | `45` | бюджет одного ответа |
| `COPILOT_IDLE_UNLOAD_S` | `300` | выгрузка модели после простоя |
| `COPILOT_LORA_PATH` | — | GGUF-адаптер (не обучался) |
| `COPILOT_EMBED_BACKEND` | `auto` | `auto` / `e5` / `hashing` |
| `COPILOT_EMBED_MODEL` | `intfloat/multilingual-e5-small` | HF-id (только из локального кэша) или папка с весами |
| `COPILOT_NORMS_DIR` | `norms/` | папка с `catalog.yaml` и `sources/*.md` (A13) |
| `COPILOT_CACHE_DIR` | `cache/copilot` | кэш эмбеддингов корпуса |
| `COPILOT_CONFIG` | — | дополнительный YAML поверх `copilot/config.yaml` |
| `SENTINEL_MOCK` | — | `1` → mock-ответы и тестовая фикстура норм |
| `HF_HUB_OFFLINE` | рекомендуется `1` | офлайн-режим HF |

Пути весов (все в `.gitignore`), первый существующий файл побеждает:
1. `weights/copilot/sentinel-copilot-q4_k_m.gguf` — зарезервировано, сейчас не существует;
2. `weights/copilot/Qwen3-4B-Instruct-2507-Q4_K_M.gguf` — рекомендуется, Apache-2.0, ~2.5 ГБ;
3. `weights/copilot/Qwen2.5-7B-Instruct-Q4_K_M.gguf` — Apache-2.0, ~4.7 ГБ.

e5 берётся из локального кэша HF (`snapshot_download(local_files_only=True)`) или из папки
`COPILOT_EMBED_MODEL`. Модели нигде не скачиваются автоматически: веса один раз кладёт A17.

## 4. Как полностью отключить copilot
1. **Ничего не делать.** В проде без `SENTINEL_MOCK` copilot выключен (`enabled: mock`): `ask` даёт
   501, `status.available=false`, VRAM не занимается, модель не грузится.
2. **Явно:** `COPILOT_ENABLED=0`. Работает даже при `SENTINEL_MOCK=1`.
3. **Не подключать роутер.** Убрать `include_router`. Copilot нигде больше не импортируется.
4. **Удалить папку `copilot/`.** На пайплайн это не влияет: ни один модуль `ml/`, `backend/pipeline`
   или `contracts/` от copilot не зависит. Ленивый `try/except` в A02 это переживёт.

## 5. Тесты
```bash
pip install -r copilot/requirements.txt
python -m pytest copilot/tests -q                                    # всё, ~4 с, без GPU/весов/сети
python -m pytest copilot -m "not gpu and not slow" -q                # как в CI A01 (module copilot)
ruff check copilot                                                   # правила ruff из pyproject A01 (E4,E7,E9,F,B)
python -m pytest copilot/tests/test_integration_readiness.py -q      # только готовность к интеграции
python -m pytest copilot/tests/test_safety.py -q                     # локальность / нет облака / нет тяжёлых импортов
```

## 6. Что реально проверено и что требует локального GPU

**Проверено в этой ветке** (Python 3.11, CPU, без весов и сети): `pytest copilot/tests` → **104 passed**.
Повторено как в CI A01 (`.github/workflows/ci.yml`, job `module copilot`): чистый venv, зависимости
через `scripts/ci_requirements.py copilot`, `SENTINEL_MOCK=1`, `HF_HUB_OFFLINE=1`, pytest-настройки
A01 (`--import-mode=importlib`). Результат: 104 passed, `ruff check copilot` → All checks passed,
`scripts/check_licenses.py` на этом дереве → OK.
- Импорт и старт приложения с роутером без весов. Subprocess с `COPILOT_ENABLED=1` без GGUF:
  `status.available=false`, `answer.error="unavailable"`, `warmup()==False`. Тяжёлые модули не
  импортируются, сокеты не открываются.
- Ошибка модели, таймаут и внутренняя ошибка дают JSON 500/504. Соседние эндпоинты (имитация
  `/api/v1/analyze`) продолжают отвечать. `warmup`/`release`/`status` не бросают исключений. Занятая
  модель даёт `busy` за время дедлайна, а не бесконечное ожидание.
- Нет автоскачивания: e5 только с `local_files_only=True`, иначе hashing. llama.cpp только
  `Llama(model_path=…)`, без `from_pretrained`. Сеть заблокирована в тестах.
- Не доходят до пользователя ни в обычном ответе, ни в SSE: выдуманные `[Nk]`, «п. 999», пункты из
  корпуса, не переданные модели, названия актов, URL, таймкоды, суммы, предложения без опоры на
  контекст. Нормы из `violations[].rules`, которых нет в каталоге, не цитируются.
- Нехватка данных даёт «По имеющимся данным это установить нельзя…». Покрыто: момент без событий,
  нет норм, невидимое (сумма, личность), упавшая задача, пустой вывод модели.
- Router с тестовым провайдером: dict, `AnalysisResult`, JSON-строка, async-провайдер, 404, 409,
  422, 503 (упал провайдер), 401 через dependency, SSE. Битые числовые поля не роняют сервис.
- Совместимость с текущим кодом соседей: двойной монтаж как у A02 (`/api/v1` + `/ui/v1` с разной
  авторизацией) и форма запроса/ответа A15 (`history[{role, text}]`, `sources`). Проверено тестами
  на синтетике, не запуском их кода.
- Бэкенд llama.cpp проверен только на фейковом модуле `llama_cpp`: ленивая загрузка, выгрузка,
  дедлайн, частичный offload (расчёт).

**Не проверено, нужен локальный GPU, веса или реальные данные:**
- реальный инференс GGUF (Qwen3-4B / Qwen2.5-7B) через llama-cpp-python или llama-server: скорость,
  потребление VRAM рядом с VLM, качество русских ответов;
- соблюдение формата `[Nk]`/`[Vk]` реальной моделью и доля ответов, которые guard урежет до «нет данных»;
- эмбеддинги e5 на реальных весах (сейчас тесты идут на hashing) и пороги RAG на настоящем
  `norms/catalog.yaml` A13;
- сборка llama-cpp-python с CUDA под Windows 11 / WSL2 (A17).

Дообучения (LoRA) **не было**, измерений качества и скорости **нет**.
