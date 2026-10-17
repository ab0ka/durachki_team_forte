# A02 — `backend/api/` (FastAPI)

HTTP-слой Kassa Sentinel: API жюри по ТЗ, те же ручки для веб-UI по cookie-сессии, очередь задач с бюджетом времени и watchdog, отдача видео с Range, встроенная резервная страница.

Работает **строго одним процессом** — очередь и тёплые модели живут в памяти.

```bash
pip install -r backend/api/requirements.txt
export SENTINEL_API_KEYS="$(python -m backend.api.make_key --quiet)"   # или несколько через запятую
export SENTINEL_UI_PASSWORD='...'            # опционально: вход в UI по паролю
python -m backend.api --host 127.0.0.1 --port 8000
# то же: uvicorn backend.api.main:app --workers 1 --no-access-log
```

## API из ТЗ (не менялось)

```bash
# 1) загрузить видео -> job_id
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "X-API-Key: $SENTINEL_API_KEYS" \
  -F "file=@X-1.mov"
# {"job_id":"0f3c...","status":"processing"}

# 2) опрашивать статус / забрать результат
curl http://localhost:8000/api/v1/jobs/0f3c... -H "X-API-Key: $SENTINEL_API_KEYS"
# пока идёт анализ:
# {"job_id":"0f3c...","status":"processing","stage":"detect","progress":0.42,"queue_position":null,"elapsed_s":12.3}
# готово:
# {"job_id":"0f3c...","status":"done","verdict":"violation",
#  "violations":[{"type":"...","timecode_sec":22.0,"description":"...",
#                 "evidence_frames":[{"timestamp_sec":22.0,"image_base64":"/9j/..."}],
#                 "type_id":"V01","confidence":0.81,"t_start":20.4,"t_end":25.0,"rules":[...]}],
#  "timeline":[[0.0,0.1],...],"meta":{...,"api":{"queue_wait_s":..,"total_s":..,"frames":[...]}}}
# не удалось завершить:
# {"job_id":"0f3c...","status":"failed","verdict":null,"violations":[],"error":"timeout: ..."}
```

- Без ключа или с неверным ключом — **401** с JSON `{"detail": ...}`. Принимается и `Authorization: Bearer <key>`.
- Обязательные поля — ровно как в ТЗ. Дополнительные: `type_id`, `confidence`, `t_start`, `t_end`, `rules[]`, `timeline`, `meta`, а во время обработки — `stage`, `progress`, `queue_position`, `elapsed_s`.
- Ответ на POST: HTTP 200 и `{"job_id","status":"processing"}`.

| Ручка (`/api/v1/...` с `X-API-Key`) | Назначение |
|---|---|
| `POST /analyze` | multipart `file` (.mov/.mp4/.m4v, ≤ 300 МБ). Принимается и «сырое» тело с `Content-Type: video/*` и `X-Filename` |
| `GET /jobs/{id}` | Статус и результат по схеме ТЗ |
| `GET /jobs` | Последние задачи (история) |
| `GET/HEAD /jobs/{id}/video` | Исходник с **HTTP Range** (206/416). Если в папке задачи есть `preview.mp4`, отдаётся он; `?original=1` — принудительно исходник |
| `GET /jobs/{id}/frames/{i}.jpg` | Кадр-доказательство №`i`. Нумерация сквозная по `violations[*].evidence_frames[*]` (см. `meta.api.frames`) |
| `POST /jobs/{id}/ask` | `{"question": "..."}` → `copilot.router` (A14). Если он недоступен — **501** |
| `GET /catalog` | `norms.load_catalog()` (A13). Если недоступен — 503 |
| `GET /healthz` | Liveness, без ключа |
| `GET /readyz` | **200 только если пайплайн импортирован и прогрет.** Иначе **503** с полем `state`: `loading`, `warming`, `unavailable` (нет `backend.pipeline` или весов) или `warmup_failed` |
| `GET /docs` | Swagger с кнопкой Authorize для X-API-Key |

Коды ошибок загрузки:
- **400** — пустой файл, нет поля `file` или битый multipart.
- **413** — больше 300 МБ.
- **415** — не то расширение, нет сигнатуры ISO-BMFF/QuickTime или **декодер не прочитал ни одного кадра**.
- **503** — на сервере нет декодера (при `video_validation=required`) или переполнен backlog (с заголовком `Retry-After`).
- **507** — нет места на диске.

## Веб-UI: ключ не попадает в бандл, URL и логи (согласовать с A15)

Ручки `/ui/v1/*` — **те же**, что `/api/v1/*`. Доступ по cookie с флагами **HttpOnly; SameSite=Strict; Path=/ui/v1** (`Secure` — при https или за cloudflared). Cookie выдаётся **только после проверки API-ключа или пароля UI**. CORS и same-origin здесь **не** аутентификация, а лишь дополнительная защита от CSRF.

```js
// 1. вход: проверяющий вводит ключ/пароль; значение уходит один раз и в JS не хранится
await fetch("/ui/v1/session", {method: "POST", headers: {"Content-Type": "application/json"},
  body: JSON.stringify({api_key: value})});          // или {password: value}
// -> 200 {"authenticated":true,"expires_at":..., "csrf_token":"..."} + Set-Cookie ks_session (12 ч)
// -> 401 неверно, 429 слишком много попыток (10/мин на IP)

// 2. при загрузке страницы: я залогинен?  GET /ui/v1/session -> 200 {csrf_token} | 401
// 3. POST/DELETE — обязательно заголовок X-CSRF-Token (из ответа шага 1 или 2)
xhr.open("POST", "/ui/v1/analyze"); xhr.setRequestHeader("X-CSRF-Token", csrf); xhr.send(formData);
// 4. GET — только cookie (браузер шлёт её сам на same-origin):
fetch(`/ui/v1/jobs/${id}`)
// 5. ПЛЕЕР: обычный <video src> не добавит X-API-Key, но cookie отправит -> Range работает
<video src="/ui/v1/jobs/ID/video" controls></video>
// 6. выход: DELETE /ui/v1/session
```

- Если задача `failed`, на `/ui/v1/jobs/{id}` дополнительно приходит `partial` — подробности неполного анализа из `partial.json`, чтобы UI мог объяснить, что произошло. **В API жюри этого нет.**
- Rate-limit на IP: вход 10/мин, загрузка 12/мин, вопросы ассистенту (ask) 20/мин, чтение 600/мин. Неудачные попытки API-ключа — 300/мин. Всё настраивается в конфиге.
- Если `frontend/dist` нет, на `/` отдаётся встроенная страница `backend/api/static/index.html`. Это рабочий пример всего потока: вход, загрузка с прогрессом, опрос статуса, кадры и плеер на cookie-эндпоинте видео. Если `frontend/dist` есть, он монтируется на `/` с SPA-fallback: `index.html` с `no-cache`, `/assets/*` с `immutable`.

## Очередь, бюджет времени, честные статусы

- **Отсчёт времени** начинается **с начала запроса загрузки**, поэтому лимит 300 с включает и загрузку, и ожидание в очереди.
- **Очередь и GPU.** Задачи идут по FIFO на `SENTINEL_WORKERS=1` потоков. Одновременно на GPU — **1** задача (`gpu_slots`); семафор держится на всё время `pipeline.run`.
- **Deadline для пайплайна:**
  `deadline_s = min(270 − ожидание, job_cap_s=120, честная доля)`, но не меньше `min_deadline_s=15`.
  - Честная доля — остаток бюджета, поделённый на задачи, ждущие за текущей. Если жюри пришлёт 8 видео разом, каждое получит ≈ 270/8 ≈ 34 с — а не «первое всё, шестое ничего».
- **Backlog.** Больше `max_queue=12` активных задач → **503 + Retry-After**.
- **`done`** — только если `backend.pipeline.run()` вернул результат. Он сериализуется через `contracts.serialize.result_to_api` (fallback — своя сериализация строго по ТЗ), затем проверяется и чинится формат: типы, base64 JPEG, отсутствующий `verdict` выводится из `violations`. Содержание API не придумывает.
  - Если у нарушения нет кадра-доказательства, API вырезает его через cv2 в момент `timecode_sec` и пишет предупреждение в `meta.warnings`.
- **`failed`** — в этих случаях:
  - пайплайн не импортировался (нет весов или модуля);
  - `run()` бросил исключение, вернул `status="failed"` или мусор;
  - сработал **watchdog**: 285 с от загрузки или `deadline_s + 20 с` от старта задачи.
  - `partial.json` по watchdog **не** превращается в `done`; частичный результат попадает только в `partial` для UI.
- **Зависший `run()`.** После срабатывания watchdog `progress()` бросает `JobCancelled` (это **BaseException**), и `run()` сворачивается. Если `run()` висит, не вызывая `progress()`, слот и GPU освобождаются и запускается подменный воркер (не больше `max_zombie_workers`).
- **Рестарт.** Незавершённые задачи из SQLite заново встают в очередь со свежим бюджетом — до `max_attempts=3` раз.
- **Фейковый пайплайн** — **только** при явном `SENTINEL_FAKE_PIPELINE=1` (изолированные тесты, демо UI). В результате `meta.fake_pipeline=true`, в `/readyz` — `"pipeline":"fake"`. **Автоматически он никогда не подставляется.**

Хранение:
- `data/jobs/<job_id>/`: `video.<ext>`, `result-*.json`, `frames/<i>.jpg`, `partial.json` (от пайплайна), `failed_details.json`.
- `data/jobs.sqlite3` в режиме WAL; поля: id, status, stage, progress, created/queued/started/finished, error, result_path, attempts, deadline_s, video_meta.
- Очистка по TTL (`job_ttl_h=72`).

## Конфигурация

Всё лежит в `backend/api/config.yaml`; любой ключ переопределяется переменной `SENTINEL_<KEY>`. Главное:

| env | по умолчанию | |
|---|---|---|
| `SENTINEL_API_KEYS` | — | Ключи через запятую. Иначе берётся `data/api_keys.txt`; если нет и его — ключ генерируется туда (права 0600), в лог попадает только отпечаток |
| `SENTINEL_UI_PASSWORD` | — | Пароль для входа в UI (опционально; API-ключ тоже подходит) |
| `SENTINEL_UI_SECRET` | авто (`data/ui_secret`) | HMAC-секрет сессий UI; смена разлогинивает всех |
| `SENTINEL_DATA_DIR` | `data` | |
| `SENTINEL_WORKERS` / `SENTINEL_GPU_SLOTS` | 1 / 1 | |
| `SENTINEL_BUDGET_S` / `SENTINEL_WATCHDOG_S` / `SENTINEL_JOB_CAP_S` | 270 / 285 / 120 | |
| `SENTINEL_MAX_QUEUE` | 12 | Backlog |
| `SENTINEL_MAX_UPLOAD_MB` | 300 | |
| `SENTINEL_VIDEO_VALIDATION` | `required` | `required` — cv2/PyAV/ffprobe должен декодировать кадр; `best_effort`; `off` |
| `SENTINEL_PIPELINE` | `backend.pipeline` | Модуль (`run`, `warmup`) или `module:function` |
| `SENTINEL_FAKE_PIPELINE` | 0 | Только для тестов и демо |
| `SENTINEL_CORS_ORIGINS` | пусто | Dev-origin'ы (например `http://localhost:5173`). CORS — не аутентификация |
| `SENTINEL_TRUSTED_PROXIES` | `127.0.0.1,::1` | Кому верить в `CF-Connecting-IP` / `X-Forwarded-*` (cloudflared работает локально) |
| `SENTINEL_JOB_TTL_H` | 72 | |
| `SENTINEL_LOG_JSON` / `SENTINEL_LOG_LEVEL` | 1 / INFO | |

Логи — JSON в stderr с `request_id` (он же в заголовке `X-Request-ID`): метод, путь **без query**, статус, мс, IP, job_id. Ключи, пароли, cookie и содержимое видео не пишутся; поля с именами key/token/secret/password маскируются как `***`.

## Тесты (без GPU и весов)

```bash
python -m pytest backend/api/tests -q
```

Всего 42 теста:
- **Авторизация.** 401 без ключа и с неверным ключом на всех ручках. `/ui/v1` закрыт без cookie, даже с same-origin-заголовками. Вход по ключу или паролю ставит HttpOnly/SameSite cookie. Без CSRF и с чужим Origin — 403. Подделанный или просроченный токен отклоняется. Rate-limit входа → 429. Ключей нет в логах.
- **Состояния пайплайна.** Нет модели → `readyz=503`, задача `failed` (а не фейк). Ошибка `warmup` → `readyz=503`. Исключение инференса → `failed`. Таймаут → `failed`, `partial` только в UI, `run()` отменяется через `progress()`. Зависший `run()` не блокирует очередь.
- **Очередь и хранение.** Полный цикл со сверкой со схемой ТЗ. 5 параллельных загрузок. Backlog → 503. Честная доля deadline. Рестарт заново ставит задачи в очередь. Очистка по TTL.
- **Загрузка.** `.txt`, PNG и не-ISO данные → 415. Пустой файл → 400. Сверх лимита → 413, мусор на диске не остаётся. Стриминг больше 3 МБ — размер и sha256 сходятся. **Настоящий MOV** из `cv2.VideoWriter` декодируется. Файл, где валиден только `ftyp`, отклоняется декодером (415). Нет декодера → 503.
- **Медиа и раздача.** Range-запросы: `0-99`, `-10`, `N-`, выход за конец, 416, HEAD. Видео для UI по cookie → 206. Кадры, copilot 501/200, SPA-fallback, request-id, CORS выключен по умолчанию.

Тесты, которым нужен cv2, при его отсутствии пропускаются автоматически (`pytest.importorskip`).

## Модели

Нет: `models.json` = `[]`. Сам API ML не запускает. Декодер — OpenCV (Apache-2.0); PyAV (BSD) и ffprobe — опционально.

## Ограничения / известные TODO
- Воркеры — только потоки. Зависший `run()` нельзя убить: он отменяется кооперативно через `progress()`, а при жёстком зависании подменяется новым потоком (GPU тогда может делиться с «зомби», пока тот не завершится). Режим процессов с kill — TODO.
- `deadline_s` передаётся один раз при старте. Если во время работы первого видео придут новые, его deadline не сократится — поэтому есть потолок `job_cap_s=120`.
- Сессия UI stateless (HMAC): отозвать её можно только TTL или сменой `SENTINEL_UI_SECRET`.

## Для интегратора
1. **Запуск одним процессом**: `python -m backend.api` (A17: служба Windows или CMD в Docker). Задать `SENTINEL_API_KEYS` (или `data/api_keys.txt`), опционально `SENTINEL_UI_PASSWORD`. **`SENTINEL_FAKE_PIPELINE` в проде не ставить.** cloudflared → `http://127.0.0.1:8000`; `SENTINEL_TRUSTED_PROXIES=127.0.0.1,::1` оставить (нужно для IP за туннелем и флага Secure у cookie).
2. **A03 (`backend.pipeline`)**: API вызывает `run(video_path, job_id, progress, deadline_s)` в потоке-воркере, а `warmup()` — один раз в фоне при старте.
   - `video_path` = `data/jobs/<id>/video.mov`.
   - `partial.json` класть в **`Path(video_path).parent`**: это неполный результат только для UI, в `done` он не превратится.
   - `progress(stage, 0..1)` после таймаута может бросить `backend.api.worker.JobCancelled` (BaseException) — `BaseException` не глотать.
   - GPU = 1. API держит семафор на всё время `run()`. Если пайплайн экспортирует свой `GPU_LOCK`/`GPU_SEMAPHORE`/`gpu_lock`, API не оборачивает `run()` (во избежание deadlock). Параллельных GPU-задач внутри `run()` не запускать.
   - `preview.mp4` (H.264) в папке задачи будет отдаваться плееру UI вместо исходного `.mov`.
3. **A01**: если есть `contracts.serialize.result_to_api`, используется он, иначе встроенный сериализатор. После мерджа удалить `backend/api/_contract_fallback.py` (импорт уже идёт с fallback).
4. **A15 (frontend)**: использовать только `/ui/v1/*` с cookie-авторизацией (см. раздел «Веб-UI» выше). Плеер — `<video src="/ui/v1/jobs/ID/video">`. API-ключ нигде не запрашивать сверх формы входа и не хранить — ни в бандле, ни в URL, ни в localStorage. Эталон — `backend/api/static/index.html`. Для Vite dev: `SENTINEL_CORS_ORIGINS=http://localhost:5173` + `credentials: "include"`, а лучше Vite-proxy на `/ui`.
5. **A14 (copilot)**: экспортировать из `copilot/router.py` callable `ask` (или `answer` / `handle_ask` / `answer_question`). API передаёт только те аргументы, которые функция принимает, из набора: `job_id, question, history, result (JSON ТЗ или None), job_dir, video_path`. Вернуть `dict` или `str`. Таймаут — `copilot_timeout_s=60`.
6. **A16 (eval)**: `meta.api.video.filename_normalized` (Х→X) — для сопоставления с разметкой; `meta.api.total_s` / `queue_wait_s` / `deadline_s` — тайминги.
7. Цель — Python 3.11: синтаксиса 3.12+ в коде нет. Проверено на Python 3.13, FastAPI 0.143, Starlette 1.6.
