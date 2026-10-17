# deploy/ — развёртывание Kassa Sentinel (A17)

**Основной путь — Windows-нативный** (прод: ноутбук RTX 4060 Laptop 8 GB, Windows 11).
Docker (`Dockerfile.gpu`, `offline_smoke.sh`) — **опциональный**, для GPU-VM из Плана Б; здесь он **не собирался** (в среде агента нет docker daemon).

## Шаги на ноутбуке
```powershell
powershell -ExecutionPolicy Bypass -File deploy\windows\setup.ps1          # один раз, нужна сеть
powershell -ExecutionPolicy Bypass -File deploy\windows\offline_smoke.ps1  # доказательство локальности
powershell -ExecutionPolicy Bypass -File deploy\windows\start.ps1          # прод: API + туннель
.venv\Scripts\python deploy\tunnel_probe.py real --url <PUBLIC_URL> --mb 30 # замер публичного URL
.venv\Scripts\python deploy\sla_bench.py --url <URL> --videos <5 реальных .mov>  # замер SLA 300 с
```

| Файл | Что делает |
|---|---|
| `merge_requirements.py` | Сводит `*/requirements.txt` + `platform.in` (torch 2.6.0/cu124). **Версии не выбирает**: несовместимые ограничения → exit 1 с источниками, запрещённые лицензии (ultralytics…) → exit 2. Lock — только настоящим resolver'ом: `--lock windows-cu124` → `uv pip compile` (платформа, Python 3.11, хэши) → `deploy/locks/*.lock.txt`. Решения по конфликтам — в `deploy/constraints.txt` (по согласованию с A01/владельцами). |
| `weights.py` | `prepare` — **отдельный онлайн-шаг**: commit sha ревизии, список файлов, sha256 каждого файла (LFS sha256 / git blob id с хаба), лицензия **этой ревизии** из метаданных хаба → `deploy/weights.lock.json` (в git; весов там нет). `verify` — только файловая система, без сети. `licenses` → `LICENSES.md`. |
| `netguard/` | Запрет исходящей сети **внутри Python-процесса** (аудит-хук PEP 578): connect/DNS к не-loopback и любые прокси → `PermissionError` + JSONL-лог. Firewall ОС не меняется. |
| `offline_worker_smoke.py` | **Реальный worker** (`backend.pipeline.run`) на локальных весах, HF offline + netguard=block; отказ при `SENTINEL_MOCK=1`; FAIL при любой попытке выйти в сеть, даже «проглоченной» пайплайном. Проверяет worker, а не публичный HTTP. |
| `healthcheck.py` | `live` — процесс отвечает; `ready` — **реальный пробный inference** через API (401 без ключа, загрузка, done, схема ТЗ, не заглушка). `start.ps1` открывает туннель только после `ready`. |
| `tunnel_probe.py` | Замер публичного URL: лестница размеров тела, до какой паузы держится ответ, Range, Set-Cookie, загрузка MOV ~30 МБ, время до результата. Ничего не обещаем заранее. |
| `sla_bench.py` | K видео одновременно, время до результата у каждого, max/медиана против 300 с. |
| `gpu_probe.py` | На конкретной машине: nvidia-smi, torch CUDA build, VRAM, fp16-matmul, ORT providers. На другой машине — запускать первым. |
| `stub_api.py` | Заглушка API (stdlib) для тестов и замера туннеля до интеграции; помечена `X-Sentinel-Stub`, `ready` её отвергает. |
| `windows/*.ps1` | `setup`, `start` (супервизор с перезапуском, один worker, система не засыпает — дисплей может гаснуть), `offline_smoke`, `monitor` (CSV раз в 30 с). |

## Как мы доказываем, что облака нет
1. Рантайм офлайн: `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1`, веса заранее подготовлены `weights.py prepare` и сверены `verify` (sha256).
2. `offline_worker_smoke` гоняет **настоящий** пайплайн с настоящими весами при запрете сети в процессе; отчёт `logs/offline_worker_smoke.json` (число попыток выхода в сеть, VRAM, время).
3. Сильная гарантия (ядро): `deploy/offline_smoke.sh` — тот же тест в `docker run --network none` (План Б / Linux).
4. Публичный API проверяется **отдельно** (`tunnel_probe.py`): контейнер или процесс без сети не может быть публичным API.

## Что реально проверено в этой сессии (Linux, без GPU, без данных)
- `pytest deploy/tests` — 38 passed: merge/конфликты/lock-вызов (с фейковым uv), weights prepare/verify на фейковом хабе (пин коммита, sha256, порча файла, красная лицензия, URL-модели), netguard (блок, лог, allowlist, **прокси на loopback**), offline smoke с фейковым пайплайном (успех / проглоченная попытка выйти в сеть / отказ от mock), healthcheck/tunnel_probe/sla_bench против stub, ASCII в `.ps1`.
- `uv pip compile` под `x86_64-pc-windows-msvc` и `x86_64-manylinux_2_28` (PyPI-часть) — работает; **lock с torch здесь не построен**: download.pytorch.org недоступен из среды агента → `setup.ps1` строит его на ноутбуке (закоммитьте `deploy/locks/windows-cu124.lock.txt`).
- **НЕ проверено**: PowerShell-скрипты (нет pwsh), Docker-сборка (нет daemon), реальный GPU, реальные веса, Cloudflare Tunnel, SLA. Никаких измерений не приводим — их дают `gpu_probe`, `offline_smoke`, `tunnel_probe`, `sla_bench` на ноутбуке.

## Cloudflare Tunnel
- Quick: `start.ps1` (без аккаунта, URL меняется при перезапуске, пишется в `logs/public_url.txt`).
- Named (стабильный URL): Zero Trust → Networks → Tunnels → Create → токен в `deploy/.env` (`TUNNEL_TOKEN`), public hostname → `http://localhost:8000`; запуск `start.ps1 -Named`.
- Если площадка режет UDP/QUIC: `CLOUDFLARED_PROTOCOL=http2`.
- Лимиты тела запроса, таймауты ответа, Range и cookies — **измерить** `tunnel_probe.py tunnel` (против `stub_api.py` за тем же туннелем) и `tunnel_probe.py real`, результат приложить к сдаче.

## Для интегратора
- Канонический контракт уже есть в ветке (`contracts/`), deploy его не импортирует — fallback-копия не нужна.
- `backend.api.main:app` (A02): нужен `GET /healthz` (<500 = жив); `POST /api/v1/analyze` должен отвечать сразу после сохранения файла (очередь), иначе прокси с таймаутом ответа оборвёт запрос.
- `backend.pipeline.run/warmup` (A03): вызывает `offline_worker_smoke.py`. Желательно класть в `result.meta["warnings"]` пропущенные из-за весов стадии — smoke их покажет.
- Модули с весами: грузить по repo id из кэша HF (`HF_HUB_CACHE`) — `prepare` пишет `refs/<rev>`, `from_pretrained(repo)` работает офлайн. Для URL-моделей — путь из `weights/MANIFEST.json`. В `models.json` указывайте `revision` (коммит) и `files` (для GGUF/ONNX обязательно), для URL — `sha256`.
- После мерджа всех веток: `python deploy/merge_requirements.py` → решить конфликты с владельцами → `setup.ps1 -Relock` → закоммитить lock; `python deploy/weights.py licenses` → `LICENSES.md`.
- TODO (optional Docker): `docker-compose.yml`, установка в `Dockerfile.gpu` из `deploy/locks/linux-cu124.lock.txt` вместо `requirements.in`, `Dockerfile.cpu`, `PLAN_B.md` (GPU-VM: сначала `gpu_probe.py`).

## Модели и лицензии
deploy/ своих моделей не использует (`models.json` = `[]`). Сводная таблица — `LICENSES.md` (генерируется).
Зависимости deploy: `requirements.txt` (huggingface_hub, packaging, PyYAML); остальные скрипты — stdlib.
