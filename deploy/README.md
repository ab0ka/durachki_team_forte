# deploy/ — развёртывание Kassa Sentinel (A17)

**Основной путь — Windows-нативный, на рабочей venv L1** (Python 3.11, lock `tools/dataprep/requirements-l1.lock.txt`: torch 2.11.0+cu128, transformers 5.20.1, bitsandbytes 0.50.2, opencv < 5).
- Веса — офлайн, уже лежат в `weights/hf/hub`.
- Docker (`Dockerfile.gpu`, `offline_smoke.sh`, `PLAN_B.md`) — опциональный путь. Он **не собирался** и не синхронизирован с lock L1.

## Команды для L2 (по порядку)
```powershell
powershell -ExecutionPolicy Bypass -File deploy\windows\setup.ps1 [-InstallMissing]   # сверка venv с lock L1, adopt весов
powershell -ExecutionPolicy Bypass -File deploy\windows\offline_smoke.ps1 -Firewall    # от администратора: доказательство локальности
powershell -ExecutionPolicy Bypass -File deploy\windows\run_all.ps1                    # preflight → сервис → READY → URL
.venv\Scripts\python deploy\tunnel_probe.py real --url <URL> --mb 30                   # замер публичного URL
.venv\Scripts\python -m eval.jury_sim --url <URL> --key <ключ> --videos data\videos --n 6   # приёмка A16, замер SLA
.venv\Scripts\python deploy\release.py mark-verified --note "A16 ok" --report <отчёт>  # зафиксировать проверенную версию
deploy\windows\status.ps1   |   deploy\windows\stop.ps1
```
Сбои — `RUNBOOK.md`, последний час — `CHECKLIST_SUBMISSION.md`, память GPU — `VRAM.md`.

## Что делает каждый файл
| Файл | Назначение |
|---|---|
| `config/deploy.yaml` | Все флаги и пороги: lock, обязательные/опциональные/исключённые модели, пороги диска и VRAM, сервис, туннель, netguard. |
| `envcheck.py` | Сверяет установленную venv с lock L1 и требованиями модулей. **Ничего не ставит.** Блокирует расхождение критичных пакетов и отсутствие рантайм-пакетов (fastapi, uvicorn, python-multipart — их нет в lock L1). |
| `merge_requirements.py` | Сводит `*/requirements.txt`, проверяет их против пинов lock L1 и сообщает только о нарушающих ограничениях; запрещённые лицензии. Lock — только resolver'ом (`uv pip compile -c <lock L1>`): пины L1 не двигаются. |
| `weights.py adopt` | **Офлайн-инвентаризация уже скачанного кэша**: commit из `refs/main`, все файлы снапшота резолвятся, полнота шардов `*.index.json`, пустые файлы, `.incomplete`; sha256 LFS из имени blob (`--hash` — пересчитать); лицензия из `README.md` снапшота. Результат → `weights.lock.json`. `verify` проверяет обязательный список из `deploy.yaml`; GGUF исключён шаблоном. `prepare` — отдельный онлайн-шаг докачки. |
| `preflight.py` | Блокирующие проверки до старта: Python, lock, CUDA-сборка (без создания CUDA-контекста), свободная VRAM, обязательные веса, конфиги модулей, диск, пути, наличие ключа (значение не печатается), порт, запрет `SENTINEL_MOCK`/`FAKE_PIPELINE`, ffmpeg/ffprobe/cloudflared. Нет обязательной модели → сервис не стартует. |
| `supervisor.py` | Один экземпляр (OS-блокировка файла), один uvicorn-воркер. Рестарт — только после смерти своего дерева процессов и освобождения порта; чужие процессы не трогает (PID + время старта). Crash-loop → стоп с понятной ошибкой (GPU OOM распознаётся). READY = `/readyz` 200 (warmup A02) + **реальная пробная задача** → только потом cloudflared. Туннель переживает рестарт API (quick URL не меняется). Система не засыпает, дисплей гаснуть может. |
| `windows/*.ps1` | Тонкие обёртки: `_env.ps1` (env из `deploy\.env`, офлайн), `setup`, `run_all`, `start` (прежние параметры), `status`, `stop`, `offline_smoke`, `firewall_python` (временное правило только на python.exe), `monitor`. |
| `offline_worker_smoke.py` | **Реальный worker** (`backend.pipeline.run` → spawn-воркер A03) на локальных весах: HF offline, netguard=block в процессе и воркере. Попытки выхода в сеть считаются по общему логу. Падает на mock/fake, на `no_violation` при неполном анализе и на любой попытке выйти в сеть. `--egress-selftest`: python.exe без netguard должен не достучаться до интернета (под правилом фаервола). Лог для питча — `runs/deploy/offline_proof_*.md`. |
| `healthcheck.py` | `live`; `ready` = `/readyz` + пробная задача + схема ТЗ, отвергает заглушку, `pipeline=fake`, `meta.mock`, `no_violation` при `complete=false`. |
| `tunnel_probe.py`, `sla_bench.py` | Замеры публичного URL и SLA. Цифры — только из них или из A16 `jury_sim`. |
| `release.py` | `mark-verified` (SHA, хэши конфигов и lock-файлов; `.env` — только хэш), `diff`, `rollback` (stash + switch --detach, без reset --hard). |
| `netguard/` | Аудит-хук PEP 578: блокирует или логирует не-loopback connect/DNS и прокси; наследуется дочерними процессами через `PYTHONPATH`. |
| `config/pipeline_vram_measured.yaml` | **Гипотеза, выключена**: оверлей VRAM-оценок A03 под измеренный пик 4.7 ГБ (см. `VRAM.md`). |
| `stub_api.py` | Заглушка API — **только** для тестов deploy и замера туннеля до интеграции. Помечена; `ready` её отвергает. |

## Решения фазы 2 (записаны по ходу)
1. **Источник истины окружения — lock L1.**
   - Свой пин torch 2.6.0/cu124 снят: он противоречил L1. Таргеты lock — `windows-cu128` / `linux-cu128`.
   - `setup.ps1` не пересинхронизирует рабочую venv. Недостающее (fastapi, uvicorn…) ставится только с `-InstallMissing` под `-c lock`.
2. **Обязательные модели** (блокируют readiness): `PekingU/rtdetr_v2_r18vd` (detect closed-set), `google/siglip2-base-patch16-224` (temporal), `Qwen/Qwen3-VL-4B-Instruct` (VLM).
   - VLM понижается до опциональной только явным `SENTINEL_NO_VLM=1` / `SENTINEL_VLM_POLICY=off`.
   - Опциональные: OWLv2 (open_vocab выключен в конфиге A05), D-FINE-medium (скачан, не используется), e5-small (копилот выключен).
   - Исключены: `*GGUF*`.
3. **netguard.** В проде — `log` (аудит, ничего не ломает). `block` — только в offline smoke: новый режим не включается в проде без замера.
4. **Офлайн-доказательство.** Временное правило фаервола только на python.exe (лаунчер venv + базовый интерпретатор, который запускают лаунчер и multiprocessing spawn); снимается в `finally`.
   - Глобально фаервол не меняется. cloudflared — отдельный процесс, публичный транспорт проверяется `tunnel_probe.py`.
5. **Логи и отчёты** — в `runs/deploy/` (в `.gitignore`), а не в `logs/`: `logs/` не игнорируется корневым `.gitignore`.
6. **Туннель после рестарта API не перезапускается**, чтобы quick URL у жюри не сменился.

## Найденные конфликты требований с lock L1 (решают владельцы, не deploy)
`python deploy/merge_requirements.py --root <интеграционное дерево>` на ветке `claude/integration-cloud`:

| Пакет | lock L1 | Требование модуля | Владелец |
|---|---|---|---|
| transformers | 5.20.1 | `>=4.57.1,<5` | `ml/vlm/requirements.txt` (A11) |
| pillow | 13.0.0 | `>=10.0,<13` | `ml/fusion/requirements.txt` (A12) |
| pyyaml | 6.0.3 | `==6.0.1` | `eval/requirements.txt` (A16) |

Свой `huggingface_hub<2` (против 2.2.0 в L1) исправлен.

## Что реально проверено (облако: Linux, без GPU, без данных)
- `pytest deploy/tests` — **64 passed, 1 skipped** (skip — парсер PowerShell, если нет pwsh; с pwsh 7.4.6 — 65 passed). Покрыто:
  - envcheck и merge против lock на синтетических фикстурах. Отдельно вручную прогнан `merge_requirements.py` по дереву ветки `claude/integration-cloud` с настоящим lock L1 — отсюда таблица конфликтов выше;
  - adopt на раскладке кэша huggingface_hub (symlink и без symlink, недостающий шард, `.incomplete`, лицензия карточки, порча sha256);
  - preflight на фейковых пробах (ключ не печатается);
  - супервизор на реальных подпроцессах: полный цикл ready → туннель → URL → status → второй экземпляр отклонён → stop; crash-loop с OOM; чужой процесс на порту не тронут; путь без psutil;
  - offline smoke: попытка выйти в сеть из дочернего процесса поймана; mock и неполный анализ отклонены;
  - release (stash + detach на временном git-репо);
  - healthcheck против `pipeline=fake`.
- Все `.ps1` разбираются парсером PowerShell 7.4.6 (Linux); ASCII-only, чтобы Windows PowerShell 5.1 их не испортил.
- **НЕ проверено**:
  - на Windows `.ps1` не запускались;
  - фаервол-правило, GPU, реальные веса и пайплайн не проверены;
  - Cloudflare Tunnel, SLA и VRAM не проверены.
  - Ни одной цифры скорости или памяти отсюда не приводим — их дают команды выше на ноутбуке L2.

## Для интегратора
- **Структура.** Ветка добавляет только `deploy/`. Канонический контракт есть в `contracts/`; deploy его не импортирует, fallback-копии нет.
- **A01.** Добавить `logs/` в корневой `.gitignore` (на всякий случай). deploy пишет в `runs/deploy/`.
- **A02.** Запуск — `uvicorn backend.api.main:app --workers 1` из супервизора.
  - `/readyz` используется как часть готовности; `pipeline=fake` блокирует.
  - Ключ для пробной задачи — env или `data/api_keys.txt` (A02 генерирует его сам).
- **A03.**
  - `offline_worker_smoke` смотрит `meta.complete`, `meta.coverage` (если появится), `meta.mock`, `meta.stages.*.origin/vram_peak_mb`, `meta.vram_events`.
  - Netguard передаётся spawn-воркеру через env и `PYTHONPATH`.
  - Риск VRAM-оценок — `VRAM.md`.
- **A11/A12/A16.** Конфликты с lock L1 — таблица выше.
- **A14.** На время мероприятия `COPILOT_ENABLED=0`. Если включать — только CPU (`VRAM.md`).
- **TODO (optional Docker):** установка в `Dockerfile.gpu` из lock L1 (cu128), compose, `Dockerfile.cpu`.

## Модели и лицензии
Своих моделей deploy не использует (`models.json` = `[]`). Лицензии весов решаются по конкретному снапшоту: `weights.py adopt` → `weights.lock.json` → `weights.py licenses` → `LICENSES.md`.

Зависимости deploy (`requirements.txt`): huggingface_hub, packaging, PyYAML, psutil — все есть в lock L1. Остальное — stdlib.
