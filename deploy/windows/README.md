# Windows-нативный запуск (основной путь)

1. Драйвер NVIDIA: `nvidia-smi` должен показать RTX 4060 и драйвер с поддержкой CUDA 12.4+.
2. `powershell -ExecutionPolicy Bypass -File deploy\windows\setup.ps1` — uv, Python 3.11, lock, установка по хэшам, `gpu_probe.py`, `.env` со случайным ключом, веса (`prepare` + `verify`).
3. `deploy\windows\offline_smoke.ps1` — настоящий worker без сети → `logs\offline_worker_smoke.json`.
4. `deploy\windows\start.ps1` — API + туннель. Перезапуск при падении и зависании (цикл супервизора), один worker.
5. В соседнем окне: `deploy\windows\monitor.ps1` — `logs\monitor.csv` (CPU/RAM/GPU/VRAM раз в 30 с).

**Электропитание.** `start.ps1` вызывает `SetThreadExecutionState`: пока скрипт работает, система не засыпает, а дисплей гаснуть может — это не мешает. Глобальный план электропитания скрипты не меняют. Закрытие крышки подчиняется плану: держите крышку открытой или вручную выставьте «Действие при закрытии крышки → Ничего не делать (от сети)». Ноутбук — от сети.

**Проверка GPU** на любой новой машине: сначала `.venv\Scripts\python deploy\gpu_probe.py`.
