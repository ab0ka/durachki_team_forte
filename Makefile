# Kassa Sentinel — Makefile (владелец A01). На Windows без make используйте команды из README / deploy/.
PY ?= python
HOST ?= 0.0.0.0
PORT ?= 8000
# Точка входа API (A02) и команды eval/весов (A16/A17) — переопределяются: make run APP=backend.api.app:app
APP ?= backend.api.main:app
DATA_DIR ?= data/videos
EVAL_CMD ?= $(PY) -m eval
WEIGHTS_CMD ?= $(PY) deploy/download_weights.py
MODULE_REQS := $(wildcard backend/*/requirements.txt ml/*/requirements.txt norms/requirements.txt \
                 copilot/requirements.txt eval/requirements.txt deploy/requirements.txt)

.PHONY: help setup setup-dev test test-fast test-gpu e2e e2e-full run eval weights lint format synth licenses ownership \
        validate clean

help:
	@echo "setup      — все зависимости (dev + requirements.txt всех модулей)"
	@echo "setup-dev  — только лёгкие (контракт, тест-кит, pytest, ruff)"
	@echo "test       — все тесты без GPU;  test-fast — без gpu/slow;  test-gpu — только gpu (без глобального SENTINEL_MOCK)"
	@echo "e2e        — сквозной mock-пайплайн на синтетическом видео + проверка схемы ответа жюри"
	@echo "e2e-full   — полная сборка: backend.pipeline.run + сериализатор жюри (явный mock -> done; без mock -> failed)"
	@echo "run        — API (uvicorn $(APP)) на $(HOST):$(PORT)"
	@echo "eval       — оценка на $(DATA_DIR) ($(EVAL_CMD));  weights — скачать веса один раз ($(WEIGHTS_CMD))"
	@echo "lint       — ruff;  licenses — сторож лицензий/облачных API/файлов;  ownership — чужие файлы в ветке"
	@echo "synth      — синтетические ролики всех сценариев в data/synth/"

setup-dev:
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e ".[dev]"

setup: setup-dev
	@for f in $(MODULE_REQS); do echo "== $$f"; $(PY) -m pip install -r $$f || exit 1; done

# SENTINEL_MOCK на всю сессию не ставится (как в CI): тест, которому нужен mock, включает его сам (monkeypatch).
test:
	$(PY) -m pytest -m "not gpu"

test-fast:
	$(PY) -m pytest -m "not gpu and not slow" -q

test-gpu:
	$(PY) -m pytest -m gpu

e2e:
	SENTINEL_MOCK=1 $(PY) -m testkit.e2e_stub

# Свежие каталоги кэша/данных оркестратора (A03): повторный прогон иначе отвечает из кэша.
e2e-full:
	t=$$(mktemp -d) && \
	SENTINEL_MOCK=1 SENTINEL_CACHE_DIR=$$t/mock/cache SENTINEL_DATA_DIR=$$t/mock/data $(PY) -m testkit.e2e_stub \
		--require-full --expect-status done --scenarios clean,V01,V03,V07 --duration 8 --fps 30 && \
	env -u SENTINEL_MOCK SENTINEL_CACHE_DIR=$$t/prod/cache SENTINEL_DATA_DIR=$$t/prod/data $(PY) -m testkit.e2e_stub \
		--require-full --expect-status failed --scenarios clean --duration 8 --fps 30

run:
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $(PY) -m uvicorn $(APP) --host $(HOST) --port $(PORT)

eval:
	HF_HUB_OFFLINE=1 $(EVAL_CMD) --data $(DATA_DIR)

weights:
	$(WEIGHTS_CMD)

lint:
	$(PY) -m ruff check .

format:
	$(PY) -m ruff format contracts testkit scripts

synth:
	$(PY) -m testkit.synth_video --out data/synth --all

licenses:
	$(PY) scripts/check_licenses.py

ownership:
	$(PY) scripts/check_ownership.py --base origin/main

validate:
	$(PY) -m contracts.validate --jury contracts/examples/job_done_tz.json

clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} + ; rm -rf .pytest_cache .ruff_cache
