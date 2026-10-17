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

.PHONY: help setup setup-dev test test-fast test-gpu e2e run eval weights lint format synth licenses ownership validate clean

help:
	@echo "setup      — все зависимости (dev + requirements.txt всех модулей)"
	@echo "setup-dev  — только лёгкие (контракт, тест-кит, pytest, ruff)"
	@echo "test       — все тесты без GPU;  test-fast — без gpu/slow;  test-gpu — только gpu"
	@echo "e2e        — сквозной mock-пайплайн на синтетическом видео + проверка схемы ответа жюри"
	@echo "run        — API (uvicorn $(APP)) на $(HOST):$(PORT)"
	@echo "eval       — оценка на $(DATA_DIR) ($(EVAL_CMD));  weights — скачать веса один раз ($(WEIGHTS_CMD))"
	@echo "lint       — ruff;  licenses — сторож лицензий/облачных API/файлов;  ownership — чужие файлы в ветке"
	@echo "synth      — синтетические ролики всех сценариев в data/synth/"

setup-dev:
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e ".[dev]"

setup: setup-dev
	@for f in $(MODULE_REQS); do echo "== $$f"; $(PY) -m pip install -r $$f || exit 1; done

test:
	SENTINEL_MOCK=1 $(PY) -m pytest -m "not gpu"

test-fast:
	SENTINEL_MOCK=1 $(PY) -m pytest -m "not gpu and not slow" -q

test-gpu:
	$(PY) -m pytest -m gpu

e2e:
	SENTINEL_MOCK=1 $(PY) -m testkit.e2e_stub

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
