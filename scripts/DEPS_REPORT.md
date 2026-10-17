# Зависимости модулей против lock L1

Снимок по наложенным деревьям всех веток `origin/claude/*` (см. `contracts/COMPAT.md`, раздел 3). Перегенерация: `python scripts/resolve_deps.py --tree <дерево> --lock-ref origin/data/labels:tools/dataprep/requirements-l1.lock.txt --out scripts/DEPS_REPORT.md`.

**Главное:** `ml/vlm` (A11) требует `transformers<5`, а в lock L1 — 5.20.1 (API ветки 5.x). Без доводки A11 под 5.x VLM не встанет в окружение L1. Стек API (`fastapi`, `uvicorn`, `starlette`, `python-multipart`, `httpx`) и `onnxruntime-gpu` в lock L1 отсутствуют — решение L1/A17.

Lock: `origin/data/labels:tools/dataprep/requirements-l1.lock.txt`. Сводка: КОНФЛИКТ — 5, НЕТ В LOCK — 12, OK — 12.

Версии берутся только из lock. Для пакетов «НЕТ В LOCK» максимальная версия не подставляется: решение за L1 — добавить в lock или убрать из модуля.

| Пакет | Статус | Lock | Модули (требование) | Деталь |
|---|---|---|---|---|
| `huggingface-hub` | КОНФЛИКТ | 2.2.0 | deploy (<2,>=0.25); ml/detect (>=0.23); ml/vlm (>=0.25) | deploy: <2,>=0.25 ⟂ lock 2.2.0 |
| `pillow` | КОНФЛИКТ | 13.0.0 | ml/fusion (<13,>=10.0); ml/vlm (>=10.1) | ml/fusion: <13,>=10.0 ⟂ lock 13.0.0 |
| `pytest` | КОНФЛИКТ | 9.1.1 | backend/api (>=8.0); backend/pipeline (>=8); copilot (>=8); eval (<9,>=8); ml/detect (>=8); ml/fusion (>=8.0); ml/ingest (>=7.4); ml/pose (>=7); ml/rules (>=8.0); ml/scene (>=8.0); ml/temporal (>=8); ml/track (>=7); ml/vlm (>=8); norms (>=7.4) | eval: <9,>=8 ⟂ lock 9.1.1 |
| `pyyaml` | КОНФЛИКТ | 6.0.3 | backend/api (>=6.0); backend/pipeline (<7,>=6.0); copilot (<7,>=6.0); deploy (>=6.0); eval (==6.0.1); ml/detect (>=6.0); ml/fusion (<7,>=6.0); ml/ingest (>=6.0); ml/pose (<7,>=6.0); ml/rules (<7,>=6.0); ml/scene (>=6.0); ml/temporal (>=6.0); ml/track (>=6.0); ml/vlm (>=6.0); norms (<7,>=6.0) | eval: ==6.0.1 ⟂ lock 6.0.3 |
| `transformers` | КОНФЛИКТ | 5.20.1 | ml/detect (>=4.52); ml/temporal (>=4.49); ml/vlm (<5,>=4.57.1) | ml/vlm: <5,>=4.57.1 ⟂ lock 5.20.1 |
| `albumentations` | НЕТ В LOCK | — | ml/detect (>=1.4) | требуют: >=1.4 — решение за L1 |
| `fastapi` | НЕТ В LOCK | — | backend/api (<1.0,>=0.115); copilot (>=0.110) | требуют: <1.0,>=0.115, >=0.110 — решение за L1 |
| `httpx` | НЕТ В LOCK | — | backend/api (>=0.27); copilot (>=0.27) | требуют: >=0.27 — решение за L1 |
| `matplotlib` | НЕТ В LOCK | — | ml/scene (>=3.7) | требуют: >=3.7 — решение за L1 |
| `onnx` | НЕТ В LOCK | — | ml/detect (>=1.16) | требуют: >=1.16 — решение за L1 |
| `onnxruntime-gpu` | НЕТ В LOCK | — | ml/detect (>=1.18); ml/pose (<2,>=1.17) | требуют: <2,>=1.17, >=1.18 — решение за L1 |
| `openpyxl` | НЕТ В LOCK | — | eval (==3.1.5) | требуют: ==3.1.5 — решение за L1 |
| `pydantic` | НЕТ В LOCK | — | copilot (<3,>=2.6) | требуют: <3,>=2.6 — решение за L1 |
| `python-multipart` | НЕТ В LOCK | — | backend/api (>=0.0.18) | требуют: >=0.0.18 — решение за L1 |
| `sentencepiece` | НЕТ В LOCK | — | ml/temporal (>=0.2) | требуют: >=0.2 — решение за L1 |
| `starlette` | НЕТ В LOCK | — | backend/api (>=0.40) | требуют: >=0.40 — решение за L1 |
| `uvicorn` | НЕТ В LOCK | — | backend/api (>=0.30) | требуют: >=0.30 — решение за L1 |
| `accelerate` | OK | 1.15.0 | ml/detect (>=1.1); ml/vlm (>=1.0) | — |
| `anyio` | OK | 4.15.1 | backend/api (>=4.4) | — |
| `av` | OK | 18.1.0 | ml/ingest (>=13.0) | — |
| `bitsandbytes` | OK | 0.50.2 | ml/vlm (>=0.45) | — |
| `jsonschema` | OK | 4.26.0 | eval (<5,>=4.21) | — |
| `numpy` | OK | 2.4.6 | backend/pipeline (>=1.26); copilot (<3,>=1.26); eval (<3,>=1.26); ml/detect (>=1.26); ml/fusion (<3,>=1.26); ml/ingest (>=1.26); ml/pose (<3,>=1.24); ml/rules (<3,>=1.24); ml/scene (>=1.26); ml/temporal (>=1.26); ml/track (<3,>=1.24); ml/vlm (>=1.26) | — |
| `opencv-python-headless` | OK | 4.14.0.94 | backend/api (>=4.8); backend/pipeline (<5,>=4.9); eval (>=4.9); ml/detect (>=4.9); ml/fusion (<6,>=4.9); ml/ingest (>=4.9); ml/pose (<5,>=4.8); ml/scene (<5,>=4.8); ml/temporal (>=4.9); ml/track (>=4.8) | — |
| `packaging` | OK | 26.3 | deploy (>=23) | — |
| `requests` | OK | 2.34.2 | eval (<3,>=2.31) | — |
| `safetensors` | OK | 0.8.0 | ml/detect (>=0.4) | — |
| `scipy` | OK | 1.17.1 | ml/track (>=1.10) | — |
| `torch` | OK | 2.11.0+cu128 | ml/detect (>=2.3); ml/temporal (>=2.3); ml/vlm (>=2.4) | — |
