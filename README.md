# Kassa Sentinel

Локальный AI-сервис для проверки видео с потолочной камеры кассы (Datathon ForteBank 2026).

Проверяющий загружает `.mov` и получает:

* вердикт `violation` / `no_violation`;
* список возможных нарушений: тип, таймкод, описание «что видно и почему это может быть нарушением», 1–3 кадра-доказательства (JPEG);
* ссылку на норму РК (документ, пункт, URL adilet.zan.kz).

Результат — подсказка для человека, а не юридическое заключение. Вся обработка видео идёт локально: облачные LLM/vision API не используются, в рантайме нет интернета. Разрешены только модели под Apache-2.0, MIT или BSD.

Подробная документация (архитектура, модели и лицензии, нормы, запуск, раздел 8 ТЗ) — в [`docs/`](docs/) (A18). Питч — в [`pitch/`](pitch/).

## API (как в ТЗ)

| Метод | Что делает |
|---|---|
| `POST /api/v1/analyze` | multipart-поле `file`, заголовок `X-API-Key` → `{"job_id": "..."}` |
| `GET /api/v1/jobs/{job_id}` | `{"job_id", "status", "verdict", "violations": [{"type", "timecode_sec", "description", "evidence_frames": [{"timestamp_sec", "image_base64"}]}]}` |

Без верного ключа — 401. Схема ответа: [`contracts/api_schema.json`](contracts/api_schema.json). Проверка ответа: `python -m contracts.validate --jury resp.json`.

## Быстрый старт

```bash
make setup-dev          # контракт, тест-кит, pytest, ruff
make test-fast          # тесты без GPU и без весов (SENTINEL_MOCK=1)
make e2e                # сквозной mock-пайплайн на синтетическом видео
make setup && make weights && make run   # полный запуск (веса качаются один раз, дальше офлайн)
```

Windows / Docker / Cloudflare Tunnel — см. [`deploy/`](deploy/) (A17).

## Структура и владение (каждый агент пишет только в свою папку)

| ID | Папка | Модуль |
|---|---|---|
| A01 | `contracts/`, `testkit/`, `.github/`, `scripts/`, корневые файлы | Контракты, скелет, тест-кит, CI |
| A02 | `backend/api/` | FastAPI, API-ключ, очередь задач |
| A03 | `backend/pipeline/` | Оркестратор, бюджет времени, кэш, CLI |
| A04 | `ml/ingest/` | Декодирование видео, сэмплинг, качество, аугментации ракурса |
| A05 | `ml/detect/` | Детекция объектов (closed-set и open-vocab) |
| A06 | `ml/track/` | Трекинг, роли людей, авто-зоны |
| A07 | `ml/pose/` | Поза и жесты рук |
| A08 | `ml/scene/` | Состояния сцены: сейф, дверь, наличные, присутствие |
| A09 | `ml/rules/` | Движок правил → события-нарушения |
| A10 | `ml/temporal/` | Временная модель (SigLIP2 + temporal head) |
| A11 | `ml/vlm/` | Локальная VLM: верификация и поиск новых типов |
| A12 | `ml/fusion/` | Слияние, вердикт, кадры-доказательства, описания |
| A13 | `norms/` | Нормативка РК, каталог нарушений |
| A14 | `copilot/` | Локальный LLM-ассистент проверяющего |
| A15 | `frontend/` | Веб-интерфейс |
| A16 | `eval/` | Метрики как у жюри, LOO, стресс-тест ракурса |
| A17 | `deploy/` | Docker, офлайн-веса, Windows, Cloudflare Tunnel |
| A18 | `docs/`, `pitch/` | Документы для сдачи, питч |

Общие пустые файлы `ml/__init__.py` и `backend/__init__.py` создавать может любой агент.

## Для разработчиков модулей

* **Контракт:** [`contracts/README.md`](contracts/README.md). Там описано, как подключиться, оба сериализатора (ответ жюри и UI-report) и политика вердикта. Что изменилось — в [`contracts/MIGRATION.md`](contracts/MIGRATION.md).
* **Тест-кит без реальных видео:** [`testkit/`](testkit/README.md). Там синтетическое видео кассы, `FakeClip` и фикстуры Detection/Track/Zone/SceneState по сценариям.
* **Тесты:**
  * `pytest <папка>` из корня, `pythonpath=["."]`;
  * тяжёлые тесты помечать `@pytest.mark.gpu` / `@pytest.mark.slow`;
  * моки включаются через `SENTINEL_MOCK=1`.
* **CI (`.github/workflows/ci.yml`):**
  * обязательные проверки: `core`, `modules` (матрица по модулям, `fail-fast: false`), `e2e`;
  * необязательная: `frontend`.
* **Интеграция:** порядок мерджа и чек-лист — [`scripts/merge_order.md`](scripts/merge_order.md). Проверка чужих файлов — `scripts/check_ownership.py`. Лицензии, облачные API и файлы в git — `scripts/check_licenses.py`. Сверка fallback-копий контракта с каноном — `scripts/check_fallbacks.py`.
* **В git не коммитим:** видео, кадры из датасета, веса (`*.mov`, `*.mp4`, `*.onnx`, `*.pt`, `*.safetensors`, `*.gguf`…), ключи и `.env`. См. `.gitignore`.
