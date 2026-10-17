# norms/ — нормативная база РК и каталог типов нарушений (A13)

Каталог `catalog.yaml` даёт для каждого типа V01–V08 и V99 название, наблюдаемые и ненаблюдаемые
признаки, исключения, шаблон описания и нормы РК (`rules[]` в форме `RuleRef`). Ещё в модуле есть
загрузчик с fallback, валидатор схемы и целостности цитат, скачиватель полных текстов с adilet и
сохранённые фрагменты источников для RAG (A14).

## Главное о достоверности (прочитать до использования)

- **adilet.zan.kz и все зеркала из среды разработки были недоступны** (сетевая политика окружения:
  403 на `adilet.zan.kz`, `zakon.uchet.kz`, `prg.kz`, `online.zakon.kz` и др.). Работал только поиск.
  Поэтому `sources/*.md` — фрагменты из поисковой выдачи (`retrieval: search_snippets`), а не полные
  тексты, и у всех норм `verified: false`.
- Порядок рассуждения: действующая норма → условия применимости → наблюдаемое действие. Тип без
  подтверждённой нормы получает `legal_status: disabled`: норма к нему не прикрепляется (`rules: []`).
  Сейчас `candidate` только у **V08** (п. 20 НБРК № 231 и подп. 4 пункта о выдаче: пересчёт
  в присутствии клиента). Остальные типы — `disabled`, и у каждого записано, почему.
- Телефон, еда, открытая дверь или второй человек в кадре сами по себе не нарушение. По видео нельзя
  установить допуск и личность. Такие случаи перечислены в `exclusions`, общих и по типам.
- `python -m norms.validate` проверяет схему и **целостность цитат** (цитата есть в сохранённом
  источнике). Это не юридическая верификация: действие редакции и применимость проверяет человек.
- `clause: null` не означает ложный источник. Это ссылка на документ без установленного номера пункта,
  а уровень проверки указан в `rule_checks`.

## Публичный API

```python
import norms
cat = norms.load_catalog()          # замороженная сигнатура; dict с ключами version, types, documents, exclusions, levels
norms.legal_status("V08")           # "confirmed" | "candidate" | "disabled"
norms.rule_refs("V08")              # list[dict] ровно с ключами RuleRef; для disabled → []
norms.exclusions("V05")             # общие + типовые «само по себе не нарушение»
norms.render_description("V08", 22.4, ["клиента у окна нет"])  # по description_template
norms.type_name("V01"); norms.get_type("V42")                  # неизвестный id → V99
norms.catalog_source()              # "yaml" | "fallback" — для meta.warnings
```

Схема типа (совместима с преамбулой, новые поля можно игнорировать):
`name, observable[], not_observable[], description_template ({t}, {facts}), severity, rules[]` +
`legal_status, legal_note, exclusions[], rule_checks[] (по одному на норму: quote_level, quote_source,
currency, applicability, conditions_visible[], conditions_not_visible[]), research_candidates[]`.
Элементы `rules[]` содержат ровно `document, clause, url, quote, verified`, поэтому `RuleRef(**r)` работает.

## Команды

```bash
python -m norms.validate                 # схема + целостность цитат + сверка fallback; код возврата 0/1
python -m norms.validate --strict        # предупреждения (verified:false и т.п.) считать ошибками
python -m norms.validate --json          # отчёт для UI/CI
python -m norms.validate --write-fallback  # после правки catalog.yaml пересобрать norms/_fallback.py
python -m norms.fetch [--dry-run] [id]   # НА МАШИНЕ С ИНТЕРНЕТОМ: полные тексты adilet → sources/*.md
python -m pytest -q norms/tests          # тесты без сети, GPU и весов
```

Процесс сверки человеком: `python -m norms.fetch`, затем `python -m norms.validate`. Цитаты со статусом
NOT_FOUND нужно исправить по полному тексту. Проверить статус и редакцию на adilet. Только после этого
ставить `verified: true` (валидатор не даст сделать это без `retrieval: fulltext`) и повышать
`legal_status`.

## Файлы

| Файл | Назначение |
|---|---|
| `catalog.yaml` | каталог (источник истины) |
| `_fallback.py` | автогенерируемая копия каталога на случай, если нет файла или PyYAML |
| `__init__.py` | `load_catalog()` и хелперы, импорт без побочных эффектов |
| `validate.py` | валидатор (CLI `python -m norms.validate`) |
| `fetch.py` | скачивание полных текстов adilet (только stdlib, в рантайме не используется) |
| `sources/*.md` | сохранённые фрагменты источников с шапкой (url, accessed, retrieval) — корпус для RAG |
| `SOURCES.md` | таблица источников: реквизиты, статус, редакция; утратившие силу и неприменимые акты |
| `MAPPING.md` | норма → требование → признак → SIGNALS → что нельзя установить (ТЗ, раздел 6) |
| `models.json` | `[]`: модуль не использует ML-моделей |
| `_contract_fallback.py` | копия замороженного контракта, используется в тестах только если нет `contracts/` |

## Зависимости и модели

`PyYAML>=6.0` (без него работает встроенный fallback), `pytest` для тестов. Моделей нет, сеть
в рантайме не нужна.

## Для интегратора

- **A12 (fusion):** брать нормы через `norms.rule_refs(type_id)`, тогда для disabled-типов
  вернётся `[]`. При пустом списке в описании писать «нормативное основание не подтверждено»
  (это уже есть в `description_template` disabled-типов). Значение `legal_status` полезно отдать
  в `Violation`/`meta` (например, `meta["legal_status"][type_id]`). Это предложение в
  дополнительные поля, контракт не меняется.
- **A09/A12:** `legal_status` относится только к ссылке на норму. Влиять ли им на вердикт, решают
  владельцы правил и fusion. Рекомендация: `exclusions` использовать как список ложных срабатываний,
  которые нужно подавлять или понижать.
- **A14 (copilot/RAG):** корпус — `norms/sources/*.md`. Блоки `<!-- not-quotable -->` — пересказы,
  цитировать их нельзя.
- **A15 (frontend):** рядом с нормой показывать `verified=false` как «требует сверки», а disabled — как «наблюдение».
- **A16/A01 (CI):** добавить `python -m norms.validate` и `pytest norms/tests` в CI.
- Env: `SENTINEL_NORMS_CATALOG` — путь к альтернативному каталогу (тесты, демо).
- `_contract_fallback.py` можно удалить после мерджа, когда `contracts/` будет в main (в этой ветке
  canonical-контракт уже есть, fallback нужен только тестам).
- Известные TODO: см. `SOURCES.md` (п. 22 НБРК № 231, номера пунктов подп. 3–4 о сумках и подп. 4
  о выдаче, видеонаблюдение и доступ в кассовый узел по НБРК № 14). Чтобы сеть агента видела
  adilet, нужно добавить `adilet.zan.kz` в Allowed domains окружения.
