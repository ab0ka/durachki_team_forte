from __future__ import annotations

from copilot.corpus import build_corpus, chunk_markdown, norm_clause, norm_url
from copilot.rag import Retriever

TOY_MD = """---
title: "Тестовые правила кассы"
url: "https://example.invalid/toy"
---
# Тестовые правила кассы

## Глава 1. Общие положения

1. Кассир ведёт операции только на рабочем месте.

2. Посторонние лица в кассовое помещение не допускаются.
1) клиенты обслуживаются через окно;
2) инкассаторы допускаются по процедуре.

### Пункт 3
Мобильный телефон при работе с ценностями не используется.
"""


def test_norm_clause_variants() -> None:
    assert norm_clause("п. 3") == norm_clause("пункт 3") == norm_clause("п.3") == "п.3"
    assert norm_clause("Статья 5") == "ст.5"
    assert norm_clause("пп. 2)") == "пп.2"
    assert norm_clause("12.") == "п.12"
    assert norm_url("https://Adilet.zan.kz/rus/docs/V123/#p5") == "adilet.zan.kz/rus/docs/v123"


def test_chunk_markdown_detects_document_clauses_and_url() -> None:
    chunks = chunk_markdown(TOY_MD, source_file="toy.md")
    assert chunks, "чанки должны быть"
    assert {c.document for c in chunks} == {"Тестовые правила кассы"}
    assert all(c.url == "https://example.invalid/toy" for c in chunks)
    clauses = [c.clause for c in chunks]
    assert "п. 1" in clauses and "п. 2" in clauses and "п. 3" in clauses
    p2 = next(c for c in chunks if c.clause == "п. 2")
    assert "окно" in p2.text, "подпункты 1)/2) остаются внутри пункта"
    assert "Глава 1" in p2.heading


def test_chunk_markdown_splits_long_sections() -> None:
    long = "# Док\n\n5. " + " ".join(["Очень длинное предложение про кассу."] * 80)
    chunks = chunk_markdown(long, max_chars=300, overlap=50)
    assert len(chunks) > 3
    assert all(len(c.text) <= 300 for c in chunks)
    assert all(c.clause == "п. 5" for c in chunks)


def test_catalog_and_sources_loaded(corpus) -> None:
    assert corpus.is_fixture  # файл-фикстура помечен fixture: true -> в ответах будет предупреждение
    kinds = {c.kind for c in corpus.chunks}
    assert kinds == {"catalog_rule", "source"}
    rules = corpus.rules_for_type("V05")
    assert rules and rules[0].clause == "п. 9"
    # источник с тем же пунктом привязан к типу
    src = [c for c in corpus.chunks if c.kind == "source" and c.clause == "п. 9"]
    assert src and "V05" in src[0].type_ids


def test_find_citation(corpus) -> None:
    assert corpus.find("ФИКСТУРА: Тестовый регламент кассы", "пункт 9") is not None
    assert corpus.find(None, "п. 9", "https://example.invalid/fixture/reglament") is not None
    assert corpus.find("ФИКСТУРА: Тестовый регламент кассы", "п. 999") is None
    assert corpus.find("Выдуманный закон", "п. 9") is None


def test_retriever_top1_is_relevant(cfg, corpus, embedder) -> None:
    r = Retriever(corpus, cfg, embedder=embedder)
    cases = {
        "можно ли кассиру разговаривать по мобильному телефону": "п. 9",
        "посторонний человек зашёл в кассовую зону": "п. 3",
        "инкассаторская сумка оставлена без присмотра": "п. 4",
        "сейф открыт и кассир ушёл": "п. 7",
    }
    for q, clause in cases.items():
        hits = r.search(q, top_k=3)
        assert hits, q
        assert hits[0].chunk.clause == clause, (q, [(h.chunk.clause, round(h.score, 3)) for h in hits])
        top = hits[0].chunk
        assert top.document and top.url  # у каждого результата есть источник


def test_retriever_dedupes_same_clause(cfg, corpus, embedder) -> None:
    hits = Retriever(corpus, cfg, embedder=embedder).search("мобильный телефон ценности", top_k=6)
    keys = [h.chunk.key for h in hits if h.chunk.clause]
    assert len(keys) == len(set(keys))


def test_type_boost(cfg, corpus, embedder) -> None:
    hits = Retriever(corpus, cfg, embedder=embedder).search("что нарушено", top_k=2, type_ids=["V07"])
    assert hits and "V07" in hits[0].chunk.type_ids


def test_empty_corpus_is_safe(cfg, embedder) -> None:
    c = build_corpus({"types": {}}, [], cfg=cfg)
    assert Retriever(c, cfg, embedder=embedder).search("телефон") == []
