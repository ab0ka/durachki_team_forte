from __future__ import annotations

from copilot.guard import check_answer, is_refusal
from copilot.prompts import NormRef, normalize_result


def _refs(corpus):
    return [NormRef(n=1, chunk=corpus.rules_for_type("V05")[0]), NormRef(n=2, chunk=corpus.rules_for_type("V03")[0])]


def test_valid_markers_become_footnotes(corpus, job_violation) -> None:
    res = normalize_result(job_violation)
    g = check_answer("Пользоваться телефоном запрещено [N1]. Сейф должен быть закрыт [N2].", _refs(corpus), corpus, res)
    assert "[1]" in g.answer and "[2]" in g.answer and "[N" not in g.answer
    assert [c["clause"] for c in g.citations] == ["п. 9", "п. 7"]
    assert all({"document", "clause", "url"} <= set(c) for c in g.citations)
    assert g.grounded and not g.removed


def test_invented_marker_removed(corpus, job_violation) -> None:
    res = normalize_result(job_violation)
    g = check_answer("Это запрещено [N1] и ещё [N7].", _refs(corpus), corpus, res)
    assert "[N7]" in g.removed
    assert len(g.citations) == 1
    assert not g.grounded  # в сыром ответе была выдуманная ссылка


def test_invented_clause_sentence_dropped(corpus, job_violation) -> None:
    res = normalize_result(job_violation)
    raw = "Пользоваться телефоном запрещено [N1]. Также нарушен пункт 999 регламента."
    g = check_answer(raw, _refs(corpus), corpus, res)
    assert "999" not in g.answer
    assert "Пользоваться телефоном запрещено [1]." == g.answer
    assert any("999" in r for r in g.removed)
    assert not g.grounded


def test_existing_clause_mention_kept(corpus, job_violation) -> None:
    res = normalize_result(job_violation)
    g = check_answer("Это требование п. 9 регламента [N1].", _refs(corpus), corpus, res)
    assert "п. 9" in g.answer and g.grounded


def test_invented_document_dropped(corpus, job_violation) -> None:
    res = normalize_result(job_violation)
    raw = "Телефон запрещён [N1]. Согласно Закону «О секретных кассах и банкоматах» это уголовно наказуемо."
    g = check_answer(raw, _refs(corpus), corpus, res)
    assert "секретных" not in g.answer and g.removed


def test_unknown_url_removed_known_kept(corpus, job_violation) -> None:
    res = normalize_result(job_violation)
    raw = ("Подробнее: https://adilet.zan.kz/rus/docs/V9999999 и https://example.invalid/fixture/reglament#p9 [N1].")
    g = check_answer(raw, _refs(corpus), corpus, res)
    assert "V9999999" not in g.answer
    assert "example.invalid/fixture/reglament" in g.answer


def test_hallucinated_timecode_dropped(corpus, job_violation) -> None:
    res = normalize_result(job_violation)
    raw = "Кассир взял телефон на 22 секунде [V1]. На 95 с кассир пересчитал деньги."
    g = check_answer(raw, _refs(corpus), corpus, res)
    assert "95" not in g.answer and "(№1, 00:22)" in g.answer
    assert "95 с" in g.removed


def test_timecode_from_question_allowed(corpus, job_violation) -> None:
    res = normalize_result(job_violation)
    g = check_answer("На 40 с признаков нарушения не обнаружено.", _refs(corpus), corpus, res,
                     question="Что происходит на 40 секунде?")
    assert "40 с" in g.answer and g.grounded


def test_unknown_violation_marker_removed(corpus, job_violation) -> None:
    res = normalize_result(job_violation)
    g = check_answer("См. [V9].", _refs(corpus), corpus, res)
    assert "[V9]" in g.removed


def test_think_block_stripped(corpus, job_violation) -> None:
    res = normalize_result(job_violation)
    g = check_answer("<think>рассуждаю про п. 777</think>Телефон запрещён [N1].", _refs(corpus), corpus, res)
    assert "777" not in g.answer and not g.removed and g.grounded


def test_refusal_detection() -> None:
    assert is_refusal("По видео это установить нельзя.")
    assert is_refusal("В источниках нет информации об этом.")
    assert not is_refusal("Кассир держит телефон.")


def test_refusal_is_grounded_without_citations(corpus, job_violation) -> None:
    res = normalize_result(job_violation)
    g = check_answer("По видео это установить нельзя.", [], corpus, res)
    assert g.refusal and g.grounded and g.citations == []


def test_clause_from_corpus_but_not_in_context_removed(corpus, job_violation) -> None:
    """п. 7 есть в корпусе, но модели передан только п. 9 — ссылка не подтверждена контекстом."""
    res = normalize_result(job_violation)
    refs = [NormRef(n=1, chunk=corpus.rules_for_type("V05")[0])]
    g = check_answer("Телефон запрещён [N1]. Сейф тоже регулирует п. 7 регламента.", refs, corpus, res)
    assert g.answer == "Телефон запрещён [1]." and "п. 7" in g.removed and not g.grounded


def test_vague_document_name_does_not_match(corpus) -> None:
    assert corpus.find("Регламент", "п. 9") is None
    assert corpus.find("Тестовый регламент кассы", "п. 9") is not None


def test_insinuation_with_caveat_word_is_still_checked(corpus, job_violation) -> None:
    """«Не означает» не даёт иммунитета: предложение без опоры на контекст всё равно вырезается."""
    res = normalize_result(job_violation)
    g = check_answer("Это не означает, что кассир тайно украл деньги и скрылся через чёрный ход.", [], corpus, res)
    assert "украл" not in g.answer
