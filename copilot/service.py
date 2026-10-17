"""Sentinel Copilot — ответы проверяющему по результату анализа и нормам (локально).

Публичное API:
    answer(job_result, question, history) -> {"answer", "citations": [{document, clause, url}], "grounded", ...}
    status() -> {"enabled", "available", "backend", "reason", ...}
    warmup() / release()  — прогрев / выгрузка локальной модели (VRAM для VLM, управляет A03)

Если copilot выключен или локальной модели нет — ответ с ``error`` = "disabled" | "unavailable"
и пустым ``answer`` (роутер отдаёт 501). Ответ никогда не подменяется выдуманным текстом и не
уходит во внешние AI API. Импорт модуля не загружает ни моделей, ни корпуса. Публичные функции
не бросают исключений: любая ошибка copilot возвращается кодом в ``error`` и не затрагивает
анализ видео.
"""
from __future__ import annotations

import logging
import queue
import re
import threading
import time
from typing import Any, Iterator, Optional, Sequence

from .config import copilot_enabled, default_config, is_mock
from .corpus import Corpus, load_corpus, norm_url
from .extractive import classify, compose, refusal_for, target_violations
from .guard import UNVERIFIED_ANSWER, check_answer, complete_prefix
from .llm import LLMBusy, LLMTimeout, make_llm
from .prompts import NormRef, build_messages, normalize_result

log = logging.getLogger(__name__)

DISCLAIMER = "Подсказка для проверяющего, не юридическое заключение."

ERROR_MESSAGES = {
    "disabled": "Ассистент отключён в этой установке (включается COPILOT_ENABLED=1).",
    "unavailable": "Ассистент недоступен: локальная языковая модель не загружена.",
    "timeout": "Ассистент не успел ответить за отведённое время. Повторите вопрос позже.",
    "busy": "Ассистент занят (GPU или другой вопрос). Повторите вопрос позже.",
    "empty_question": "Пустой вопрос.",
    "not_ready": "Анализ ещё не завершён — вопросы доступны после результата.",
    "llm_error": "Ошибка локальной модели. Ответ не сформирован.",
    "internal_error": "Внутренняя ошибка ассистента. Ответ не сформирован.",
}


def source_label(cite: dict[str, Any]) -> str:
    """Читаемая подпись источника для UI: «[1] Документ, п. 9 — url (норма каталога; не сверено)»."""
    label = f"[{cite.get('ref')}] " if cite.get("ref") else ""
    label += str(cite.get("short") or cite.get("document") or "")
    if cite.get("clause"):
        label += f", {cite['clause']}"
    if cite.get("url"):
        label += f" — {cite['url']}"
    tags = ["норма каталога" if cite.get("relation") == "catalog" else "найдено поиском"]
    if cite.get("status") == "candidate":
        tags.append("кандидат")
    if not cite.get("verified"):
        tags.append("не сверено")
    return f"{label} ({'; '.join(tags)})"


class CopilotService:
    def __init__(self, cfg: Optional[dict[str, Any]] = None, llm: Any = None,
                 corpus: Optional[Corpus] = None, retriever: Any = None) -> None:
        self.cfg = cfg or default_config()
        self._llm = llm
        self._llm_resolved = llm is not None
        self._llm_injected = llm is not None
        self._resolved_at = 0.0
        self._corpus = corpus
        self._retriever = retriever
        self._lock = threading.RLock()
        self.warnings: list[str] = []
        self.gpu_lock: Any = None  # общий замок GPU от A03 (acquire(timeout=...)/release())

    # ------------------------------------------------------------------ lazy resources
    RETRY_UNAVAILABLE_S = 30.0  # модель/llama-server могли появиться позже старта API

    @property
    def llm(self) -> Any:
        with self._lock:
            stale = (self._llm is None and not self._llm_injected
                     and time.monotonic() - self._resolved_at > self.RETRY_UNAVAILABLE_S)
            if not self._llm_resolved or stale:
                self.warnings = []
                self._llm = make_llm(self.cfg, self.warnings) if copilot_enabled(self.cfg) else None
                self._llm_resolved = True
                self._resolved_at = time.monotonic()
            return self._llm

    @property
    def corpus(self) -> Corpus:
        with self._lock:
            if self._corpus is None:
                self._corpus = load_corpus(self.cfg)
            return self._corpus

    @property
    def retriever(self) -> Any:
        with self._lock:
            if self._retriever is None:
                from .rag import Retriever

                self._retriever = Retriever(self.corpus, self.cfg)
            return self._retriever

    def reload(self) -> None:
        """Перечитать каталог норм/корпус (например, после обновления norms/ от A13)."""
        with self._lock:
            self._corpus = None
            self._retriever = None

    # ------------------------------------------------------------------ status / VRAM
    def status(self) -> dict[str, Any]:
        enabled = copilot_enabled(self.cfg) or (self._llm_resolved and self._llm is not None)
        try:
            llm = self.llm if enabled else None
            info = llm.info() if llm is not None and hasattr(llm, "info") else {}
        except Exception as exc:  # статус не должен падать ни при каких настройках
            log.warning("copilot status error: %r", exc)
            self.warnings.append(f"ошибка инициализации: {type(exc).__name__}")
            llm, info = None, {}
        reason = None if llm is not None else ("disabled" if not enabled else "unavailable")
        return {
            "enabled": bool(enabled),
            "available": llm is not None,
            "backend": getattr(llm, "name", None),
            "mock": getattr(llm, "name", "") == "mock" or (is_mock() and llm is not None),
            "reason": reason,
            "message": ERROR_MESSAGES.get(reason or "", ""),
            "llm": info,
            "warnings": list(dict.fromkeys(self.warnings)),
            "disclaimer": DISCLAIMER,
        }

    def warmup(self) -> bool:
        """Прогрев (модель + индекс норм). Никогда не бросает: False — copilot недоступен/ошибка."""
        try:
            llm = self.llm
            if llm is None:
                return False
            if hasattr(llm, "warmup"):
                llm.warmup()
            _ = self.retriever
            return True
        except Exception as exc:
            log.warning("copilot warmup failed: %r", exc)
            self.warnings.append(f"прогрев не удался: {type(exc).__name__}")
            return False

    def release(self) -> None:
        """Выгрузить модель (освободить VRAM). Никогда не бросает."""
        llm = self._llm
        try:
            if llm is not None and hasattr(llm, "release"):
                llm.release()
        except Exception as exc:
            log.warning("copilot release failed: %r", exc)

    # ------------------------------------------------------------------ helpers
    def _clean_history(self, history: Any) -> list[dict[str, str]]:
        gcfg = self.cfg.get("guard", {})
        turns: list[dict[str, str]] = []
        for t in history if isinstance(history, list) else []:
            if not isinstance(t, dict) or t.get("role") not in {"user", "assistant"}:
                continue
            content = str(t.get("content") or t.get("text") or "").strip()  # {role, text} от A15 тоже
            if content:
                turns.append({"role": str(t["role"]), "content": content})
        turns = turns[-int(gcfg.get("max_history_turns", 8)):]
        budget = int(gcfg.get("max_history_chars", 4000))
        out: list[dict[str, str]] = []
        for t in reversed(turns):
            if budget <= 0:
                break
            content = t["content"][-budget:]
            budget -= len(content)
            out.append({"role": t["role"], "content": content})
        return list(reversed(out))

    def _norm_refs(self, question: str, result: dict[str, Any], history: Sequence[dict[str, str]],
                   warnings: list[str]) -> list[NormRef]:
        corpus = self.corpus
        max_refs = int(self.cfg.get("llm", {}).get("max_norm_refs", 6))
        targets = target_violations(question, result)
        type_ids = [str(v.get("type_id")) for v in targets if v.get("type_id")]
        chunks: list[tuple[Any, float, str]] = []
        # 1) нормы каталога по типам нарушений, о которых спрашивают
        for tid in dict.fromkeys(type_ids):
            for c in corpus.rules_for_type(tid)[:2]:
                chunks.append((c, 1.0, "catalog"))
        # 2) нормы, уже приложенные к результату (A12 берёт их из каталога) — перепроверяем
        for v in targets:
            for r in v.get("rules") or []:
                if not isinstance(r, dict):
                    continue
                hit = corpus.find(r.get("document"), r.get("clause"), r.get("url"))
                if hit is None:
                    warnings.append(f"норма из результата не найдена в каталоге: {r.get('document')} {r.get('clause') or ''}".strip())
                elif hit.kind == "catalog_rule" or set(hit.type_ids) & set(type_ids):
                    chunks.append((hit, 0.9, "catalog"))
                else:
                    chunks.append((hit, 0.9, "retrieved"))
        # 3) RAG по вопросу (+ предыдущий вопрос для коротких уточнений)
        query = question
        last_user = next((t["content"] for t in reversed(history) if t["role"] == "user"), "")
        if last_user and len(question.split()) < 6:
            query = f"{last_user} {question}"
        names = " ".join(str(v.get("type") or "") for v in targets[:2])
        try:
            hits = self.retriever.search(f"{query} {names}".strip(), type_ids=type_ids)
        except Exception as exc:  # RAG не должен ронять ответ
            warnings.append(f"поиск по нормам недоступен: {type(exc).__name__}")
            hits = []
        # фрагмент источника, связанный с типом события через каталог (тот же URL+пункт), — тоже «каталог»
        chunks.extend((h.chunk, h.score, "catalog" if set(h.chunk.type_ids) & set(type_ids) else "retrieved")
                      for h in hits)
        refs: list[NormRef] = []
        seen: set[Any] = set()
        for c, score, relation in chunks:
            keys = {c.key if c.key[1] else (c.key[0], c.id)}
            if c.url and c.key[1]:
                keys.add((norm_url(c.url), c.key[1]))  # одна норма из каталога и из источника — один раз
            if keys & seen:
                continue
            seen |= keys
            events = [int(v["_n"]) for v in result.get("violations") or []
                      if relation == "catalog" and v.get("type_id") in c.type_ids]
            refs.append(NormRef(n=len(refs) + 1, chunk=c, score=float(score), relation=relation, events=events))
            if len(refs) >= max_refs:
                break
        return refs

    def _error(self, code: str, job_id: str = "", **extra: Any) -> dict[str, Any]:
        out = {"answer": "", "citations": [], "sources": [], "grounded": False, "refusal": False, "error": code,
               "message": ERROR_MESSAGES.get(code, code), "job_id": job_id, "disclaimer": DISCLAIMER,
               "warnings": list(dict.fromkeys(self.warnings))}
        out.update(extra)
        return out

    # ------------------------------------------------------------------ main
    def _prepare(self, job_result: Any, question: str, history: Any) -> dict[str, Any]:
        result = normalize_result(job_result)
        job_id = str(result.get("job_id") or "")
        q = re.sub(r"\s+", " ", str(question or "")).strip()[: int(self.cfg.get("guard", {}).get("max_question_chars", 2000))]
        if not q:
            return {"error": self._error("empty_question", job_id)}
        if not copilot_enabled(self.cfg) and not (self._llm_resolved and self._llm is not None):
            return {"error": self._error("disabled", job_id)}
        llm = self.llm
        if llm is None:
            return {"error": self._error("unavailable", job_id)}
        if result.get("status") == "processing":
            return {"error": self._error("not_ready", job_id)}
        hist = self._clean_history(history)
        warnings: list[str] = list(self.corpus.warnings)
        intent = classify(q)
        refs = self._norm_refs(q, result, hist, warnings)
        return {"result": result, "job_id": job_id, "q": q, "hist": hist, "llm": llm, "intent": intent,
                "refs": refs, "warnings": warnings}

    def _check(self, p: dict[str, Any], raw: str) -> Any:
        gcfg = self.cfg.get("guard", {})
        return check_answer(raw, p["refs"], self.corpus, p["result"], p["q"], p["hist"],
                            float(gcfg.get("timecode_tolerance_s", 1.5)), float(gcfg.get("min_support_ratio", 0.3)))

    def _finish(self, p: dict[str, Any], raw: str, backend: str, t0: float) -> dict[str, Any]:
        g = self._check(p, raw)
        warnings = p["warnings"] + g.warnings
        if self.corpus.is_fixture and not any("фикстур" in w.lower() for w in warnings):
            warnings.append("нормы взяты из ТЕСТОВОЙ фикстуры — не использовать как ссылки на нормы РК")
        answer = g.answer or UNVERIFIED_ANSWER
        return {
            "answer": answer,
            "citations": g.citations,
            "sources": [source_label(c) for c in g.citations],
            "grounded": g.grounded,
            "refusal": g.refusal,
            "intent": p["intent"],
            "backend": backend,
            "job_id": p["job_id"],
            "removed_citations": g.removed,
            "warnings": list(dict.fromkeys(warnings)),
            "latency_ms": int((time.monotonic() - t0) * 1000),
            "disclaimer": DISCLAIMER,
            "error": None,
        }

    def _generate(self, p: dict[str, Any], deadline: float) -> Iterator[str]:
        """Поток сырых фрагментов ответа (до пост-проверки)."""
        llm = p["llm"]
        refusal = refusal_for(p["intent"], p["result"], self.corpus)
        if refusal is not None or not getattr(llm, "is_generative", False):
            text = refusal or compose(p["intent"], p["q"], p["result"], p["refs"], self.corpus)
            yield from re.findall(r"\S+\s*", text)
            return
        lcfg = self.cfg.get("llm", {})
        messages = build_messages(p["result"], p["refs"], self.corpus, p["q"], p["hist"])
        gpu_lock = self.gpu_lock if getattr(llm, "name", "") == "llama_cpp" else None
        if gpu_lock is not None and not gpu_lock.acquire(timeout=max(0.0, deadline - time.monotonic())):
            raise LLMBusy("shared GPU lock not acquired before deadline")
        try:
            yield from llm.stream_chat(messages, max_tokens=int(lcfg.get("max_tokens", 512)),
                                       temperature=float(lcfg.get("temperature", 0.2)), deadline=deadline)
        finally:
            if gpu_lock is not None:
                gpu_lock.release()

    def answer(self, job_result: Any, question: str, history: Any = None) -> dict[str, Any]:
        try:
            return self._answer(job_result, question, history)
        except Exception as exc:  # copilot не должен ронять вызывающий код (API/анализ)
            log.exception("copilot internal error")
            return self._error("internal_error", detail=type(exc).__name__)

    def _answer(self, job_result: Any, question: str, history: Any = None) -> dict[str, Any]:
        t0 = time.monotonic()
        p = self._prepare(job_result, question, history)
        if "error" in p:
            return p["error"]
        deadline = t0 + float(self.cfg.get("llm", {}).get("timeout_s", 45))
        backend = str(getattr(p["llm"], "name", "llm"))
        try:
            raw = "".join(self._generate(p, deadline))
        except (LLMTimeout, TimeoutError):
            return self._error("timeout", p["job_id"], backend=backend)
        except LLMBusy:
            return self._error("busy", p["job_id"], backend=backend)
        except Exception as exc:
            log.exception("copilot LLM error")
            return self._error("llm_error", p["job_id"], backend=backend, detail=type(exc).__name__)
        if time.monotonic() > deadline:
            return self._error("timeout", p["job_id"], backend=backend)
        return self._finish(p, raw, backend, t0)

    def stream(self, job_result: Any, question: str, history: Any = None) -> Iterator[dict[str, Any]]:
        """События: {"event":"token","text"} … {"event":"done", <ответ как answer()>} | {"event":"error", …}.

        В ``token`` уходят только ПРОВЕРЕННЫЕ предложения (тот же guard, что и для answer()),
        поэтому неподтверждённые цитаты пользователю не показываются даже временно. ``done.answer`` —
        окончательный текст (UI заменяет им накопленные токены). Генерация идёт в потоке-
        производителе: замки модели/GPU берутся и отпускаются в одном потоке.
        """
        try:
            yield from self._stream(job_result, question, history)
        except Exception as exc:
            log.exception("copilot internal error (stream)")
            yield {"event": "error", **self._error("internal_error", detail=type(exc).__name__)}

    def _stream(self, job_result: Any, question: str, history: Any = None) -> Iterator[dict[str, Any]]:
        t0 = time.monotonic()
        p = self._prepare(job_result, question, history)
        if "error" in p:
            yield {"event": "error", **p["error"]}
            return
        deadline = t0 + float(self.cfg.get("llm", {}).get("timeout_s", 45))
        backend = str(getattr(p["llm"], "name", "llm"))
        q: "queue.Queue[tuple[str, Any]]" = queue.Queue()
        cancel = threading.Event()

        def produce() -> None:
            try:
                for piece in self._generate(p, deadline):
                    if cancel.is_set():
                        return
                    q.put(("piece", piece))
                q.put(("end", None))
            except BaseException as exc:  # передаём потребителю
                q.put(("exc", exc))

        worker = threading.Thread(target=produce, name="copilot-stream", daemon=True)
        worker.start()
        parts: list[str] = []
        checked = ""   # сырой префикс, уже прошедший проверку
        emitted = ""   # проверенный текст, уже отданный клиенту
        try:
            while True:
                try:
                    kind, val = q.get(timeout=max(0.05, deadline - time.monotonic() + 1.0))
                except queue.Empty:
                    yield {"event": "error", **self._error("timeout", p["job_id"], backend=backend)}
                    return
                if kind == "piece":
                    parts.append(val)
                    prefix = complete_prefix("".join(parts))
                    if len(prefix) > len(checked):
                        checked = prefix
                        clean = self._check(p, prefix).answer
                        if clean != UNVERIFIED_ANSWER and clean.startswith(emitted) and len(clean) > len(emitted):
                            yield {"event": "token", "text": clean[len(emitted):]}
                            emitted = clean
                    continue
                if kind == "exc":
                    if isinstance(val, (LLMTimeout, TimeoutError)):
                        code = "timeout"
                    elif isinstance(val, LLMBusy):
                        code = "busy"
                    else:
                        log.error("copilot LLM error: %r", val)
                        yield {"event": "error", **self._error("llm_error", p["job_id"], backend=backend,
                                                               detail=type(val).__name__)}
                        return
                    yield {"event": "error", **self._error(code, p["job_id"], backend=backend)}
                    return
                break
        finally:
            cancel.set()
        done = self._finish(p, "".join(parts), backend, t0)
        if done["answer"].startswith(emitted) and len(done["answer"]) > len(emitted):
            yield {"event": "token", "text": done["answer"][len(emitted):]}
        yield {"event": "done", **done}


# ---------------------------------------------------------------------- module-level API
_service: Optional[CopilotService] = None
_service_lock = threading.Lock()


def get_service() -> CopilotService:
    global _service
    with _service_lock:
        if _service is None:
            _service = CopilotService()
        return _service


def set_service(service: Optional[CopilotService]) -> None:
    """Подменить сервис (тесты / интегратор с собственным конфигом)."""
    global _service
    with _service_lock:
        _service = service


def answer(job_result: Any, question: str, history: Optional[list[dict[str, str]]] = None) -> dict[str, Any]:
    return get_service().answer(job_result, question, history or [])


def status() -> dict[str, Any]:
    try:
        return get_service().status()
    except Exception as exc:  # даже битый конфиг не должен ронять API
        log.warning("copilot status failed: %r", exc)
        return {"enabled": False, "available": False, "backend": None, "mock": False, "reason": "unavailable",
                "message": ERROR_MESSAGES["unavailable"], "llm": {}, "warnings": [type(exc).__name__],
                "disclaimer": DISCLAIMER}


def warmup() -> bool:
    try:
        return get_service().warmup()
    except Exception as exc:
        log.warning("copilot warmup failed: %r", exc)
        return False


def release() -> None:
    try:
        get_service().release()
    except Exception as exc:
        log.warning("copilot release failed: %r", exc)


def set_gpu_lock(lock: Any) -> None:
    """A03: общий замок GPU (любой объект с acquire(timeout=)/release())."""
    get_service().gpu_lock = lock
