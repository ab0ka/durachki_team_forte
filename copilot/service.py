"""Sentinel Copilot — ответы проверяющему по результату анализа и нормам (локально).

Публичное API:
    answer(job_result, question, history) -> {"answer", "citations": [{document, clause, url}], "grounded", ...}
    status() -> {"enabled", "available", "backend", "reason", ...}
    warmup() / release()  — прогрев / выгрузка локальной модели (VRAM для VLM, управляет A03)

Если copilot выключен или локальной модели нет — ответ с ``error`` = "disabled" | "unavailable"
и пустым ``answer`` (роутер отдаёт 501). Ответ никогда не подменяется выдуманным текстом и не
уходит во внешние AI API. Импорт модуля не загружает ни моделей, ни корпуса.
"""
from __future__ import annotations

import logging
import queue
import re
import threading
import time
from typing import Any, Iterator, Optional, Sequence

from .config import copilot_enabled, default_config, is_mock
from .corpus import Corpus, load_corpus
from .extractive import classify, compose, refusal_for, target_violations
from .guard import check_answer
from .llm import LLMTimeout, ThinkFilter, make_llm
from .prompts import NormRef, build_messages, normalize_result

log = logging.getLogger(__name__)

DISCLAIMER = "Подсказка для проверяющего, не юридическое заключение."

ERROR_MESSAGES = {
    "disabled": "Ассистент отключён в этой установке (COPILOT_ENABLED=0).",
    "unavailable": "Ассистент недоступен: локальная языковая модель не загружена.",
    "timeout": "Ассистент не успел ответить за отведённое время. Повторите вопрос позже.",
    "busy": "GPU занят анализом видео. Повторите вопрос позже.",
    "empty_question": "Пустой вопрос.",
    "not_ready": "Анализ ещё не завершён — вопросы доступны после результата.",
    "llm_error": "Ошибка локальной модели. Ответ не сформирован.",
}


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
        llm = self.llm if enabled else None
        info = llm.info() if llm is not None and hasattr(llm, "info") else {}
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

    def warmup(self) -> None:
        llm = self.llm
        if llm is not None and hasattr(llm, "warmup"):
            llm.warmup()
        if llm is not None:
            _ = self.retriever

    def release(self) -> None:
        llm = self._llm
        if llm is not None and hasattr(llm, "release"):
            llm.release()

    # ------------------------------------------------------------------ helpers
    def _clean_history(self, history: Any) -> list[dict[str, str]]:
        gcfg = self.cfg.get("guard", {})
        turns: list[dict[str, str]] = []
        for t in history or []:
            if isinstance(t, dict) and t.get("role") in {"user", "assistant"} and str(t.get("content") or "").strip():
                turns.append({"role": str(t["role"]), "content": str(t["content"]).strip()})
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
        chunks: list[tuple[Any, float]] = []
        # 1) нормы каталога по типам нарушений, о которых спрашивают
        for tid in dict.fromkeys(type_ids):
            for c in corpus.rules_for_type(tid)[:2]:
                chunks.append((c, 1.0))
        # 2) нормы, уже приложенные к результату (A12 берёт их из каталога) — перепроверяем
        for v in targets:
            for r in v.get("rules") or []:
                if not isinstance(r, dict):
                    continue
                hit = corpus.find(r.get("document"), r.get("clause"), r.get("url"))
                if hit is None:
                    warnings.append(f"норма из результата не найдена в каталоге: {r.get('document')} {r.get('clause') or ''}".strip())
                else:
                    chunks.append((hit, 0.9))
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
        chunks.extend((h.chunk, h.score) for h in hits)
        refs: list[NormRef] = []
        seen: set[Any] = set()
        for c, score in chunks:
            key = c.key if c.key[1] else (c.key[0], c.id)
            if key in seen:
                continue
            seen.add(key)
            refs.append(NormRef(n=len(refs) + 1, chunk=c, score=float(score)))
            if len(refs) >= max_refs:
                break
        return refs

    def _error(self, code: str, job_id: str = "", **extra: Any) -> dict[str, Any]:
        out = {"answer": "", "citations": [], "grounded": False, "refusal": False, "error": code,
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

    def _finish(self, p: dict[str, Any], raw: str, backend: str, t0: float) -> dict[str, Any]:
        g = check_answer(raw, p["refs"], self.corpus, p["result"], p["q"], p["hist"],
                         float(self.cfg.get("guard", {}).get("timecode_tolerance_s", 1.5)))
        warnings = p["warnings"] + g.warnings
        if self.corpus.is_fixture:
            warnings.append("нормы взяты из ТЕСТОВОЙ фикстуры — не использовать как ссылки на нормы РК")
        answer = g.answer
        if not answer:
            answer = "Ответ не прошёл проверку источников и был удалён. Нет данных для подтверждённого ответа."
        return {
            "answer": answer,
            "citations": g.citations,
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
            raise _Busy()
        try:
            yield from llm.stream_chat(messages, max_tokens=int(lcfg.get("max_tokens", 512)),
                                       temperature=float(lcfg.get("temperature", 0.2)), deadline=deadline)
        finally:
            if gpu_lock is not None:
                gpu_lock.release()

    def answer(self, job_result: Any, question: str, history: Any = None) -> dict[str, Any]:
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
        except _Busy:
            return self._error("busy", p["job_id"], backend=backend)
        except Exception as exc:
            log.exception("copilot LLM error")
            return self._error("llm_error", p["job_id"], backend=backend, detail=type(exc).__name__)
        if time.monotonic() > deadline:
            return self._error("timeout", p["job_id"], backend=backend)
        return self._finish(p, raw, backend, t0)

    def stream(self, job_result: Any, question: str, history: Any = None) -> Iterator[dict[str, Any]]:
        """События: {"event":"token","text"} … {"event":"done", <ответ как answer()>} | {"event":"error", …}.

        Токены идут ДО пост-проверки: UI обязан заменить текст на ``done.answer``.
        Генерация идёт в отдельном потоке-производителе: замки модели/GPU берутся и отпускаются
        в одном потоке, даже если потребитель (Starlette) дёргает next() из разных потоков.
        """
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
        flt = ThinkFilter()
        try:
            while True:
                try:
                    kind, val = q.get(timeout=max(0.05, deadline - time.monotonic() + 1.0))
                except queue.Empty:
                    yield {"event": "error", **self._error("timeout", p["job_id"], backend=backend)}
                    return
                if kind == "piece":
                    parts.append(val)
                    visible = flt.feed(val)
                    if visible:
                        yield {"event": "token", "text": visible}
                    continue
                if kind == "exc":
                    if isinstance(val, (LLMTimeout, TimeoutError)):
                        code = "timeout"
                    elif isinstance(val, _Busy):
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
        yield {"event": "done", **self._finish(p, "".join(parts), backend, t0)}


class _Busy(Exception):
    pass


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
    return get_service().status()


def warmup() -> None:
    get_service().warmup()


def release() -> None:
    get_service().release()


def set_gpu_lock(lock: Any) -> None:
    """A03: общий замок GPU (любой объект с acquire(timeout=)/release())."""
    get_service().gpu_lock = lock
