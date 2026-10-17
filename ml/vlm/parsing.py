"""Надёжное извлечение JSON из ответа VLM: code fences, <think>, мусор вокруг, одинарные кавычки,
висячие запятые, Python-литералы, обрезанный хвост. Затем — нормализация полей под схему."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

VERIFY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "visible": {"type": "boolean"},
        "type_id": {"type": "string", "pattern": "^V[0-9]{2}$"},
        "observed_facts": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "best_frame": {"type": "integer", "minimum": 1},
        "uncertain_because": {"type": "string"},
    },
    "required": ["visible", "type_id", "observed_facts", "confidence", "uncertain_because"],
}


def discover_schema(items: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "findings": {
                "type": "array",
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "properties": {
                        "item": {"type": "string", "enum": list(items)},
                        "observed_facts": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "frame": {"type": "integer", "minimum": 1},
                    },
                    "required": ["item", "observed_facts", "confidence"],
                },
            },
            "uncertain_because": {"type": "string"},
        },
        "required": ["findings"],
    }


@dataclass
class ParseResult:
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    repaired: bool = False
    raw: str = ""


_THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.S | re.I)
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)(```|$)", re.S)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_UNQUOTED_KEY_RE = re.compile(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)')
_PY_LITERALS = ((r"\bTrue\b", "true"), (r"\bFalse\b", "false"), (r"\bNone\b", "null"))


def _strip_wrappers(text: str) -> str:
    text = _THINK_RE.sub("", text or "")
    m = _FENCE_RE.search(text)
    if m and "{" in m.group(1):
        return m.group(1)
    return text


def _balanced_candidates(text: str) -> list[str]:
    """Все top-level {...} блоки с учётом строк; незакрытый хвостовой блок — тоже (для ремонта)."""
    out: list[str] = []
    depth = 0
    start = -1
    in_str: Optional[str] = None
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == in_str:
                in_str = None
            continue
        if ch in ('"', "'") and depth > 0:
            in_str = ch
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                out.append(text[start:i + 1])
                start = -1
    if depth > 0 and start >= 0:
        out.append(text[start:])  # обрезанный ответ (max_new_tokens)
    return out


def _close_truncated(s: str) -> str:
    """Дописывает недостающие кавычки/скобки у обрезанного JSON."""
    stack: list[str] = []
    in_str = False
    esc = False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
    s2 = s + ('"' if in_str else "")
    s2 = re.sub(r",\s*$", "", s2.rstrip())
    s2 = re.sub(r'(,\s*"[^"]*"\s*:?\s*)$', "", s2) if not in_str else s2  # висячий ключ без значения
    s2 = re.sub(r":\s*$", ": null", s2)
    return s2 + "".join(reversed(stack))


def _repair(s: str) -> str:
    s = s.replace("“", '"').replace("”", '"').replace("«", '"').replace("»", '"')
    for pat, rep in _PY_LITERALS:
        s = re.sub(pat, rep, s)
    if '"' not in s and "'" in s:
        s = s.replace("'", '"')
    else:
        # одинарные кавычки вокруг ключей/значений: 'key': 'value'
        s = re.sub(r"(?<=[{,\[:\s])'([^'\"]*)'(?=\s*[:,}\]])", r'"\1"', s)
    s = _UNQUOTED_KEY_RE.sub(r'\1"\2"\3', s)
    s = _TRAILING_COMMA_RE.sub(r"\1", s)
    return s


def extract_json(text: str) -> ParseResult:
    raw = text or ""
    body = _strip_wrappers(raw)
    cands = _balanced_candidates(body)
    if not cands:
        return ParseResult(ok=False, error="no_json_object", raw=raw)
    # предпочитаем самый длинный валидный объект
    for cand in sorted(cands, key=len, reverse=True):
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return ParseResult(ok=True, data=obj, raw=raw)
        except json.JSONDecodeError:
            pass
    for cand in sorted(cands, key=len, reverse=True):
        for attempt in (_repair(cand), _close_truncated(_repair(cand))):
            try:
                obj = json.loads(attempt)
                if isinstance(obj, dict):
                    return ParseResult(ok=True, data=obj, repaired=True, raw=raw)
            except json.JSONDecodeError:
                continue
    return ParseResult(ok=False, error="invalid_json", raw=raw)


# ---------------------------------------------------------------- нормализация полей

_YES = {"true", "yes", "да", "1", "visible", "видно", "y", "истина"}
_NO = {"false", "no", "нет", "0", "not visible", "не видно", "n", "ложь", "none", "null"}
_CONF_WORDS = {"very high": 0.9, "high": 0.8, "высокая": 0.8, "medium": 0.55, "moderate": 0.55,
               "средняя": 0.55, "low": 0.3, "низкая": 0.3, "very low": 0.15}


def coerce_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in _YES:
            return True
        if s in _NO:
            return False
    return None


def coerce_conf(v: Any) -> Optional[float]:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        x = float(v)
    elif isinstance(v, str):
        s = v.strip().lower().replace(",", ".")
        if s in _CONF_WORDS:
            return _CONF_WORDS[s]
        m = re.search(r"-?\d+(\.\d+)?", s)
        if not m:
            return None
        x = float(m.group(0))
        if "%" in s:
            x /= 100.0
    else:
        return None
    if x > 1.0 and x <= 100.0:
        x /= 100.0
    if x != x:  # NaN
        return None
    return max(0.0, min(1.0, x))


def coerce_type_id(v: Any, allowed: Optional[set[str]] = None) -> Optional[str]:
    if v is None:
        return None
    m = re.search(r"V\s*-?\s*(\d{1,2})", str(v).upper())
    if not m:
        return None
    tid = f"V{int(m.group(1)):02d}"
    if allowed is not None and tid not in allowed:
        return "V99" if "V99" in allowed else None
    return tid


def coerce_str_list(v: Any, max_items: int = 6) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        parts = [p.strip(" -•\t") for p in re.split(r"[\n;]+", v)]
    elif isinstance(v, (list, tuple)):
        parts = []
        for x in v:
            if isinstance(x, dict):  # {"fact": "..."} — бывает у маленьких моделей
                x = x.get("fact") or x.get("text") or next(iter(x.values()), "")
            parts.append(str(x).strip())
    else:
        parts = [str(v).strip()]
    out: list[str] = []
    for p in parts:
        if p and p not in out:
            out.append(p)
    return out[:max_items]


def coerce_int(v: Any) -> Optional[int]:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, str):
        m = re.search(r"\d+", v)
        return int(m.group(0)) if m else None
    return None


def normalize_verify(data: dict[str, Any], allowed_types: Optional[set[str]] = None) -> tuple[Optional[dict[str, Any]], str]:
    """Приводит ответ verify к схеме. (None, причина) — если нет обязательного 'visible'."""
    d = {str(k).strip().lower(): v for k, v in data.items()}
    visible = coerce_bool(d.get("visible", d.get("is_visible", d.get("violation"))))
    if visible is None:
        return None, "missing_visible"
    conf = coerce_conf(d.get("confidence", d.get("conf")))
    out = {
        "visible": visible,
        "type_id": coerce_type_id(d.get("type_id", d.get("type")), allowed_types),
        "observed_facts": coerce_str_list(d.get("observed_facts", d.get("facts"))),
        "confidence": 0.5 if conf is None else conf,
        "confidence_missing": conf is None,
        "best_frame": coerce_int(d.get("best_frame", d.get("frame"))),
        "uncertain_because": str(d.get("uncertain_because") or "").strip()[:300],
    }
    return out, ""


def normalize_discover(data: dict[str, Any], items: list[str]) -> tuple[Optional[dict[str, Any]], str]:
    d = {str(k).strip().lower(): v for k, v in data.items()}
    raw = d.get("findings")
    if raw is None:
        # формат «чек-лист»: {"phone": true, ...}
        flat = [{"item": k, "observed_facts": [], "confidence": None} for k in items if coerce_bool(d.get(k))]
        if not flat and not any(k in d for k in items):
            return None, "missing_findings"
        raw = flat
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return None, "findings_not_list"
    findings = []
    for f in raw:
        if not isinstance(f, dict):
            continue
        fd = {str(k).strip().lower(): v for k, v in f.items()}
        item = str(fd.get("item") or fd.get("name") or "").strip().lower()
        if item not in items:
            continue
        if coerce_bool(fd.get("present", fd.get("visible", True))) is False:
            continue
        conf = coerce_conf(fd.get("confidence"))
        findings.append({
            "item": item,
            "observed_facts": coerce_str_list(fd.get("observed_facts", fd.get("fact", fd.get("facts")))),
            "confidence": 0.5 if conf is None else conf,
            "frame": coerce_int(fd.get("frame", fd.get("best_frame"))),
        })
    return {"findings": findings, "uncertain_because": str(d.get("uncertain_because") or "")[:300]}, ""


def regex_fallback_verify(text: str, allowed_types: Optional[set[str]] = None) -> Optional[dict[str, Any]]:
    """Последний шанс: вытащить ключевые поля регэкспами из текста, который не стал JSON."""
    t = text or ""
    mv = re.search(r"visible\W{0,4}\s*(true|false|yes|no|да|нет)", t, re.I)
    if not mv:
        return None
    mc = re.search(r"confidence\W{0,4}\s*([0-9.]+%?)", t, re.I)
    mt = re.search(r"type_id\W{0,4}\s*(V\s*-?\d{1,2})", t, re.I)
    conf = coerce_conf(mc.group(1)) if mc else None
    return {
        "visible": coerce_bool(mv.group(1)),
        "type_id": coerce_type_id(mt.group(1), allowed_types) if mt else None,
        "observed_facts": [],
        "confidence": 0.5 if conf is None else conf,
        "confidence_missing": conf is None,
        "best_frame": None,
        "uncertain_because": "regex_fallback",
    }
