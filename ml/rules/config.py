"""Загрузка и валидация ``rules.yaml`` (декларативные правила)."""
from __future__ import annotations

import copy
import os
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .expr import CompiledExpr, ExprError, compile_expr

try:
    from contracts.sentinel_contracts import SIGNALS  # noqa
except ImportError:  # пока ветка A01 не смерджена
    from ._contract_fallback import SIGNALS  # noqa

__all__ = [
    "RuleSpec",
    "RulesConfig",
    "ConfigError",
    "DEFAULT_RULES_PATH",
    "load_rules_config",
    "parse_rules_config",
    "apply_overrides",
    "read_raw_config",
]

DEFAULT_RULES_PATH = Path(__file__).with_name("rules.yaml")
ENV_RULES_PATH = "SENTINEL_RULES_YAML"
T_PEAK_POLICIES = ("onset", "onset_plus", "max", "center")
SMOOTH_METHODS = ("none", "median", "mean")
TYPE_ID_RE = re.compile(r"^V\d{2}$")
CONF_VARS = ("mean_strength", "max_strength", "duration", "dur_term", "base", "w_strength",
             "w_duration", "tau_s", "n_frames")


class ConfigError(ValueError):
    """Ошибка в rules.yaml."""


@dataclass
class RuleSpec:
    id: str
    type_id: str
    comment: str
    strength: CompiledExpr
    gate: Optional[CompiledExpr]
    enabled: bool = True
    on: float = 0.5
    off: float = 0.35
    smooth_method: str = "median"
    smooth_s: float = 0.6
    min_duration_s: float = 1.5
    merge_gap_s: float = 1.5
    t_peak_policy: str = "onset_plus"
    t_peak_offset_s: float = 0.5
    conf_base: float = 0.35
    conf_w_strength: float = 0.35
    conf_w_duration: float = 0.25
    conf_tau_s: float = 6.0
    conf_min: float = 0.05
    conf_max: float = 0.97
    conf_expr: Optional[CompiledExpr] = None
    weight: float = 1.0
    facts: list[str] = field(default_factory=list)
    highlight: list[str] = field(default_factory=list)
    yield_to: list[str] = field(default_factory=list)
    yield_factor: float = 0.7
    max_events: int = 3

    @property
    def names(self) -> frozenset[str]:
        names = set(self.strength.names)
        if self.gate is not None:
            names |= self.gate.names
        return frozenset(names)


@dataclass
class RulesConfig:
    version: int
    rules: list[RuleSpec]
    signal_defaults: dict[str, float] = field(default_factory=dict)
    default_value: float = 0.0
    ffill_max_s: float = 1.5
    merge_policy: str = "max"
    derive_from_tracks: bool = True
    presence_tol_s: float = 0.6
    bbox_tol_s: float = 1.0
    max_bboxes: int = 6
    nms_window_s: float = 5.0
    same_type_merge_gap_s: float = 2.0
    corroboration_bonus: float = 0.05
    max_events: int = 12
    min_confidence: float = 0.2
    severity_factor: dict[str, float] = field(default_factory=lambda: {"high": 1.0, "medium": 0.97, "low": 0.93})
    known_signals: frozenset[str] = frozenset(SIGNALS)
    source_path: Optional[str] = None
    warnings: list[str] = field(default_factory=list)

    @property
    def enabled_rules(self) -> list[RuleSpec]:
        return [r for r in self.rules if r.enabled]

    def rule(self, rule_id: str) -> RuleSpec:
        for r in self.rules:
            if r.id == rule_id:
                return r
        raise KeyError(rule_id)


def _num(d: Mapping[str, Any], key: str, default: float, lo: float = -1e9, hi: float = 1e9, where: str = "") -> float:
    v = d.get(key, default)
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ConfigError(f"{where}{key}: ожидалось число, получено {v!r}") from None
    if not (lo <= f <= hi):
        raise ConfigError(f"{where}{key}={f} вне диапазона [{lo}, {hi}]")
    return f


def _deep_merge(base: Mapping[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(dict(base))
    for k, v in (over or {}).items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _template_fields(template: str) -> list[str]:
    fields = []
    for _, fname, _, _ in string.Formatter().parse(template):
        if fname is None:
            continue
        if not fname or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", fname):
            raise ConfigError(f"Недопустимый плейсхолдер {{{fname}}} в шаблоне {template!r}")
        fields.append(fname)
    return fields


def read_raw_config(path: Optional[str | os.PathLike] = None) -> tuple[dict[str, Any], str]:
    import yaml  # ленивый импорт

    p = Path(path or os.environ.get(ENV_RULES_PATH) or DEFAULT_RULES_PATH)
    with open(p, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{p}: корень YAML должен быть словарём")
    return dict(raw), str(p)


def _parse_value(text: str) -> Any:
    import yaml

    return yaml.safe_load(text)


def apply_overrides(raw: Mapping[str, Any], sets: Sequence[str]) -> dict[str, Any]:
    """Применить переопределения ``a.b.c=value`` (значение — YAML-литерал).

    ``rules.<rule_id>.<param>`` адресует правило по id (rules в YAML — список).
    """
    out = copy.deepcopy(dict(raw))
    for item in sets or ():
        if "=" not in item:
            raise ConfigError(f"Переопределение без '=': {item!r}")
        key, value = item.split("=", 1)
        parts = [p for p in key.strip().split(".") if p]
        if not parts:
            raise ConfigError(f"Пустой ключ в {item!r}")
        val = _parse_value(value)
        node: Any = out
        if parts[0] == "rules" and len(parts) >= 3:
            rules = node.get("rules") or []
            target = next((r for r in rules if isinstance(r, dict) and r.get("id") == parts[1]), None)
            if target is None:
                raise ConfigError(f"Нет правила с id={parts[1]!r}")
            node, parts = target, parts[2:]
        for p in parts[:-1]:
            nxt = node.get(p)
            if not isinstance(nxt, dict):
                nxt = {}
                node[p] = nxt
            node = nxt
        node[parts[-1]] = val
    return out


def parse_rules_config(raw: Mapping[str, Any], source_path: Optional[str] = None) -> RulesConfig:
    warnings: list[str] = []
    glob = dict(raw.get("global") or {})
    defaults = dict(raw.get("defaults") or {})
    extra_signals = [str(s) for s in (raw.get("extra_signals") or [])]
    known = frozenset(SIGNALS) | frozenset(extra_signals)
    rules_raw = raw.get("rules")
    if not isinstance(rules_raw, list) or not rules_raw:
        raise ConfigError("rules: ожидался непустой список правил")

    sev = glob.get("severity_factor") or {"high": 1.0, "medium": 0.97, "low": 0.93}
    cfg = RulesConfig(
        version=int(raw.get("version", 1)),
        rules=[],
        signal_defaults={str(k): float(v) for k, v in (raw.get("signal_defaults") or {}).items()},
        default_value=_num(glob, "default_value", 0.0, where="global."),
        ffill_max_s=_num(glob, "ffill_max_s", 1.5, 0, 60, "global."),
        merge_policy=str(glob.get("merge_policy", "max")),
        derive_from_tracks=bool(glob.get("derive_from_tracks", True)),
        presence_tol_s=_num(glob, "presence_tol_s", 0.6, 0, 10, "global."),
        bbox_tol_s=_num(glob, "bbox_tol_s", 1.0, 0, 10, "global."),
        max_bboxes=int(_num(glob, "max_bboxes", 6, 0, 50, "global.")),
        nms_window_s=_num(glob, "nms_window_s", 5.0, 0, 600, "global."),
        same_type_merge_gap_s=_num(glob, "same_type_merge_gap_s", 2.0, 0, 600, "global."),
        corroboration_bonus=_num(glob, "corroboration_bonus", 0.05, 0, 0.5, "global."),
        max_events=int(_num(glob, "max_events", 12, 1, 1000, "global.")),
        min_confidence=_num(glob, "min_confidence", 0.2, 0, 1, "global."),
        severity_factor={str(k): float(v) for k, v in sev.items()},
        known_signals=known,
        source_path=source_path,
        warnings=warnings,
    )
    if cfg.merge_policy not in ("max", "min", "last"):
        raise ConfigError(f"global.merge_policy: {cfg.merge_policy!r} (ожидалось max|min|last)")

    seen: set[str] = set()
    for i, item in enumerate(rules_raw):
        if not isinstance(item, Mapping):
            raise ConfigError(f"rules[{i}]: ожидался словарь")
        r = _deep_merge(defaults, item)
        rid = str(r.get("id") or "").strip()
        where = f"rules[{rid or i}]."
        if not rid:
            raise ConfigError(f"rules[{i}]: нет id")
        if rid in seen:
            raise ConfigError(f"Дублирующийся id правила: {rid}")
        seen.add(rid)
        type_id = str(r.get("type_id") or "")
        if not TYPE_ID_RE.match(type_id):
            raise ConfigError(f"{where}type_id={type_id!r}: ожидалось V01..V99")
        try:
            strength = compile_expr(r.get("strength"))
            gate = compile_expr(r["gate"]) if r.get("gate") not in (None, "") else None
        except ExprError as exc:
            raise ConfigError(f"{where}: {exc}") from None
        for nm in sorted(strength.names | (gate.names if gate else frozenset())):
            if nm not in known:
                warnings.append(f"{rid}: сигнал {nm!r} не входит в SIGNALS/extra_signals")
        tp = r.get("t_peak") or {}
        if isinstance(tp, str):
            tp = {"policy": tp}
        policy = str(tp.get("policy", "onset_plus"))
        if policy not in T_PEAK_POLICIES:
            raise ConfigError(f"{where}t_peak.policy={policy!r} (ожидалось {'|'.join(T_PEAK_POLICIES)})")
        smooth = r.get("smooth") or {}
        if isinstance(smooth, str):
            smooth = {"method": smooth}
        method = str(smooth.get("method", "median"))
        if method not in SMOOTH_METHODS:
            raise ConfigError(f"{where}smooth.method={method!r}")
        conf = r.get("confidence") or {}
        conf_expr = None
        if conf.get("expr"):
            try:
                conf_expr = compile_expr(conf["expr"], allowed_names=set(CONF_VARS))
            except ExprError as exc:
                raise ConfigError(f"{where}confidence.expr: {exc}") from None
        on = _num(r, "on", 0.5, 0, 1, where)
        off = _num(r, "off", min(on, 0.35), 0, 1, where)
        if off > on:
            raise ConfigError(f"{where}off={off} > on={on}")
        facts = [str(f) for f in (r.get("facts") or [])]
        for tmpl in facts:
            _template_fields(tmpl)
        spec = RuleSpec(
            id=rid,
            type_id=type_id,
            comment=str(r.get("comment") or "").strip(),
            strength=strength,
            gate=gate,
            enabled=bool(r.get("enabled", True)),
            on=on,
            off=off,
            smooth_method=method,
            smooth_s=_num(smooth, "window_s", 0.6, 0, 30, where + "smooth."),
            min_duration_s=_num(r, "min_duration_s", 1.5, 0, 3600, where),
            merge_gap_s=_num(r, "merge_gap_s", 1.5, 0, 600, where),
            t_peak_policy=policy,
            t_peak_offset_s=_num(tp, "offset_s", 0.5, -30, 30, where + "t_peak."),
            conf_base=_num(conf, "base", 0.35, 0, 1, where + "confidence."),
            conf_w_strength=_num(conf, "w_strength", 0.35, 0, 1, where + "confidence."),
            conf_w_duration=_num(conf, "w_duration", 0.25, 0, 1, where + "confidence."),
            conf_tau_s=_num(conf, "tau_s", 6.0, 0.01, 3600, where + "confidence."),
            conf_min=_num(conf, "min", 0.05, 0, 1, where + "confidence."),
            conf_max=_num(conf, "max", 0.97, 0, 1, where + "confidence."),
            conf_expr=conf_expr,
            weight=_num(r, "weight", 1.0, 0, 2, where),
            facts=facts,
            highlight=[str(h) for h in (r.get("highlight") or [])],
            yield_to=[str(y) for y in (r.get("yield_to") or [])],
            yield_factor=_num(r, "yield_factor", 0.7, 0, 1, where),
            max_events=int(_num(r, "max_events", 3, 1, 100, where)),
        )
        cfg.rules.append(spec)
    return cfg


_CACHE: dict[tuple[str, float], RulesConfig] = {}


def load_rules_config(path: Optional[str | os.PathLike] = None, overrides: Sequence[str] = ()) -> RulesConfig:
    """Загрузить rules.yaml (кэш по пути+mtime, если нет переопределений)."""
    p = Path(path or os.environ.get(ENV_RULES_PATH) or DEFAULT_RULES_PATH)
    key = (str(p.resolve()), p.stat().st_mtime)
    if not overrides and key in _CACHE:
        return _CACHE[key]
    raw, src = read_raw_config(p)
    if overrides:
        raw = apply_overrides(raw, overrides)
    cfg = parse_rules_config(raw, source_path=src)
    if not overrides:
        _CACHE[key] = cfg
    return cfg
