#!/usr/bin/env python3
"""Команда замера для L2 (локально, реальные данные; владелец A01). Облака нет, GPU не нужен.

Берёт уже готовые результаты пайплайна (по одному JSON на видео: UI-report / ``result_to_api`` / сериализованный
``AnalysisResult``, например ``preds_raw`` из прогона eval A16) и считает, что увидит жюри после строгого
сериализатора ``result_to_jury_api`` (production-режим: mock не допускается)::

    python scripts/l2_contract_check.py --preds <папка с JSON> --labels <labels.csv> \\
        --out reports/a01_l2_contract.json [--compare-set jury.embed_rule_in_description=true] \\
        [--repeat <вторая папка того же прогона>]

Метрики: доля валидных строгих ответов жюри, статусы и причины ``failed`` (в т.ч. «нет meta.coverage»),
FP/FN/«failed» относительно разметки, таймкод (первое нарушение против секунды разметки: |Δ| медиана/p90,
попадание ±1/±2/±3 с, знаковый сдвиг — для подбора политики t_peak замером), latency (``meta.timings.total``),
пик VRAM (``meta.vram_events``, если есть), воспроизводимость (sha256 строгого ответа между прогонами).
``--compare-set k=v`` — одна гипотеза за раз: тот же расчёт с изменённым флагом ``contracts/config.yaml``.

Разметка: CSV ``video,label,second`` (``label``: нарушение / без нарушения / violation / no_violation).
Имена нормализуются: кириллическая «Х» -> латинская «X», расширение и регистр отбрасываются.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contracts.serialize import (  # noqa: E402
    _load_config_cached,
    api_to_result,
    fingerprint,
    resolve_outcome,
    result_to_jury_api,
)
from contracts.validate import jury_response_errors  # noqa: E402

_CYR = str.maketrans({"Х": "X", "х": "x", "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O",
                      "Р": "P", "С": "C", "Т": "T"})


def norm_id(name: str) -> str:
    stem = Path(str(name).strip()).stem if "." in Path(str(name).strip()).name else str(name).strip()
    return stem.translate(_CYR).upper().replace(" ", "")


def read_labels(path: Optional[str]) -> dict[str, dict]:
    if not path:
        return {}
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            label = (row.get("label") or "").strip().lower()
            viol = label in ("нарушение", "violation", "1", "true", "да")
            sec = (row.get("second") or "").strip()
            out[norm_id(row.get("video", ""))] = {"violation": viol, "second": float(sec) if sec else None}
    return out


def load_result(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "result" in data and isinstance(data["result"], dict):
        data = data["result"]
    return api_to_result(data)


def _vram_peak(meta: dict) -> Optional[float]:
    best = None
    for ev in meta.get("vram_events") or []:
        if isinstance(ev, dict):
            for k in ("peak_mb", "max_allocated_mb", "allocated_mb", "reserved_mb"):
                v = ev.get(k)
                if isinstance(v, (int, float)):
                    best = v if best is None else max(best, v)
    return best


def evaluate(preds: Path, labels: dict[str, dict]) -> dict[str, Any]:
    rows = []
    for f in sorted(preds.glob("*.json")):
        try:
            res = load_result(f)
        except Exception as e:  # noqa: BLE001
            rows.append({"video": norm_id(f.name), "load_error": f"{type(e).__name__}: {e}"})
            continue
        jury = result_to_jury_api(res, allow_mock=False)
        oc = resolve_outcome(res, allow_mock=False)
        meta = res.meta or {}
        timings = meta.get("timings") if isinstance(meta.get("timings"), dict) else {}
        rows.append({
            "video": norm_id(f.name), "status": jury["status"], "verdict": jury["verdict"],
            "reason": oc.error, "jury_errors": jury_response_errors(jury),
            "timecodes": [v["timecode_sec"] for v in jury["violations"]],
            "types": [v.get("type_id") for v in oc.violations], "n_frames": [len(v["evidence_frames"])
                                                                           for v in jury["violations"]],
            "has_norm_in_description": [("adilet.zan.kz" in v["description"]) for v in jury["violations"]],
            "latency_s": timings.get("total"), "vram_peak": _vram_peak(meta),
            "coverage": (meta.get("coverage") or {}).get("ratio") if isinstance(meta.get("coverage"), dict)
            else meta.get("coverage"),
            "fingerprint": fingerprint(jury),
        })
    return {"rows": rows, "summary": summarize(rows, labels)}


def _q(xs: list[float], q: float) -> Optional[float]:
    if not xs:
        return None
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))], 3)


def summarize(rows: list[dict], labels: dict[str, dict]) -> dict[str, Any]:
    ok_rows = [r for r in rows if "load_error" not in r]
    status: dict[str, int] = {}
    reasons: dict[str, int] = {}
    for r in ok_rows:
        status[r["status"]] = status.get(r["status"], 0) + 1
        if r["status"] == "failed" and r["reason"]:
            key = r["reason"].split(":")[0][:80]
            reasons[key] = reasons.get(key, 0) + 1
    conf = {"tp": 0, "fn": 0, "fp": 0, "tn": 0, "failed_on_violation": 0, "failed_on_clean": 0, "unlabeled": 0}
    deltas: list[float] = []
    for r in ok_rows:
        gt = labels.get(r["video"])
        if gt is None:
            conf["unlabeled"] += 1
            continue
        if r["status"] != "done":
            conf["failed_on_violation" if gt["violation"] else "failed_on_clean"] += 1
        elif gt["violation"]:
            conf["tp" if r["verdict"] == "violation" else "fn"] += 1
            if r["verdict"] == "violation" and gt["second"] is not None and r["timecodes"]:
                deltas.append(min(r["timecodes"], key=lambda t: abs(t - gt["second"])) - gt["second"])
        else:
            conf["fp" if r["verdict"] == "violation" else "tn"] += 1
    absd = [abs(d) for d in deltas]
    lat = [r["latency_s"] for r in ok_rows if isinstance(r.get("latency_s"), (int, float))]
    vram = [r["vram_peak"] for r in ok_rows if isinstance(r.get("vram_peak"), (int, float))]
    n_viol = sum(len(r["timecodes"]) for r in ok_rows)
    return {
        "n": len(rows), "load_errors": len(rows) - len(ok_rows),
        "jury_json_valid": sum(1 for r in ok_rows if not r["jury_errors"]),
        "status": status, "failed_reasons": reasons, "confusion": conf,
        "timecode": {"n": len(deltas), "abs_median": _q(absd, 0.5), "abs_p90": _q(absd, 0.9),
                     "signed_median": _q(deltas, 0.5),
                     "hit_1s": sum(d <= 1.0 for d in absd), "hit_2s": sum(d <= 2.0 for d in absd),
                     "hit_3s": sum(d <= 3.0 for d in absd)},
        "latency_s": {"median": _q(lat, 0.5), "p90": _q(lat, 0.9), "max": max(lat) if lat else None},
        "vram_peak_max": max(vram) if vram else None,
        "violations_with_norm_in_description": sum(sum(r["has_norm_in_description"]) for r in ok_rows),
        "violations_total": n_viol,
    }


def _with_overrides(sets: list[str]):
    """Контекст: временный config.yaml с переопределениями ``section.key=value``."""
    import yaml

    base = yaml.safe_load((ROOT / "contracts" / "config.yaml").read_text(encoding="utf-8")) or {}
    for item in sets:
        key, _, raw = item.partition("=")
        section, _, name = key.partition(".")
        base.setdefault(section, {})[name] = yaml.safe_load(raw)
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
    yaml.safe_dump(base, tmp, allow_unicode=True)
    tmp.close()
    return tmp.name


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Замер A01 для L2: что увидит жюри после строгого сериализатора")
    ap.add_argument("--preds", required=True, help="папка: по JSON на видео (UI-report / AnalysisResult)")
    ap.add_argument("--labels", default=None, help="CSV video,label,second")
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare-set", action="append", default=[], help="гипотеза: section.key=value")
    ap.add_argument("--repeat", default=None, help="папка второго прогона — проверка воспроизводимости")
    args = ap.parse_args(argv)
    labels = read_labels(args.labels)
    report: dict[str, Any] = {"baseline": evaluate(Path(args.preds), labels)}
    if args.compare_set:
        old = os.environ.get("SENTINEL_CONTRACTS_CONFIG")
        os.environ["SENTINEL_CONTRACTS_CONFIG"] = _with_overrides(args.compare_set)
        _load_config_cached.cache_clear()
        try:
            variant = evaluate(Path(args.preds), labels)
        finally:
            if old is None:
                os.environ.pop("SENTINEL_CONTRACTS_CONFIG", None)
            else:
                os.environ["SENTINEL_CONTRACTS_CONFIG"] = old
            _load_config_cached.cache_clear()
        base_fp = {r["video"]: r.get("fingerprint") for r in report["baseline"]["rows"]}
        changed = [r["video"] for r in variant["rows"] if base_fp.get(r["video"]) != r.get("fingerprint")]
        report["variant"] = {"set": args.compare_set, "summary": variant["summary"], "changed_videos": changed}
    if args.repeat:
        rep = evaluate(Path(args.repeat), labels)
        a = {r["video"]: r["fingerprint"] for r in report["baseline"]["rows"] if "fingerprint" in r}
        b = {r["video"]: r["fingerprint"] for r in rep["rows"] if "fingerprint" in r}
        common = sorted(set(a) & set(b))
        report["reproducibility"] = {"n": len(common), "identical": sum(a[v] == b[v] for v in common),
                                     "different": [v for v in common if a[v] != b[v]]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v.get("summary", v) if isinstance(v, dict) else v for k, v in report.items()},
                     ensure_ascii=False, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
