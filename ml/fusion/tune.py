"""Подбор порогов вердикта по типам на размеченных данных.

    python -m ml.fusion.tune --preds preds_raw/ --labels data/labels.csv [--holdout 0.3] [--cv 5]
                             [--target-recall 1.0] [--out ml/fusion/thresholds.yaml]

* Пороги подбираются ТОЛЬКО на train-части; метрики — на отложенной калибровочной части (holdout),
  дубликаты (одинаковые входы, напр. Х-4≡Х-47) всегда попадают в одну часть. ``--cv K`` дополнительно
  оценивает саму процедуру grouped K-fold. Подбор на всей выборке — только явно (``--fit-all``), с пометкой.
* Цель на train: полнота ≥ target-recall (пропуск недопустим), затем минимум ложных срабатываний;
  внутри «плато» равных решений берём точку ближе к нижней границе (запас по полноте на новых видео).
* Вход ``--preds``: JSON на видео (имя файла = id ролика): ответ API/UI-отчёт с ``meta.scores_raw``,
  сам ``AnalysisResult.meta`` или голый ``scores_raw``. Если в нём есть ``inputs`` — слияние пересчитывается
  текущим config.yaml (тюнинг конфигурации без повторного inference).
* Скоры — эвристические (источники коррелированы), не вероятности: пороги — точки отсечения, не «P≥τ».
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from .catalog import resolve_catalog
from .config import load_config, load_thresholds, threshold_for
from .core import fuse_scores

GRID = [round(0.05 + 0.01 * i, 2) for i in range(91)]  # 0.05..0.95
ID_COLS = ("video", "file", "filename", "name", "video_id", "id", "ролик", "видео", "файл", "имя")
LABEL_COLS = ("label", "violation", "verdict", "is_violation", "class", "target", "нарушение", "разметка",
              "метка", "результат")
TIME_COLS = ("timecode", "timecode_sec", "time", "second", "sec", "t", "event_sec", "секунда", "время", "таймкод")


# --------------------------------------------------------------------------- разбор входов


def normalize_id(name: str) -> str:
    """'Х-3.mov' (кириллица) / 'x_3' / 'X–3' → 'X-3'."""
    s = Path(str(name).strip()).name
    s = re.sub(r"\.(mov|mp4|avi|mkv|json)$", "", s, flags=re.IGNORECASE)
    s = s.replace("Х", "X").replace("х", "x").upper()
    s = re.sub(r"[\s_–—]+", "-", s)
    return re.sub(r"-+", "-", s).strip("-")


def parse_label(value: Any) -> Optional[bool]:
    s = str(value or "").strip().lower()
    if not s:
        return None
    if "без" in s or s in {"0", "no", "нет", "false", "no_violation", "clean", "чисто", "норма", "ok"}:
        return False
    if s in {"1", "yes", "да", "true", "violation"} or "наруш" in s or "violation" in s:
        return True
    return None


def parse_seconds(value: Any) -> Optional[float]:
    s = str(value or "").strip().lower().replace(",", ".")
    m = re.search(r"(\d+):(\d+(?:\.\d+)?)", s)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    m = re.search(r"\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else None


@dataclass
class Label:
    video: str
    positive: bool
    second: Optional[float] = None


def read_labels(path: str | Path) -> dict[str, list[Label]]:
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "cp1251"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    rows = list(csv.reader(io.StringIO(text), dialect))
    if not rows:
        return {}
    header = [h.strip().lower() for h in rows[0]]

    def col(names: Sequence[str]) -> Optional[int]:
        for n in names:
            if n in header:
                return header.index(n)
        return None

    has_header = any(h in header for h in ID_COLS + LABEL_COLS + TIME_COLS)
    i_id = col(ID_COLS) if has_header else None
    i_lab = col(LABEL_COLS) if has_header else None
    i_t = col(TIME_COLS) if has_header else None
    i_id = 0 if i_id is None else i_id
    i_lab = (1 if len(header) > 1 else 0) if i_lab is None else i_lab
    out: dict[str, list[Label]] = {}
    for row in rows[1:] if has_header else rows:
        if not row or len(row) <= max(i_id, i_lab):
            continue
        pos = parse_label(row[i_lab])
        if pos is None:
            continue
        sec = parse_seconds(row[i_t]) if i_t is not None and i_t < len(row) else None
        if sec is None and pos:
            sec = parse_seconds(re.sub(r"^\D*", "", row[i_lab])) if re.search(r"\d", row[i_lab]) else None
        vid = normalize_id(row[i_id])
        out.setdefault(vid, []).append(Label(vid, pos, sec if pos else None))
    return out


def _find_scores_raw(obj: Any) -> Optional[dict[str, Any]]:
    if not isinstance(obj, dict):
        return None
    if "candidates" in obj or "inputs" in obj:
        return obj
    meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else obj
    raw = meta.get("scores_raw") if isinstance(meta, dict) else None
    return raw if isinstance(raw, dict) else None


@dataclass
class VideoPred:
    video: str
    cands: list[tuple[str, float, float]] = field(default_factory=list)   # (type_id, score, t_peak) — гейт пройден
    fingerprint: str = ""


def load_preds(preds_dir: str | Path, cfg: dict[str, Any]) -> dict[str, VideoPred]:
    types, _ = resolve_catalog(None)
    zero = {"default": 0.0, "per_type": {}}
    out: dict[str, VideoPred] = {}
    for p in sorted(Path(preds_dir).glob("*.json")):
        try:
            raw = _find_scores_raw(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
        if raw is None:
            continue
        vid = normalize_id(p.stem)
        inputs = raw.get("inputs")
        fp_src = json.dumps(inputs if inputs else raw.get("candidates"), sort_keys=True, ensure_ascii=False)
        pred = VideoPred(vid, fingerprint=hashlib.md5(fp_src.encode("utf-8")).hexdigest())
        if isinstance(inputs, dict):
            outcome = fuse_scores(inputs.get("events") or [], inputs.get("risk") or [],
                                  float(raw.get("duration") or 0.0), types, cfg, zero)
            for c in outcome.candidates:
                if not c.gate and c.suppressed_by is None and c.confidence > 0:
                    pred.cands.append((c.type_id, c.confidence, c.t_peak))
        else:
            for c in raw.get("candidates") or []:
                if not c.get("gate") and not c.get("suppressed_by") and float(c.get("confidence", 0)) > 0:
                    pred.cands.append((str(c["type_id"]), float(c["confidence"]), float(c.get("t_peak", 0.0))))
        out[vid] = pred
    return out


# --------------------------------------------------------------------------- оценка и подбор


@dataclass
class Sample:
    video: str
    positive: bool
    seconds: list[float]
    cands: list[tuple[str, float, float]]
    group: str


def predict(s: Sample, thr: dict[str, float], default: float) -> list[tuple[str, float, float]]:
    return [c for c in s.cands if c[1] >= thr.get(c[0], default)]


def evaluate(samples: Sequence[Sample], thr: dict[str, float], default: float, tol: float = 3.0
             ) -> dict[str, Any]:
    tp = fn = fp = tn = tc_hit = tc_total = 0
    missed, false_alarms = [], []
    for s in samples:
        hits = predict(s, thr, default)
        if s.positive:
            if hits:
                tp += 1
                if s.seconds:
                    tc_total += 1
                    if any(min(abs(h[2] - sec), abs(h[2] - (sec + 1.0))) <= tol or sec <= h[2] <= sec + 1.0
                           for h in hits for sec in s.seconds):
                        tc_hit += 1
            else:
                fn += 1
                missed.append(s.video)
        elif hits:
            fp += 1
            false_alarms.append(s.video)
        else:
            tn += 1
    pos, neg = tp + fn, fp + tn
    return {
        "n": len(samples), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "recall": round(tp / pos, 4) if pos else None,
        "fp_rate": round(fp / neg, 4) if neg else None,
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "timecode_hit_rate": round(tc_hit / tc_total, 4) if tc_total else None,
        "missed": missed, "false_alarms": false_alarms,
    }


def _cost(samples: Sequence[Sample], thr: dict[str, float], default: float, allowed_fn: int) -> tuple[int, int]:
    m = evaluate(samples, thr, default)
    return (max(0, m["fn"] - allowed_fn), m["fp"])


def fit(samples: Sequence[Sample], start: dict[str, float], default: float, target_recall: float = 1.0,
        bias: float = 0.35, passes: int = 4) -> dict[str, float]:
    """Покоординатный спуск по порогам типов: (пропуски сверх допустимых, ложные) → min; внутри плато — bias."""
    n_pos = sum(1 for s in samples if s.positive)
    allowed_fn = int(math.floor((1.0 - target_recall) * n_pos + 1e-9))
    types = sorted({c[0] for s in samples for c in s.cands})
    thr = dict(start)
    for _ in range(passes):
        changed = False
        for t in types:
            costs = []
            for g in GRID:
                trial = dict(thr)
                trial[t] = g
                costs.append((_cost(samples, trial, default, allowed_fn), g))
            best = min(c for c, _ in costs)
            plateau = [g for c, g in costs if c == best]
            # плато может быть несвязным — берём связный участок, содержащий минимальный порог плато
            run = [plateau[0]]
            for g in plateau[1:]:
                if abs(g - run[-1] - 0.01) < 1e-6:
                    run.append(g)
                else:
                    break
            choice = run[int(round(bias * (len(run) - 1)))]
            if abs(thr.get(t, default) - choice) > 1e-9:
                thr[t] = choice
                changed = True
        if not changed:
            break
    return thr


def split_groups(samples: Sequence[Sample], holdout: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    """Стратифицированный по классу сплит групп (дубли — в одной части)."""
    rng = random.Random(seed)
    groups: dict[str, list[Sample]] = {}
    for s in samples:
        groups.setdefault(s.group, []).append(s)
    train, hold = [], []
    for positive in (True, False):
        keys = sorted(k for k, g in groups.items() if any(x.positive for x in g) == positive)
        rng.shuffle(keys)
        n_hold = int(round(holdout * len(keys)))
        if holdout > 0 and len(keys) >= 3:
            n_hold = max(1, n_hold)
        for i, k in enumerate(keys):
            (hold if i < n_hold else train).extend(groups[k])
    return train, hold


def kfold(samples: Sequence[Sample], k: int, seed: int) -> list[tuple[list[Sample], list[Sample]]]:
    rng = random.Random(seed)
    groups: dict[str, list[Sample]] = {}
    for s in samples:
        groups.setdefault(s.group, []).append(s)
    folds: list[list[str]] = [[] for _ in range(k)]
    for positive in (True, False):
        keys = sorted(g for g, v in groups.items() if any(x.positive for x in v) == positive)
        rng.shuffle(keys)
        for i, key in enumerate(keys):
            folds[i % k].append(key)
    out = []
    for i in range(k):
        test_keys = set(folds[i])
        train = [s for g, v in groups.items() if g not in test_keys for s in v]
        test = [s for g in folds[i] for s in groups[g]]
        out.append((train, test))
    return out


def build_samples(preds: dict[str, VideoPred], labels: dict[str, list[Label]]) -> tuple[list[Sample], list[str]]:
    samples, warnings = [], []
    for vid, labs in sorted(labels.items()):
        pred = preds.get(vid)
        if pred is None:
            warnings.append(f"нет предсказаний для {vid}")
            continue
        positive = any(lab.positive for lab in labs)
        secs = [lab.second for lab in labs if lab.positive and lab.second is not None]
        samples.append(Sample(vid, positive, secs, pred.cands, pred.fingerprint or vid))
    for vid in sorted(set(preds) - set(labels)):
        warnings.append(f"нет разметки для {vid}")
    return samples, warnings


# --------------------------------------------------------------------------- CLI


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Подбор порогов ml.fusion (train/holdout, без подгонки под всю выборку)")
    ap.add_argument("--preds", required=True, help="папка с JSON-ответами/scores_raw по видео")
    ap.add_argument("--labels", required=True, help="CSV разметки (id ролика, нарушение/без, секунда)")
    ap.add_argument("--holdout", type=float, default=0.3, help="доля групп в калибровочной части (0..0.9)")
    ap.add_argument("--cv", type=int, default=0, help="grouped K-fold оценка процедуры (0 — нет)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--target-recall", type=float, default=1.0)
    ap.add_argument("--bias", type=float, default=0.35, help="позиция внутри плато (0 — макс. запас полноты)")
    ap.add_argument("--tol", type=float, default=3.0, help="допуск таймкода, с")
    ap.add_argument("--fit-all", action="store_true", help="подобрать на ВСЕЙ выборке (без честной оценки)")
    ap.add_argument("--out", default="", help="куда записать YAML (по умолчанию — только stdout)")
    ap.add_argument("--report", default="", help="куда записать JSON-отчёт")
    args = ap.parse_args(argv)

    cfg = load_config()
    current = load_thresholds()
    default = float(current.get("default", 0.45))
    start = {t: threshold_for(current, t) for t in (current.get("per_type") or {})}
    preds = load_preds(args.preds, cfg)
    labels = read_labels(args.labels)
    samples, warnings = build_samples(preds, labels)
    if not samples:
        print("нет пересечения предсказаний и разметки", file=sys.stderr)
        return 2
    report: dict[str, Any] = {"warnings": warnings, "n_samples": len(samples),
                              "baseline_all": evaluate(samples, start, default, args.tol)}

    if args.cv and args.cv >= 2:
        folds = []
        for train, test in kfold(samples, args.cv, args.seed):
            thr = fit(train, start, default, args.target_recall, args.bias)
            folds.append(evaluate(test, thr, default, args.tol))
        report["cv"] = {
            "k": args.cv,
            "fn": sum(f["fn"] for f in folds), "fp": sum(f["fp"] for f in folds),
            "tp": sum(f["tp"] for f in folds), "tn": sum(f["tn"] for f in folds),
            "missed": sorted(v for f in folds for v in f["missed"]),
            "false_alarms": sorted(v for f in folds for v in f["false_alarms"]),
        }

    if args.fit_all:
        train, hold = list(samples), []
        report["note"] = "пороги подобраны на ВСЕЙ выборке: оценка ниже оптимистична"
    else:
        train, hold = split_groups(samples, args.holdout, args.seed)
    thr = fit(train, start, default, args.target_recall, args.bias)
    report["train"] = evaluate(train, thr, default, args.tol)
    report["holdout"] = evaluate(hold, thr, default, args.tol) if hold else None
    report["train_videos"] = sorted(s.video for s in train)
    report["holdout_videos"] = sorted(s.video for s in hold)

    per_type = {t: round(v, 2) for t, v in sorted({**start, **thr}.items())}
    doc = {
        "version": 1,
        "tuned": True,
        "default": default,
        "per_type": per_type,
        "tuned_on": {
            "mode": "fit_all" if args.fit_all else "train_split",
            "seed": args.seed, "holdout": 0.0 if args.fit_all else args.holdout,
            "target_recall": args.target_recall, "bias": args.bias,
            "n_train": len(train), "n_holdout": len(hold),
            "holdout_metrics": {k: v for k, v in (report["holdout"] or {}).items()
                                if k not in ("missed", "false_alarms")},
            "cv": {k: v for k, v in (report.get("cv") or {}).items() if k not in ("missed", "false_alarms")},
            "score_note": "эвристические скоры коррелированных источников, не вероятности",
        },
    }
    try:
        import yaml  # lazy

        text = yaml.safe_dump(doc, allow_unicode=True, sort_keys=False)
    except ImportError:
        text = json.dumps(doc, ensure_ascii=False, indent=2)
    print(text)
    print(json.dumps({k: report[k] for k in ("train", "holdout", "cv") if k in report}, ensure_ascii=False,
                     indent=2), file=sys.stderr)
    if args.out:
        Path(args.out).write_text("# Сгенерировано python -m ml.fusion.tune\n" + text, encoding="utf-8")
    if args.report:
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
