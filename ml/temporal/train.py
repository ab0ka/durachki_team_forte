"""Обучение на всех размеченных видео и сохранение чекпоинтов в weights/temporal/<kind>/.

    python -m ml.temporal.train [--kind baseline|temporal|both] [--select auto|baseline|temporal]
                                [--threshold auto|0.6] [--cv-report reports/temporal_cv.json]

Порог берётся из отчёта CV (deploy_threshold выбранной модели) или из config.yaml (infer.event_threshold).
Голова типа учится только при наличии проверенного type_id в labels.csv.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional, Sequence

from .config import load_config, reports_dir, weights_dir
from .dataset import Sample, fit_spec, prepare_samples
from .features import TextBank
from .predictors import fit_predictor, has_type_labels, save_checkpoint

log = logging.getLogger("sentinel.temporal")


def train_all(samples: Sequence[Sample], bank: TextBank, cfg: dict, kinds: Sequence[str], out_dir: Path,
              select: str = "auto", threshold: str = "auto", cv_report: Optional[dict] = None,
              epochs: Optional[int] = None) -> dict:
    t0 = time.monotonic()
    deadline = t0 + 60.0 * float(cfg["train"].get("max_minutes", 8.0))
    spec = fit_spec([fs for s in samples for fs in s.versions], bank, cfg)
    prepare_samples(list(samples), bank, spec)
    info: dict = {"kinds": {}, "n_samples": len(samples), "n_pos": sum(s.label for s in samples),
                  "type_labels": has_type_labels(samples)}
    for kind in kinds:
        model = fit_predictor(kind, samples, spec, cfg, seed=int(cfg["train"]["seed"]), epochs=epochs,
                              deadline=deadline)
        thr, src = float(cfg["infer"]["event_threshold"]), "config.infer.event_threshold"
        if threshold not in ("auto", "", None):
            thr, src = float(threshold), "cli"
        elif cv_report and kind in cv_report:
            thr, src = float(cv_report[kind]["deploy_threshold"]), "cv_report.deploy_threshold (по всем OOF)"
        meta = {"threshold": thr, "threshold_source": src, "n_train": len(samples),
                "train_groups": sorted(s.group for s in samples),
                "prompts": {"en": bank.texts_en, "ru": bank.texts_ru, "owner": bank.owner},
                "cv": ({k: cv_report[kind][k] for k in ("threshold_free", "nested", "protocol", "outer", "inner")}
                       if cv_report and kind in cv_report else None)}
        d = save_checkpoint(out_dir, model, spec, meta)
        info["kinds"][kind] = {"dir": str(d), "threshold": thr, "threshold_source": src,
                               "type_head_trained": bool(model.type_trained)}
        log.info("train: %s сохранён в %s (порог %.3f, %s)", kind, d, thr, src)
    if select == "auto":
        select = (cv_report or {}).get("selected") or ("baseline" if "baseline" in kinds else kinds[0])
        if select not in kinds:
            select = kinds[0]
    (out_dir / "selected.json").write_text(json.dumps({"kind": select}), encoding="utf-8")
    info["selected"] = select
    info["elapsed_s"] = round(time.monotonic() - t0, 1)
    return info


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="ml.temporal: обучение")
    ap.add_argument("--index", default=None)
    ap.add_argument("--kind", default="both", choices=["baseline", "temporal", "both"])
    ap.add_argument("--select", default="auto", choices=["auto", "baseline", "temporal"])
    ap.add_argument("--threshold", default="auto")
    ap.add_argument("--cv-report", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-aug", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config()
    from .data_index import bank_for_index, load_samples

    samples, idx = load_samples(cfg, a.index, max_aug=a.max_aug)
    bank = bank_for_index(cfg, idx)
    kinds = ["baseline", "temporal"] if a.kind == "both" else [a.kind]
    if "temporal" in kinds:
        try:
            import torch  # noqa: F401
        except ImportError:
            log.warning("torch не установлен — обучаю только baseline")
            kinds = [k for k in kinds if k != "temporal"] or ["baseline"]
    rep_path = Path(a.cv_report) if a.cv_report else reports_dir(cfg) / "temporal_cv.json"
    cv_report = json.loads(rep_path.read_text(encoding="utf-8")) if rep_path.exists() else None
    out = Path(a.out) if a.out else weights_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    info = train_all(samples, bank, cfg, kinds, out, a.select, a.threshold, cv_report, a.epochs)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
