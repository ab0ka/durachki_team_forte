"""Извлечение признаков для обучения (один раз, на локальной машине с видео).

    python -m ml.temporal.extract --videos data/raw --labels data/labels.csv [--n-aug 2] [--mock] [--force]

Для каждого видео: md5 (дубликаты по md5 извлекаются один раз и попадают в одну группу), оригинал + n_aug
копий «другой угол» (фиксированная случайная перспектива на всё видео). Кэш — cache/features/<sha>.npz,
индекс — cache/features/index.json (в git не попадает: cache/ в .gitignore).
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from .augment import make_augmenter
from .config import cache_dir, feature_signature, load_config, stable_hash
from .features import FeatureSet, cache_path, extract_clip, get_backend, get_text_bank
from .labels import assign_groups, file_md5, find_videos, read_labels

log = logging.getLogger("sentinel.temporal")


def extract_dataset(videos_dir: str, labels_csv: str, cfg: dict, n_aug: Optional[int] = None,
                    force: bool = False, backend_name: Optional[str] = None, limit: Optional[int] = None,
                    index_out: Optional[str] = None) -> dict:
    t_all = time.monotonic()
    backend = get_backend(cfg, force=backend_name)
    get_text_bank(cfg, backend)  # закэшировать текстовые эмбеддинги заранее
    n_aug = int(cfg["augment"]["n_aug"] if n_aug is None else n_aug)
    labels = read_labels(labels_csv, cfg["labels"].get("time_offset_s", 0.0))
    vids = find_videos(videos_dir)
    missing = sorted(n for n in labels if n not in vids)
    unlabeled = sorted(n for n in vids if n not in labels)
    if missing:
        log.warning("extract: в разметке, но нет видео: %s", missing)
    if unlabeled:
        log.warning("extract: видео без разметки (пропущены): %s", unlabeled)
    names = sorted((n for n in labels if n in vids), key=lambda s: (len(s), s))[: limit or None]
    md5s = {n: file_md5(vids[n]) for n in names}
    sub = {n: labels[n] for n in names}
    assign_groups(sub, md5s, cfg["labels"].get("duplicate_groups", []))
    sig = feature_signature(cfg, backend.name)
    f = cfg["features"]
    on_cuda = str(getattr(backend, "device", "cpu")).startswith("cuda") or backend.name == "mock"
    fps = float(f["fps"])  # для обучения всегда полная частота и все виды (CPU — просто дольше)
    views = list(f["views"])
    by_md5: dict[str, list[str]] = {}
    entries = []
    for i, n in enumerate(names):
        lab, h = sub[n], md5s[n]
        if h in by_md5:  # точный дубликат — признаки уже есть
            feats = by_md5[h]
        else:
            feats = []
            clip = None
            for k in range(n_aug + 1):
                key = {"md5": h, "sig": sig, "fps": fps, "views": views, "aug": k,
                       "aug_cfg": cfg["augment"] if k else None}
                p = cache_path(cfg, key)
                if not p.exists() or force:
                    if clip is None:
                        from ._video import load_clip

                        clip = load_clip(str(vids[n]), sample_fps=fps)
                    aug = make_augmenter(int(stable_hash([h, k], 8), 16), cfg) if k else None
                    t0 = time.monotonic()
                    fs = extract_clip(clip, backend, cfg, augmenter=aug, fps=fps, views=views)
                    fs.meta.update({"aug": k, "md5": h})
                    fs.save(p)
                    log.info("extract [%d/%d] %s aug=%d: %d кадров за %.1f с", i + 1, len(names), n, k, len(fs),
                             time.monotonic() - t0)
                feats.append(str(p.relative_to(cache_dir(cfg))) if p.is_relative_to(cache_dir(cfg)) else str(p))
            by_md5[h] = feats
        entries.append({"name": n, "file": vids[n].name, "md5": h, "label": lab.label, "times": lab.times,
                        "type_id": lab.type_id, "group": lab.group, "features": feats})
    idx = {"version": 1, "backend": backend.name, "model_id": sig["model_id"], "fps": fps, "views": views,
           "n_aug": n_aug, "on_cuda": on_cuda, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "videos": entries, "missing_videos": missing, "unlabeled_videos": unlabeled,
           "groups": {g: sorted(e["name"] for e in entries if e["group"] == g)
                      for g in sorted({e["group"] for e in entries})},
           "elapsed_s": round(time.monotonic() - t_all, 1)}
    out = Path(index_out) if index_out else cache_dir(cfg) / "index.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")
    dups = {g: m for g, m in idx["groups"].items() if len(m) > 1}
    log.info("extract: %d видео, групп %d, дубликаты: %s -> %s", len(entries), len(idx["groups"]), dups, out)
    return idx


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="ml.temporal: извлечение признаков SigLIP2")
    ap.add_argument("--videos", default="data/raw")
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--n-aug", type=int, default=None)
    ap.add_argument("--mock", action="store_true", help="mock-бэкенд (без весов)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--index", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config()
    idx = extract_dataset(a.videos, a.labels, cfg, a.n_aug, a.force, "mock" if a.mock else None, a.limit, a.index)
    print(f"готово: {len(idx['videos'])} видео за {idx['elapsed_s']} с")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
