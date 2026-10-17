"""Кэш весов: SENTINEL_WEIGHTS (по умолчанию <repo>/weights), поддержка HF_HUB_OFFLINE.

Однократная предзагрузка (с интернетом):  python -m ml.detect.weights --download
Дальше всё работает офлайн:              HF_HUB_OFFLINE=1
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

from .config import env_flag, load_config

log = logging.getLogger("ml.detect.weights")

REPO_ROOT = Path(__file__).resolve().parents[2]


def weights_dir() -> Path:
    return Path(os.environ.get("SENTINEL_WEIGHTS", str(REPO_ROOT / "weights"))).expanduser()


def hf_cache_dir() -> Path:
    return weights_dir() / "hf"


def is_offline() -> bool:
    return env_flag("HF_HUB_OFFLINE") or env_flag("TRANSFORMERS_OFFLINE") or env_flag("SENTINEL_OFFLINE")


def local_model_dir(model_id: str) -> Optional[Path]:
    """Если модель сохранена через save_pretrained в weights/detect/<name> — используем её напрямую."""
    name = model_id.rstrip("/").split("/")[-1]
    for cand in (Path(model_id), weights_dir() / "detect" / name, weights_dir() / name):
        if cand.is_dir() and (cand / "config.json").exists():
            return cand
    return None


def resolve(path_rel: str) -> Path:
    p = Path(path_rel)
    return p if p.is_absolute() else weights_dir() / p


def pretrained_kwargs() -> dict:
    return {"cache_dir": str(hf_cache_dir()), "local_files_only": is_offline()}


def model_ids() -> list[str]:
    cfg = load_config()
    ids = [cfg["closed_set"]["hf_model_id"], *cfg["closed_set"].get("hf_fallback_ids", [])]
    if cfg["open_vocab"].get("enabled", True):
        ids.append(cfg["open_vocab"]["hf_model_id"])
    return ids


def download_all(ids: Optional[list[str]] = None) -> list[Path]:
    """Скачивает снапшоты моделей в weights/hf (нужен интернет один раз)."""
    from huggingface_hub import snapshot_download

    out = []
    for mid in ids or model_ids():
        log.info("download %s → %s", mid, hf_cache_dir())
        out.append(Path(snapshot_download(mid, cache_dir=str(hf_cache_dir()),
                                          allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])))
    return out


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Предзагрузка весов детектора в SENTINEL_WEIGHTS")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--ids", nargs="*", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    if a.download:
        for p in download_all(a.ids):
            print(p)
    else:
        print("weights_dir:", weights_dir(), "offline:", is_offline())
        for mid in model_ids():
            print(mid, "local:", local_model_dir(mid))


if __name__ == "__main__":
    main()
