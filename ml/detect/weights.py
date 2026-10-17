"""Кэш весов: SENTINEL_WEIGHTS (по умолчанию <repo>/weights).

Скачивание — ТОЛЬКО на этапе setup (с интернетом):  python -m ml.detect.weights --download
Импорт и инференс никогда не ходят в сеть (from_pretrained(..., local_files_only=True)); HF_HUB_OFFLINE=1
рекомендуется выставлять в рантайме дополнительно.
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
    # HF_HUB_CACHE (как у deploy/weights.py) → стандартная раскладка weights/hf/hub → старая weights/hf.
    env = os.environ.get("HF_HUB_CACHE")
    if env and Path(env).expanduser().is_dir():
        return Path(env).expanduser()
    hub = weights_dir() / "hf" / "hub"
    return hub if hub.is_dir() else weights_dir() / "hf"


def is_offline() -> bool:
    return env_flag("HF_HUB_OFFLINE") or env_flag("TRANSFORMERS_OFFLINE") or env_flag("SENTINEL_OFFLINE")


def local_model_dir(model_id: str) -> Optional[Path]:
    """Если модель сохранена через save_pretrained в weights/detect/<name> — используем её напрямую."""
    name = model_id.rstrip("/").split("/")[-1]
    for cand in (Path(model_id), weights_dir() / "detect" / name, weights_dir() / name):
        if cand.is_dir() and (cand / "config.json").exists():
            return cand
    return None


def is_cached(model_id: str) -> bool:
    """Есть ли модель локально (save_pretrained-папка или снапшот в weights/hf) — без сети."""
    if local_model_dir(model_id) is not None:
        return True
    snap = hf_cache_dir() / ("models--" + model_id.replace("/", "--")) / "snapshots"
    return snap.is_dir() and any(snap.iterdir())


def resolve(path_rel: str) -> Path:
    p = Path(path_rel)
    return p if p.is_absolute() else weights_dir() / p


def pretrained_kwargs() -> dict:
    """Для инференса: ТОЛЬКО локальные файлы. Скачивание — исключительно на этапе setup (download_all)."""
    return {"cache_dir": str(hf_cache_dir()), "local_files_only": True}


def model_ids(include_optional: bool = False) -> list[str]:
    from .config import apply_env_extras

    cfg = apply_env_extras(load_config())
    ids = [cfg["closed_set"]["hf_model_id"], *cfg["closed_set"].get("hf_fallback_ids", [])]
    if include_optional or cfg["open_vocab"].get("enabled", False):
        ids.append(cfg["open_vocab"]["hf_model_id"])
    return ids


def download_all(ids: Optional[list[str]] = None) -> list[Path]:
    """Скачивает снапшоты моделей в weights/hf (нужен интернет один раз)."""
    from huggingface_hub import snapshot_download

    out = []
    for mid in ids or model_ids(include_optional=True):
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
            print(mid, "local_dir:", local_model_dir(mid), "in_cache:", is_cached(mid))


if __name__ == "__main__":
    main()
