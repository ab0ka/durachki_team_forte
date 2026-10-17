"""Протокол прогона: что именно мерили. Baseline и кандидат пишутся одной и той же процедурой.

protocol.json рядом с metrics.json: git SHA (+dirty), хэш и содержимое eval-конфига, хэш конфига пайплайна
и SENTINEL_*-флаги, версии пакетов (без импорта самих пакетов), GPU, список входов и их хэш, хэш labels.csv,
вид оценки (exploratory / holdout_frozen), режим, subset. Сравнение двух прогонов допустимо, только если
совпадают inputs_sha и labels_sha256 (eval.ablate это проверяет).

VramSampler — пик memory.used по nvidia-smi (вся видеокарта, включая другие процессы — это и есть бюджет 8 ГБ).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Optional

PROTOCOL_VERSION = 1
PACKAGES = ("torch", "torchvision", "transformers", "bitsandbytes", "accelerate", "opencv-python-headless",
            "opencv-python", "numpy", "onnxruntime", "onnxruntime-gpu", "av", "scikit-learn", "scipy", "requests",
            "PyYAML", "fastapi", "uvicorn")
FLAG_PREFIXES = ("SENTINEL_", "HF_HUB_OFFLINE", "CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF")


def sha256_bytes(b: bytes, n: int = 16) -> str:
    return hashlib.sha256(b).hexdigest()[:n]


def sha256_json(obj: Any, n: int = 16) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode(), n)


def sha256_file(path: str | Path, n: int = 16, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()[:n]


def env_versions(packages: Iterable[str] = PACKAGES) -> dict:
    from importlib import metadata

    out: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform()}
    pk = {}
    for p in packages:
        try:
            pk[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            continue
    out["packages"] = pk
    return out


def gpu_info() -> list[dict]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10, check=False)
        out = []
        for line in r.stdout.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 3:
                out.append({"name": parts[0], "memory_total_mb": float(parts[1]), "driver": parts[2]})
        return out
    except Exception:  # noqa: BLE001
        return []


def env_flags(overrides: Optional[dict] = None) -> dict:
    flags = {k: v for k, v in os.environ.items() if k.startswith(FLAG_PREFIXES)}
    flags.update({k: str(v) for k, v in (overrides or {}).items()})
    for k in list(flags):
        if any(s in k.upper() for s in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
            flags[k] = "***"           # ключи в протокол не пишем
    return dict(sorted(flags.items()))


def inputs_fingerprint(videos: list[tuple[str, Path]], labels_csv: Optional[str | Path] = None,
                       hash_files: bool = False) -> dict:
    """Хэш входов: sha256 из labels.csv, если есть, иначе (имя, размер) или содержимое файла (hash_files)."""
    known: dict[str, str] = {}
    if labels_csv and Path(labels_csv).exists():
        from .labels import read_labels_csv

        known = {r.video_id: (r.sha256 or r.md5) for r in read_labels_csv(labels_csv) if (r.sha256 or r.md5)}
    items = []
    for vid, p in sorted(videos, key=lambda x: x[0]):
        if vid in known:
            fp = known[vid]
        elif hash_files and Path(p).exists():
            fp = sha256_file(p)
        else:
            fp = f"{Path(p).name}:{Path(p).stat().st_size if Path(p).exists() else 0}"
        items.append([vid, fp])
    return {"n": len(items), "ids": [i[0] for i in items], "inputs_sha": sha256_json(items)}


def build_protocol(*, label: str, cfg: dict, videos: list[tuple[str, Path]], labels_csv: Optional[str | Path],
                   mode: str, subset: Optional[str], pipeline_config: Optional[str | Path] = None,
                   env_overrides: Optional[dict] = None, run_id: Optional[str] = None) -> dict:
    from .metrics import EVAL_VERSION, git_sha

    pc_sha = None
    if pipeline_config and Path(pipeline_config).exists():
        pc_sha = sha256_file(pipeline_config)
    kind = str((cfg.get("dataset") or {}).get("evaluation_kind") or "exploratory")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": run_id or f"{_dt.datetime.now():%Y%m%d-%H%M%S}",
        "label": label, "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "git_sha": git_sha(), "eval_version": EVAL_VERSION,
        "eval_config_sha": sha256_json(cfg), "eval_config": cfg,
        "pipeline_config": str(pipeline_config) if pipeline_config else None, "pipeline_config_sha": pc_sha,
        "pipeline_env": env_flags(env_overrides),
        "env": env_versions(), "gpu": gpu_info(),
        "inputs": inputs_fingerprint(videos, labels_csv),
        "labels_sha256": sha256_file(labels_csv) if labels_csv and Path(labels_csv).exists() else None,
        "evaluation_kind": kind, "mode": mode, "subset": subset or "all",
    }


def same_procedure(a: dict, b: dict) -> list[str]:
    """Причины, по которым два протокола НЕЛЬЗЯ сравнивать напрямую (пусто — можно)."""
    probs = []
    for k in ("labels_sha256", "mode", "subset", "eval_version"):
        if a.get(k) != b.get(k):
            probs.append(f"{k}: {a.get(k)} ≠ {b.get(k)}")
    if (a.get("inputs") or {}).get("inputs_sha") != (b.get("inputs") or {}).get("inputs_sha"):
        probs.append("inputs_sha: разные наборы видео")
    if sha256_json({k: v for k, v in (a.get("eval_config") or {}).items() if k != "runner"}) != \
            sha256_json({k: v for k, v in (b.get("eval_config") or {}).items() if k != "runner"}):
        probs.append("eval_config: разные пороги оценки")
    return probs


class VramSampler:
    """Пик memory.used (МБ) GPU 0 по nvidia-smi, опрос раз в interval_s. Без nvidia-smi — None."""

    def __init__(self, interval_s: float = 0.5, gpu_index: int = 0) -> None:
        self.interval_s, self.gpu_index = interval_s, gpu_index
        self.peak_mb: Optional[float] = None
        self._stop = threading.Event()
        self._t: Optional[threading.Thread] = None
        self.available = bool(shutil.which("nvidia-smi"))

    def _read(self) -> Optional[float]:
        try:
            r = subprocess.run(["nvidia-smi", f"--id={self.gpu_index}", "--query-gpu=memory.used",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=False)
            return float(r.stdout.strip().splitlines()[0])
        except Exception:  # noqa: BLE001
            return None

    def _loop(self) -> None:
        while not self._stop.is_set():
            v = self._read()
            if v is not None:
                self.peak_mb = v if self.peak_mb is None else max(self.peak_mb, v)
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "VramSampler":
        if self.available:
            self._t = threading.Thread(target=self._loop, daemon=True)
            self._t.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._t:
            self._t.join(timeout=5)


def torch_peak_mb(reset: bool = False) -> Optional[float]:
    """torch.cuda.max_memory_allocated — только если torch УЖЕ импортирован пайплайном (eval torch не тянет)."""
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        if reset:
            torch.cuda.reset_peak_memory_stats()
            return None
        return round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
    except Exception:  # noqa: BLE001
        return None


def wait_s(t0: float) -> float:
    return round(time.perf_counter() - t0, 3)
