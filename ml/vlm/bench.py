"""python -m ml.vlm.bench --frames dir/ — замер скорости и VRAM бэкенда для выбора локально.

Берёт картинки из папки (jpg/png, по порядку имён) или синтетику (--synthetic), делает N вызовов verify-промпта
по K кадров и печатает ТОЛЬКО агрегаты: время загрузки, p50/p95 задержки, доля валидного JSON, выходные токены,
пик VRAM (torch.cuda для in-process; nvidia-smi — для внешнего llama-server). Сырые ответы — в локальный дамп.
Ничего не выдумываем: поля, которые нельзя измерить на этой машине, остаются null.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional


def _nvidia_smi_used_mb() -> Optional[float]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5)
        vals = [float(x) for x in out.stdout.split() if x.strip()]
        return max(vals) if vals else None
    except Exception:  # noqa: BLE001
        return None


def _torch_peak_mb() -> Optional[float]:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 2**20, 1)
    except ImportError:
        pass
    return None


def _load_images(frames_dir: Optional[str], synthetic: int, max_side: int) -> list[Any]:
    from PIL import Image

    from .frames import resize_max_side

    if frames_dir:
        paths = sorted(p for p in Path(frames_dir).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
        return [resize_max_side(Image.open(p).convert("RGB"), max_side) for p in paths]
    import numpy as np

    rng = np.random.default_rng(0)
    imgs = []
    for _ in range(max(1, synthetic)):
        arr = (rng.random((848, 464, 3)) * 255).astype("uint8")
        imgs.append(resize_max_side(Image.fromarray(arr, "RGB"), max_side))
    return imgs


def _pct(xs: list[float], q: float) -> Optional[float]:
    if not xs:
        return None
    xs = sorted(xs)
    k = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return round(xs[k], 3)


def run(argv: Optional[list[str]] = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(prog="python -m ml.vlm.bench", description=__doc__.split("\n\n")[0])
    ap.add_argument("--frames", default=None, help="папка с кадрами (jpg/png)")
    ap.add_argument("--synthetic", type=int, default=0, help="N синтетических кадров вместо --frames")
    ap.add_argument("--n", type=int, default=5, help="число вызовов")
    ap.add_argument("--images-per-call", type=int, default=5)
    ap.add_argument("--max-side", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--preset", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--dump-dir", default=None)
    ap.add_argument("--no-dump", action="store_true")
    ap.add_argument("--out", default=None, help="записать агрегаты в JSON-файл")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    from .budget import visual_tokens
    from .catalog import catalog_types, load_catalog
    from .config import load_config, preset_cfg
    from .localdump import LocalDump
    from .parsing import extract_json, normalize_verify, verify_schema
    from .prompting import render_verify
    from .runtime import ensure_loaded, get_backend

    over: dict[str, Any] = {}
    if args.backend:
        over["backend"] = args.backend
    if args.preset:
        over["preset"] = args.preset
    cfg = load_config(args.config, overrides=over or None)
    fcfg = cfg.get("frames") or {}
    max_side = int(args.max_side or fcfg.get("max_side", 448))
    mnt = int(args.max_new_tokens or (cfg.get("generation") or {}).get("max_new_tokens_verify", 220))
    px = int(preset_cfg(cfg, "transformers").get("px_per_token", 28))
    rep: dict[str, Any] = {"backend_cfg": cfg.get("backend"), "preset": cfg.get("preset"), "max_side": max_side,
                           "images_per_call": args.images_per_call, "calls_requested": args.n}
    if not args.frames and not args.synthetic:
        rep["status"] = "no_input (use --frames DIR or --synthetic N)"
        return rep
    imgs = _load_images(args.frames, args.synthetic or args.images_per_call, max_side)
    rep["n_images_loaded"] = len(imgs)
    rep["vram_before_mb_nvidia_smi"] = _nvidia_smi_used_mb()

    be, reason = get_backend(cfg)
    if be is None:
        rep.update(status="backend_unavailable", reason=reason)
        return rep
    t0 = time.perf_counter()
    ok, why = ensure_loaded(be)
    rep["load_s"] = round(time.perf_counter() - t0, 2)
    rep["backend"] = be.info()
    if not ok:
        rep.update(status="backend_load_failed", reason=why)
        return rep
    rep["vram_after_load_mb_nvidia_smi"] = _nvidia_smi_used_mb()

    allowed = set(catalog_types(load_catalog()))
    dump = LocalDump(None if args.no_dump else (args.dump_dir or (cfg.get("privacy") or {}).get("local_dump_dir")),
                     tag="bench")
    lat, outs, parsed, repaired, vis = [], [], 0, 0, []
    k = max(1, args.images_per_call)
    for i in range(max(1, args.n)):
        batch = [imgs[(i * k + j) % len(imgs)] for j in range(k)]
        times = [float(j) for j in range(len(batch))]
        system, user = render_verify("V99", [], times, None, str(cfg.get("lang", "ru")))
        vis.append(sum(visual_tokens(im.size, px) for im in batch))
        res = be.generate(batch, system, user, max_new_tokens=mnt,
                          timeout_s=float((cfg.get("generation") or {}).get("call_timeout_s", 12.0)),
                          json_schema=verify_schema(sorted(allowed), len(batch)))
        lat.append(res.latency_s)
        if res.completion_tokens is not None:
            outs.append(res.completion_tokens)
        pr = extract_json(res.text)
        if pr.ok and normalize_verify(pr.data, allowed, len(batch))[0] is not None:
            parsed += 1
            repaired += int(pr.repaired)
        dump({"kind": "bench", "call": i, "images": batch, "prompt": user, "raw": [res.text],
              "latency_s": res.latency_s, "error": res.error})
    warm = lat[1:] if len(lat) > 1 else lat  # первый вызов — прогрев
    rep.update({
        "status": "ok",
        "calls": len(lat),
        "latency_first_s": round(lat[0], 3) if lat else None,
        "latency_p50_s": _pct(warm, 0.5),
        "latency_p95_s": _pct(warm, 0.95),
        "latency_mean_s": round(statistics.mean(warm), 3) if warm else None,
        "json_valid_rate": round(parsed / max(1, len(lat)), 3),
        "json_repaired_rate": round(repaired / max(1, len(lat)), 3),
        "out_tokens_mean": round(statistics.mean(outs), 1) if outs else None,
        "visual_tokens_est_per_call": round(statistics.mean(vis), 1) if vis else None,
        "vram_peak_mb_torch": _torch_peak_mb(),
        "vram_after_mb_nvidia_smi": _nvidia_smi_used_mb(),
        "dump_dir": dump.path,
    })
    if args.out:
        Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return rep


def main(argv: Optional[list[str]] = None) -> int:
    rep = run(argv)
    json.dump(rep, sys.stdout, ensure_ascii=False, indent=2, default=str)
    sys.stdout.write("\n")
    return 0 if rep.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
