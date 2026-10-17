"""Пред-разметка кадров OWLv2 (google/owlv2-base-patch16-ensemble, Apache-2.0), локально.

    python datasets/det/prelabel_owlv2.py                # все кадры из data/frames/index.csv
    python datasets/det/prelabel_owlv2.py --batch 2 --device cpu

Веса — HF-кэш weights/hf/hub (как в deploy/weights.py): первый запуск качает ~0.6 ГБ, дальше HF_HUB_OFFLINE=1.
Промпты и пороги — CLASSES в common.py. Пороги занижены: удалить лишний бокс быстрее, чем нарисовать.
Выход: data/frames/_owlv2/predictions.jsonl (по строке на кадр, боксы в пикселях xyxy) + meta.json.
Повторный запуск досчитывает только новые кадры (если модель и промпты те же).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASSES, EXCLUSIVE, FRAMES_DIR, INDEX_CSV, PRED_JSONL, ROOT, read_index  # noqa: E402

MODEL_ID = "google/owlv2-base-patch16-ensemble"


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def nms(boxes: list[dict], thr: float) -> list[dict]:
    keep: list[dict] = []
    for b in sorted(boxes, key=lambda b: -b["score"]):
        if all(iou(b["xyxy"], k["xyxy"]) < thr for k in keep):
            keep.append(b)
    return keep


def postprocess(scores, boxes_cxcywh, width: int, height: int, nms_iou: float = 0.3,
                min_side: float = 6.0, max_area: float = 0.5, cross_iou: float = 0.7) -> list[dict]:
    """scores: (P, C) — уже max по промптам класса; boxes: (P, 4) cxcywh в долях квадрата max(W, H)
    (OWLv2 дополняет кадр до квадрата снизу/справа). → боксы в пикселях исходного кадра."""
    side = float(max(width, height))
    out: list[dict] = []
    for ci, c in enumerate(CLASSES):
        col = scores[:, ci]
        cand = []
        for p in (col >= c["thr"]).nonzero()[0]:
            cx, cy, w, h = (float(v) * side for v in boxes_cxcywh[p])
            x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
            x2, y2 = min(float(width), cx + w / 2), min(float(height), cy + h / 2)
            if x2 - x1 < min_side or y2 - y1 < min_side or (x2 - x1) * (y2 - y1) > max_area * width * height:
                continue
            cand.append({"label": c["name"], "score": round(float(col[p]), 4),
                         "xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]})
        out += nms(cand, nms_iou)[: c["max"]]
    for a, b in EXCLUSIVE:                       # открытый и закрытый сейф на одном месте — оставить более уверенный
        drop = set()
        for i, x in enumerate(out):
            for j, y in enumerate(out):
                if x["label"] == a and y["label"] == b and iou(x["xyxy"], y["xyxy"]) > 0.5:
                    drop.add(j if x["score"] >= y["score"] else i)
        out = [x for k, x in enumerate(out) if k not in drop]
    # один объект — один бокс: разметчику проще сменить класс клавишей, чем удалять дубль другого класса
    keep: list[dict] = []
    for b in sorted(out, key=lambda b: -b["score"]):
        if all(iou(b["xyxy"], k["xyxy"]) <= cross_iou for k in keep):
            keep.append(b)
    order = {c["name"]: i for i, c in enumerate(CLASSES)}
    return sorted(keep, key=lambda b: (order[b["label"]], -b["score"]))


def prompts_hash() -> str:
    return hashlib.sha1(json.dumps([[c["name"], c["prompts"], c["thr"], c["max"]] for c in CLASSES]).encode()).hexdigest()[:12]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=Path, default=INDEX_CSV)
    ap.add_argument("--frames", type=Path, default=FRAMES_DIR)
    ap.add_argument("--out", type=Path, default=PRED_JSONL)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("HF_HUB_CACHE", ROOT / "weights" / "hf" / "hub")))
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="только первые N кадров (проверка)")
    args = ap.parse_args(argv)

    import numpy as np
    import torch
    from PIL import Image
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    rows = read_index(args.index)
    tag = prompts_hash()
    done: set[str] = set()
    if args.out.is_file():
        meta_p = args.out.parent / "meta.json"
        meta_old = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.is_file() else {}
        if meta_old.get("prompts_hash") == tag and meta_old.get("model") == args.model:
            done = {json.loads(line)["file"] for line in args.out.read_text(encoding="utf-8").splitlines() if line.strip()}
        else:
            print("модель/промпты изменились — предсказания пересчитываются заново")
            args.out.unlink()
    todo = [r for r in rows if r["file"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"кадров: {len(rows)}, уже есть: {len(done)}, считать: {len(todo)}")
    if not todo:
        return 0

    dev = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    dtype = torch.float16 if dev == "cuda" else torch.float32
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    proc = Owlv2Processor.from_pretrained(args.model, cache_dir=str(args.cache_dir))
    model = Owlv2ForObjectDetection.from_pretrained(args.model, cache_dir=str(args.cache_dir)).to(dev, dtype).eval()
    queries: list[str] = []
    groups: list[list[int]] = []                 # индексы промптов каждого класса
    for c in CLASSES:
        groups.append(list(range(len(queries), len(queries) + len(c["prompts"]))))
        queries += c["prompts"]
    # фактическая ревизия весов — для воспроизводимости
    ref = args.cache_dir / f"models--{args.model.replace('/', '--')}" / "refs" / "main"
    revision = getattr(model.config, "_commit_hash", None) or (ref.read_text().strip() if ref.is_file() else None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    (args.out.parent / "meta.json").write_text(json.dumps(
        {"model": args.model, "revision": revision, "license": "Apache-2.0", "prompts_hash": tag, "device": dev,
         "classes": [{k: c[k] for k in ("name", "prompts", "thr", "max")} for c in CLASSES]},
        ensure_ascii=False, indent=1), encoding="utf-8")

    t0 = time.time()
    n_boxes = 0
    with open(args.out, "a", encoding="utf-8") as f, torch.inference_mode():
        text = proc.tokenizer(queries, padding="max_length", max_length=16, truncation=True, return_tensors="pt")
        for i in range(0, len(todo), args.batch):
            chunk = todo[i:i + args.batch]
            imgs = [Image.open(args.frames / r["file"]).convert("RGB") for r in chunk]
            pix = proc.image_processor(images=imgs, return_tensors="pt")["pixel_values"].to(dev, dtype)
            n = len(chunk)
            out = model(input_ids=text["input_ids"].repeat(n, 1).to(dev),
                        attention_mask=text["attention_mask"].repeat(n, 1).to(dev), pixel_values=pix)
            logits = torch.sigmoid(out.logits.float()).cpu().numpy()          # (B, P, Q)
            boxes = out.pred_boxes.float().cpu().numpy()                      # (B, P, 4)
            for b, r in enumerate(chunk):
                cls_scores = np.stack([logits[b][:, g].max(axis=1) for g in groups], axis=1)   # (P, C)
                dets = postprocess(cls_scores, boxes[b], imgs[b].width, imgs[b].height)
                n_boxes += len(dets)
                f.write(json.dumps({"file": r["file"], "width": imgs[b].width, "height": imgs[b].height,
                                    "boxes": dets}, ensure_ascii=False) + "\n")
            f.flush()
            if (i // args.batch) % 25 == 0:
                done_n = i + n
                print(f"  {done_n}/{len(todo)}  {done_n / (time.time() - t0):.1f} кадр/с", flush=True)
    print(f"готово: {len(todo)} кадров, {n_boxes} боксов, {time.time() - t0:.0f} с → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
