"""Проверка локального окружения, весов, данных и меток L1 — без GPU-нагрузки и без сети.

Секции (по умолчанию все): env, weights, data, labels; опционально eval (совместимость с eval.labels команды).
Печатает только OK/WARN/FAIL и агрегаты (без содержимого меток и описаний сцен). Код возврата 1, если есть FAIL.

usage:
  python -I -X utf8 tools/dataprep/check_local.py --root .                     # быстрые проверки
  python -I -X utf8 tools/dataprep/check_local.py --root . --hash              # + sha256 видео и весов
  python -I -X utf8 tools/dataprep/check_local.py --root . --write-weights-manifest   # снять эталон sha256 весов
  python -I -X utf8 tools/dataprep/check_local.py --root . --eval-src <корень ветки с eval/>   # + eval.labels
CUDA трогается только через torch.cuda.is_available() (без контекста и памяти); --no-cuda — не трогать вовсе.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata as md
import json
import re
import shutil
import sys
from pathlib import Path

HF_MODELS = ["google/siglip2-base-patch16-224", "google/owlv2-base-patch16-ensemble", "PekingU/rtdetr_v2_r18vd",
             "ustc-community/dfine-medium-obj2coco", "intfloat/multilingual-e5-small", "Qwen/Qwen3-VL-4B-Instruct"]
GGUF_GLOB = "copilot/qwen2.5-7b-instruct-q4_k_m-*-of-*.gguf"
POSE_GLOB = "pose/**/*rtmpose-m*/end2end.onnx"
LABELS_COLS = ["video_id", "file", "label", "t_sec", "md5", "group", "t_all", "conflict", "raw"]
TYPES_REQUIRED = ["video_id", "t_sec", "type_id", "vlm_type_id", "source", "confirmed", "conflict", "notes"]
VALID_TYPES = {f"V0{i}" for i in range(1, 9)} | {"V99"}


class Report:
    def __init__(self):
        self.n = {"OK": 0, "WARN": 0, "FAIL": 0}

    def __call__(self, level: str, section: str, msg: str):
        self.n[level] += 1
        print(f"{level:4s} [{section}] {msg}", flush=True)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_lock(p: Path) -> dict[str, str]:
    pins = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s;#]+)", line.strip())
        if m:
            pins[m.group(1).lower().replace("_", "-")] = m.group(2)
    return pins


def check_env(root: Path, r: Report, no_cuda: bool):
    v = sys.version_info
    r("OK" if (v.major, v.minor) == (3, 11) else "FAIL", "env", f"python {v.major}.{v.minor}.{v.micro} ({sys.executable})")
    lock = root / "tools/dataprep/requirements-l1.lock.txt"
    if lock.exists():
        pins = read_lock(lock)
        bad, missing = [], []
        for name, want in pins.items():
            try:
                have = md.version(name)
            except md.PackageNotFoundError:
                missing.append(name)
                continue
            if have != want:
                bad.append(f"{name} {have}≠{want}")
        lvl = "OK" if not bad and not missing else "FAIL"
        r(lvl, "env", f"lock: {len(pins)} pins, mismatched {len(bad)} {bad[:5]}, missing {len(missing)} {missing[:5]}")
    else:
        r("FAIL", "env", f"нет {lock}")
    try:
        import torch
        cuda = "skipped" if no_cuda else str(torch.cuda.is_available())
        lvl = "OK" if (no_cuda or torch.cuda.is_available()) and str(torch.version.cuda).startswith("12.") else "FAIL"
        r(lvl, "env", f"torch {torch.__version__} (CUDA build {torch.version.cuda}), cuda.is_available={cuda}")
    except Exception as e:  # noqa: BLE001
        r("FAIL", "env", f"torch: {type(e).__name__}: {e}")
    try:
        import onnxruntime as ort
        prov = ort.get_available_providers()
        ok = ort.__version__.startswith("1.23.") and "CUDAExecutionProvider" in prov
        r("OK" if ok else "WARN", "env", f"onnxruntime {ort.__version__} providers={prov} (ожидается 1.23.x: ≥1.24 — CUDA 13)")
    except Exception as e:  # noqa: BLE001
        r("FAIL", "env", f"onnxruntime: {type(e).__name__}: {e}")
    for exe in ("ffmpeg", "ffprobe"):
        r("OK" if shutil.which(exe) else "FAIL", "env", f"{exe}: {shutil.which(exe) or 'не найден в PATH'}")


def weight_files(w: Path) -> list[Path]:
    files = []
    hub = w / "hf/hub"
    if hub.exists():
        files += [p for p in hub.glob("models--*/snapshots/*/**/*") if p.is_file()]
    files += sorted(w.glob(GGUF_GLOB)) + sorted(w.glob(POSE_GLOB))
    return sorted(files)


def check_weights(root: Path, r: Report, do_hash: bool, write_manifest: bool):
    w = root / "weights"
    hub = w / "hf/hub"
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        try_to_load_from_cache = None
    for rid in HF_MODELS:
        d = hub / ("models--" + rid.replace("/", "--"))
        ref = d / "refs/main"
        if not ref.exists():
            r("FAIL", "weights", f"{rid}: нет {ref.relative_to(root)}")
            continue
        snap = d / "snapshots" / ref.read_text().strip()
        size = sum(p.stat().st_size for p in snap.rglob("*") if p.is_file())
        resolved = try_to_load_from_cache(rid, "config.json", cache_dir=str(hub)) if try_to_load_from_cache else None
        ok = snap.exists() and isinstance(resolved, str)
        r("OK" if ok else "FAIL", "weights", f"{rid}: {size / 2**30:.2f} GiB, офлайн config.json={'да' if ok else 'нет'}")
    gg = sorted(w.glob(GGUF_GLOB))
    r("OK" if len(gg) == 2 else "FAIL", "weights",
      f"copilot GGUF шардов {len(gg)} ({sum(p.stat().st_size for p in gg) / 2**30:.2f} GiB); COPILOT_MODEL_PATH → "
      f"{gg[0].relative_to(root).as_posix() if gg else '—'}")
    pose = sorted(w.glob(POSE_GLOB))
    r("OK" if pose else "FAIL", "weights", f"RTMPose ONNX: {pose[0].relative_to(root).as_posix() if pose else 'нет'}")
    if not (do_hash or write_manifest):
        return
    man_path = w / "L1_weights_manifest.json"
    cur = {p.relative_to(w).as_posix(): [p.stat().st_size, sha256_file(p)] for p in weight_files(w)}
    if write_manifest:
        man_path.write_text(json.dumps(cur, indent=1), encoding="utf-8")
        r("OK", "weights", f"эталон sha256 записан: {man_path.relative_to(root)} ({len(cur)} файлов)")
        return
    if not man_path.exists():
        r("WARN", "weights", "нет weights/L1_weights_manifest.json — снимите эталон --write-weights-manifest")
        return
    ref = json.loads(man_path.read_text(encoding="utf-8"))
    changed = [k for k in ref if k in cur and cur[k] != ref[k]]
    gone = [k for k in ref if k not in cur]
    new = [k for k in cur if k not in ref]
    lvl = "OK" if not changed and not gone else "FAIL"
    r(lvl, "weights", f"sha256: {len(ref)} в эталоне, изменено {len(changed)}, пропало {len(gone)}, новых {len(new)}")


def read_csv(p: Path) -> list[dict]:
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def check_data(root: Path, r: Report, do_hash: bool):
    d = root / "data"
    movs = sorted((d / "raw").glob("*.mov")) if (d / "raw").exists() else []
    r("OK" if len(movs) == 42 else "FAIL", "data", f"data/raw: {len(movs)} .mov (ожидается 42)")
    vids = d / "videos"
    same = vids.exists() and {p.name for p in vids.glob("*.mov")} == {p.name for p in movs}
    r("OK" if same else "WARN", "data", f"data/videos (junction → raw): {'совпадает' if same else 'нет или отличается'}")
    lab = d / "labels.csv"
    if not lab.exists():
        r("FAIL", "data", "нет data/labels.csv")
        return
    rows = read_csv(lab)
    cols_ok = list(rows[0].keys()) == LABELS_COLS if rows else False
    files_ok = all((d / "raw" / x["file"]).exists() for x in rows)
    pos = sum(x["label"] == "violation" for x in rows)
    conf = sorted(x["video_id"] for x in rows if x.get("conflict") == "1")
    ok = cols_ok and files_ok and len(rows) == 42 and pos == 22
    r("OK" if ok else "FAIL", "data", f"labels.csv: {len(rows)} строк, violation {pos}, колонки eval.labels "
      f"{'да' if cols_ok else 'НЕТ'}, файлы на месте {'да' if files_ok else 'НЕТ'}")
    r("OK" if conf else "WARN", "data", f"conflict=1 (не решено, решает человек): {conf}")
    mf = d / "manifest.csv"
    if not mf.exists():
        r("WARN", "data", "нет data/manifest.csv (tools/dataprep/build_manifest.py data)")
    elif do_hash:
        man = read_csv(mf)
        bad = [m["video_id"] for m in man if sha256_file(d / "raw" / m["file"]) != m["sha256"]]
        r("OK" if not bad else "FAIL", "data", f"sha256 видео vs manifest: {len(man) - len(bad)}/{len(man)} совпали {bad}")


def check_labels(root: Path, r: Report):
    p = root / "data/labels_types.csv"
    if not p.exists():
        r("WARN", "labels", "нет data/labels_types.csv (tools/dataprep/types_review.py --root .)")
        return
    rows = read_csv(p)
    miss = [c for c in TYPES_REQUIRED if rows and c not in rows[0]]
    if miss:
        r("FAIL", "labels", f"labels_types.csv: нет колонок {miss} (старый формат — пересоберите types_review.py)")
        return
    events = {(x["video_id"], float(x["t_sec"])) for x in read_csv(root / "data/labels.csv") if x["label"] == "violation"}
    keys = {(x["video_id"], float(x["t_sec"])) for x in rows}
    r("OK" if keys == events else "FAIL", "labels",
      f"события labels_types ↔ labels.csv: {len(keys & events)}/{len(events)} совпали, лишних {len(keys - events)}")
    conf = [x for x in rows if x["confirmed"] == "1"]
    leak = [x["video_id"] for x in rows if x["confirmed"] != "1" and x["type_id"].strip()]
    r("OK" if not leak else "FAIL", "labels",
      f"type_id только у подтверждённых: неподтверждённых с type_id {len(leak)} {leak[:5]} (VLM ≠ разметка)")
    bad = [x["video_id"] for x in conf if x["type_id"] and x["type_id"] not in VALID_TYPES]
    r("OK" if not bad else "FAIL", "labels", f"подтверждено человеком {len(conf)}/{len(rows)}; недопустимых type_id {bad}")
    unres = [x for x in rows if x["conflict"]]
    flagged = all(x["notes"].startswith(("UNRESOLVED", "PAIR_DECISION")) for x in unres)
    r("OK" if flagged else "FAIL", "labels",
      f"md5-конфликт: {len(unres)} строк, UNRESOLVED {sum(x['notes'].startswith('UNRESOLVED') for x in unres)}, "
      f"с решением человека {sum(x['notes'].startswith('PAIR_DECISION') for x in unres)}")


def check_eval(root: Path, r: Report, src: Path):
    sys.path.insert(0, str(src))
    try:
        from eval.labels import load_ground_truth, read_labels_csv, read_types_csv
    except Exception as e:  # noqa: BLE001
        r("FAIL", "eval", f"не импортируется eval.labels из {src}: {type(e).__name__}: {e}")
        return
    rows = read_labels_csv(root / "data/labels.csv")
    types = read_types_csv(root / "data/labels_types.csv")
    gt = load_ground_truth(root / "data/labels.csv", root / "data/labels_types.csv")
    n_types = sum(len(g.types) for g in gt.values())
    conf = sorted(k for k, g in gt.items() if g.conflict)
    r("OK", "eval", f"read_labels_csv: {len(rows)} строк; read_types_csv: {sum(len(v) for v in types.values())} "
      f"типов (= подтверждённые); load_ground_truth: {len(gt)} групп, типов в GT {n_types}, conflict-групп {conf}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--only", default="env,weights,data,labels")
    ap.add_argument("--hash", action="store_true", help="sha256 видео (vs manifest) и весов (vs эталон)")
    ap.add_argument("--write-weights-manifest", action="store_true")
    ap.add_argument("--eval-src", type=Path, default=None, help="корень кода с пакетом eval/ (ветка интеграции)")
    ap.add_argument("--no-cuda", action="store_true")
    a = ap.parse_args()
    root = a.root.resolve()
    r = Report()
    only = set(a.only.split(","))
    if "env" in only:
        check_env(root, r, a.no_cuda)
    if "weights" in only:
        check_weights(root, r, a.hash, a.write_weights_manifest)
    if "data" in only:
        check_data(root, r, a.hash)
    if "labels" in only:
        check_labels(root, r)
    if a.eval_src:
        check_eval(root, r, a.eval_src.resolve())
    print(f"итог: OK {r.n['OK']}, WARN {r.n['WARN']}, FAIL {r.n['FAIL']}")
    sys.exit(1 if r.n["FAIL"] else 0)


if __name__ == "__main__":
    main()
