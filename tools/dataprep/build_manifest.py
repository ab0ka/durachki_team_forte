"""data/labels.csv + data/raw/*.mov -> data/manifest.csv, data/review_queue.csv. Печатает только агрегаты."""
import csv
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

DATA = Path(sys.argv[1])
_nm = DATA / "_dl" / "name_map.json"  # создаётся при распаковке; без него orig_name = file
NAME_MAP = json.load(open(_nm, encoding="utf-8")) if _nm.exists() else {}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe(p):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,codec_name:stream_side_data=rotation:format=duration",
         "-of", "json", str(p)], capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    s = j["streams"][0]
    num, den = map(int, s["avg_frame_rate"].split("/"))
    rot = 0
    for sd in s.get("side_data_list", []) or []:
        rot = int(sd.get("rotation", 0) or 0)
    return dict(width=s["width"], height=s["height"], codec=s["codec_name"], rotation=rot,
                duration_s=round(float(j["format"]["duration"]), 3), nb_frames=int(s.get("nb_frames", 0)),
                avg_fps=round(num / den, 3) if den else 0.0, r_fps=s["r_frame_rate"])


rows = list(csv.DictReader(open(DATA / "labels.csv", encoding="utf-8")))
man = []
for r in rows:
    p = DATA / "raw" / r["file"]
    m = dict(video_id=r["video_id"], file=r["file"], orig_name=NAME_MAP.get(f"raw/{r['file']}", {}).get("orig", r["file"]),
             label=r["label"], t_sec=r["t_sec"], group=r["group"], md5=r["md5"], sha256=sha256(p),
             size_bytes=p.stat().st_size, **probe(p))
    man.append(m)

by_sha = {}
for m in man:
    by_sha.setdefault(m["sha256"], []).append(m)
for m in man:
    dups = [d["video_id"] for d in by_sha[m["sha256"]] if d is not m]
    m["dup_of"] = ";".join(dups)
    labels = {(d["label"], d["t_sec"]) for d in by_sha[m["sha256"]]}
    m["needs_review"] = int(len(labels) > 1)

with open(DATA / "manifest.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(man[0].keys()), lineterminator="\n")
    w.writeheader()
    w.writerows(man)

rq = [m for m in man if m["needs_review"]]
with open(DATA / "review_queue.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["video_id", "dup_of", "label", "t_sec", "reason"])
    for m in rq:
        w.writerow([m["video_id"], m["dup_of"], m["label"], m["t_sec"],
                    "same sha256 as dup_of, different timecode; keep both, verify by human"])

d = [m["duration_s"] for m in man]
print(f"videos={len(man)} unique_sha256={len(by_sha)} pos={sum(m['label']=='violation' for m in man)} "
      f"neg={sum(m['label']=='no_violation' for m in man)}")
print(f"duration min/median/max = {min(d):.1f}/{statistics.median(d):.1f}/{max(d):.1f}s total={sum(d)/60:.1f}min")
print("resolutions:", sorted({(m['width'], m['height'], m['rotation']) for m in man}),
      "codecs:", sorted({m['codec'] for m in man}),
      f"avg_fps range {min(m['avg_fps'] for m in man)}..{max(m['avg_fps'] for m in man)}")
print("dup groups:", [[x["video_id"] for x in v] for v in by_sha.values() if len(v) > 1],
      "needs_review:", [m["video_id"] for m in rq])
