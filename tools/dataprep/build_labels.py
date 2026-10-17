"""open_labels.xlsx -> data/labels.csv (схема eval.labels: video_id,file,label,t_sec,md5,group,t_all,conflict,raw)."""
import csv
import hashlib
import re
import sys
import zipfile
from pathlib import Path

DATA = Path(sys.argv[1])


def read_xlsx(path):
    z = zipfile.ZipFile(path)
    x = z.read("xl/worksheets/sheet1.xml").decode("utf-8")
    rows = []
    for row in re.findall(r"<row[^>]*>(.*?)</row>", x, re.S):
        cells = {}
        for attrs, inner in re.findall(r"<c ([^>]*?)(?:/>|>(.*?)</c>)", row, re.S):
            col = re.match(r'.*?r="([A-Z]+)\d+"', attrs).group(1)
            v = re.search(r"<v>(.*?)</v>", inner or "")
            cells[col] = v.group(1) if v else "".join(re.findall(r"<t[^>]*>(.*?)</t>", inner or "", re.S))
        rows.append(cells)
    return rows[1:]


def norm_id(name):
    # X-1, X-2 — латинская X; остальные — кириллическая Х (U+0425). Единый id: латинская X.
    m = re.fullmatch(r"\s*[XХxх]\s*-\s*(\d+)\s*", name)
    if not m:
        raise ValueError(name)
    return f"X-{int(m.group(1))}"


def parse_t(s):
    s = (s or "").strip().lower()
    if not s:
        return None
    mins = re.search(r"(\d+)\s*мин", s)
    secs = re.search(r"(\d+)\s*сек", s)
    if not (mins or secs):
        raise ValueError(s)
    return (int(mins.group(1)) * 60 if mins else 0) + (int(secs.group(1)) if secs else 0)


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


out = []
for r in read_xlsx(DATA / "open_labels.xlsx"):
    vid = norm_id(r["A"])
    lab = r["C"].strip().lower()
    label = {"нарушение": "violation", "без нарушения": "no_violation"}[lab]
    t = parse_t(r.get("B"))
    assert (t is None) == (label == "no_violation"), (vid, t, label)
    f = f"{vid}.mov"
    out.append(dict(video_id=vid, file=f, label=label, t_sec="" if t is None else t, md5=md5(DATA / "raw" / f),
                    t_all="" if t is None else str(t), raw=" | ".join(x.strip() for x in (r.get("B", ""), r["C"]) if x.strip())))

# group: одинаковый md5 -> одна группа (id видео с наименьшим номером) — для group k-fold
by_md5 = {}
for o in out:
    by_md5.setdefault(o["md5"], []).append(o["video_id"])
for o in out:
    o["group"] = min(by_md5[o["md5"]], key=lambda v: int(v.split("-")[1]))
# conflict (как в eval.labels.assign_groups): в md5-группе разные метки или таймкоды -> проверка человеком
for o in out:
    members = [x for x in out if x["md5"] == o["md5"]]
    o["conflict"] = int(len({x["label"] for x in members}) > 1 or len({x["t_sec"] for x in members if x["t_sec"] != ""}) > 1)

out.sort(key=lambda o: int(o["video_id"].split("-")[1]))
with open(DATA / "labels.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["video_id", "file", "label", "t_sec", "md5", "group", "t_all", "conflict", "raw"], lineterminator="\n")
    w.writeheader()
    w.writerows(out)
print(len(out), "rows;", sum(o["label"] == "violation" for o in out), "violation")
print("dups:", {k: v for k, v in by_md5.items() if len(v) > 1})
