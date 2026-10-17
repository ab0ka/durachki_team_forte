"""Контакт-листы 4x3 с подписями времени — для ПРОСМОТРА ЧЕЛОВЕКОМ (или локальной VLM), локально.
Результат класть в cache/ (в git не попадает). Шрифт — Windows arialbd.ttf.

usage: sheets.py <video> <out.jpg> <t1,t2,...|overview[:N]> [scale] [crop=x1,y1,x2,y2 (норм.)]
Кадр для t — ближайший по PTS (последовательное декодирование, видео VFR).
"""
import sys

import cv2
from PIL import Image, ImageDraw, ImageFont

video, out, spec = sys.argv[1], sys.argv[2], sys.argv[3]
scale = float(sys.argv[4]) if len(sys.argv) > 4 else 0.75
crop = None
for a in sys.argv[5:]:
    if a.startswith("crop="):
        crop = [float(v) for v in a[5:].split(",")]

cap = cv2.VideoCapture(video)
frames = []  # (t, bgr)
while True:
    ok, fr = cap.read()
    if not ok:
        break
    frames.append((cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, fr))
cap.release()
dur = frames[-1][0]

if spec.startswith("overview"):
    n = int(spec.split(":")[1]) if ":" in spec else 12
    times = [dur * (i + 0.5) / n for i in range(n)]
else:
    times = [float(v) for v in spec.split(",")]

cols = 4
rows = (len(times) + cols - 1) // cols
font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 22)
tiles = []
for t in times:
    if t < 0 or t > dur + 0.05:
        tiles.append(None)
        continue
    ft, fr = min(frames, key=lambda p: abs(p[0] - t))
    img = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    if crop:
        W, H = img.size
        img = img.crop((int(crop[0] * W), int(crop[1] * H), int(crop[2] * W), int(crop[3] * H)))
    img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    label = f"t={ft:.1f}s"
    d.rectangle([0, 0, 118, 30], fill=(0, 0, 0))
    d.text((5, 3), label, fill=(255, 255, 0), font=font)
    tiles.append(img)

tw = max(t.width for t in tiles if t)
th = max(t.height for t in tiles if t)
sheet = Image.new("RGB", (cols * tw + (cols - 1) * 4, rows * th + (rows - 1) * 4), (40, 40, 40))
for i, t in enumerate(tiles):
    if t:
        sheet.paste(t, ((i % cols) * (tw + 4), (i // cols) * (th + 4)))
sheet.save(out, quality=88)
print(out, f"dur={dur:.2f}", sheet.size)
