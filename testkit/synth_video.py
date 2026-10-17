"""Синтетическое видео кассы «камера сверху» для интеграционных тестов без реальных данных (A01).

    gt = make_video(tmp_path / "v01.mov", "V01", duration=20, fps=30, size=(464, 848))
    # gt = {"verdict": "violation", "type_id": "V01", "t": 9.0, "t_start": ..., "t_end": ..., ...}

Сцена: стол с ноутбуком, стационарным телефоном, счётчиком банкнот, лотком наличных; серый сейф на тумбе
(дверца открывается); стеклянная перегородка с окном и стеклянной дверью; люди — эллипс «плеч» + голова
(тёмные волосы, со стороны лица — светлый сегмент) + руки; наличные — зелёные пачки; инкассаторская сумка —
крупная зелёная с ремнём и пломбой. Сценарии: ``clean``, ``clean_collector``, ``V01``…``V08``
(см. ``testkit.scenarios``). Ракурсы ``view``: default, mirror, rot180, landscape, perspective — для проверки
устойчивости к другому углу камеры (никаких захардкоженных пиксельных зон!).

CLI::

    python -m testkit.synth_video --out data/synth --all --duration 20 --fps 30 [--view perspective]
"""
from __future__ import annotations

import argparse
import json
import math
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

from .scenarios import (
    BODY_A,
    BODY_B,
    DEFAULT_SIZE,
    DOOR_HINGE,
    DOOR_LEN,
    HAND_R,
    HEAD_R,
    LAYOUT,
    ROLE_COLORS,
    SCENARIOS,
    VIEWS,
    ItemState,
    PersonState,
    SceneScript,
    Snapshot,
    view_transform,
)

__all__ = ["make_video", "render_frames", "render_snapshot", "SynthFrames", "add_rotation_metadata", "main"]

SKIN = (150, 180, 220)
HAIR = (40, 40, 50)
FLOOR_CASHIER = (205, 208, 212)
FLOOR_CLIENT = (182, 188, 192)
GLASS = (232, 214, 178)
FRAME = (105, 105, 110)


def _np():
    import numpy as np  # лениво
    return np


def _cv2():
    import cv2  # лениво
    return cv2


class _Canvas:
    def __init__(self, size: tuple[int, int]):
        self.w, self.h = size

    def p(self, x: float, y: float) -> tuple[int, int]:
        return int(round(x * (self.w - 1))), int(round(y * (self.h - 1)))

    def r(self, frac_of_width: float) -> int:
        return max(1, int(round(frac_of_width * self.w)))

    def rect(self, img, box, color, thickness=-1):
        _cv2().rectangle(img, self.p(box[0], box[1]), self.p(box[2], box[3]), color, thickness, _cv2().LINE_AA)


# ----------------------------------------------------------------------------- статический фон

_BG_CACHE: dict[tuple[int, int], Any] = {}


def _background(size: tuple[int, int]):
    if size in _BG_CACHE:
        return _BG_CACHE[size].copy()
    np, cv2 = _np(), _cv2()
    c = _Canvas(size)
    img = np.empty((c.h, c.w, 3), np.uint8)
    img[:] = FLOOR_CASHIER
    y_part = c.p(0, LAYOUT["partition"][3])[1]
    img[y_part:] = FLOOR_CLIENT
    step = c.r(0.09)
    for x in range(0, c.w, step):  # плитка
        cv2.line(img, (x, 0), (x, c.h - 1), (190, 193, 197), 1)
    for y in range(0, c.h, step):
        cv2.line(img, (0, y), (c.w - 1, y), (190, 193, 197), 1)
    cv2.rectangle(img, (0, 0), (c.w - 1, c.h - 1), (80, 80, 85), c.r(0.012))  # стены
    # стол и предметы на нём
    c.rect(img, LAYOUT["desk"], (170, 196, 214))
    c.rect(img, LAYOUT["desk"], (120, 140, 160), 2)
    lx1, ly1, lx2, ly2 = LAYOUT["laptop"]
    c.rect(img, LAYOUT["laptop"], (55, 55, 60))
    c.rect(img, (lx1 + 0.01, ly1 + 0.006, lx2 - 0.01, ly1 + 0.045), (120, 90, 60))  # экран
    c.rect(img, LAYOUT["desk_phone"], (30, 30, 30))
    px1, py1, px2, py2 = LAYOUT["desk_phone"]
    c.rect(img, (px1 + 0.01, py1 + 0.012, px2 - 0.01, py2 - 0.008), (90, 90, 90))
    c.rect(img, LAYOUT["money_counter"], (125, 125, 130))
    mx1, my1, mx2, my2 = LAYOUT["money_counter"]
    c.rect(img, (mx1 + 0.015, my1 + 0.03, mx2 - 0.015, my1 + 0.04), (40, 40, 40))
    c.rect(img, LAYOUT["cash_tray"], (95, 95, 100), 2)
    c.rect(img, LAYOUT["document"], (245, 245, 245))
    dx1, dy1, dx2, _ = LAYOUT["document"]
    for k in range(4):
        yy = dy1 + 0.012 + k * 0.012
        cv2.line(img, c.p(dx1 + 0.008, yy), c.p(dx2 - 0.008, yy), (150, 150, 150), 1)
    # тумба сейфа
    c.rect(img, LAYOUT["safe_stand"], (70, 70, 76))
    # перегородка с окном
    c.rect(img, LAYOUT["partition"], GLASS)
    x1, y1, x2, y2 = LAYOUT["partition"]
    cv2.line(img, c.p(x1, y1), c.p(x2, y1), FRAME, 2)
    cv2.line(img, c.p(x1, y2), c.p(x2, y2), FRAME, 2)
    c.rect(img, (0.30, y1, 0.42, y2), (245, 240, 230))  # окно выдачи
    _BG_CACHE[size] = img
    return img.copy()


# ----------------------------------------------------------------------------- динамика

def _draw_safe(img, c: _Canvas, open_frac: float) -> None:
    cv2 = _cv2()
    x1, y1, x2, y2 = LAYOUT["safe"]
    front_y = y1 + (y2 - y1) * 0.68
    c.rect(img, (x1, y1, x2, front_y), (178, 183, 188))      # верхняя грань
    c.rect(img, (x1, front_y, x2, y2), (132, 137, 142))      # передняя грань
    c.rect(img, LAYOUT["safe"], (90, 92, 96), 2)
    if open_frac < 0.15:
        hx, hy = c.p(x2 - 0.04, (front_y + y2) / 2)
        cv2.circle(img, (hx, hy), c.r(0.012), (50, 50, 50), -1, cv2.LINE_AA)
        c.rect(img, (x1 + 0.03, front_y + 0.008, x1 + 0.07, y2 - 0.008), (60, 60, 60))  # кодовый замок
        return
    c.rect(img, (x1 + 0.006, front_y, x2 - 0.006, y2), (22, 22, 22))  # тёмный проём
    for k in (0.35, 0.7):
        yy = front_y + (y2 - front_y) * k
        cv2.line(img, c.p(x1 + 0.01, yy), c.p(x2 - 0.01, yy), (70, 70, 70), 1)
    theta = math.radians(100.0 * min(1.0, open_frac))
    hinge = c.p(x2, y2)
    length = c.r(x2 - x1)
    end = (int(hinge[0] - length * math.cos(theta)), int(hinge[1] + length * math.sin(theta)))
    cv2.line(img, hinge, end, (120, 125, 130), c.r(0.022), cv2.LINE_AA)  # дверца
    cv2.line(img, hinge, end, (80, 82, 86), 2, cv2.LINE_AA)


def _draw_door(img, c: _Canvas, open_frac: float) -> None:
    cv2 = _cv2()
    box = LAYOUT["door"]
    if open_frac < 0.05:
        c.rect(img, box, GLASS)
        c.rect(img, box, FRAME, 2)
        cv2.circle(img, c.p(box[0] + 0.03, (box[1] + box[3]) / 2), c.r(0.008), (60, 60, 60), -1)
        return
    c.rect(img, box, FLOOR_CLIENT)  # проём
    cv2.line(img, c.p(box[0], box[1]), c.p(box[0], box[3]), FRAME, 3)
    theta = math.radians(85.0 * min(1.0, open_frac))
    hinge = c.p(*DOOR_HINGE)
    length = c.r(DOOR_LEN)
    end = (int(hinge[0] - length * math.cos(theta)), int(hinge[1] + length * math.sin(theta)))
    cv2.line(img, hinge, end, FRAME, c.r(0.022), cv2.LINE_AA)
    cv2.line(img, hinge, end, GLASS, max(1, c.r(0.022) - 4), cv2.LINE_AA)


def _draw_item(img, c: _Canvas, it: ItemState) -> None:
    cv2 = _cv2()
    x1, y1, x2, y2 = it.bbox
    if it.label == "cash":
        for k in range(3):  # пачка
            off = 0.004 * k
            c.rect(img, (x1 + off, y1 + off, x2 + off - 0.008, y2 + off - 0.006), (70, 150, 60))
            c.rect(img, (x1 + off, y1 + off, x2 + off - 0.008, y2 + off - 0.006), (40, 100, 35), 1)
        c.rect(img, (x1 + 0.015, y1 + 0.012, x2 - 0.02, y2 - 0.015), (125, 200, 115))
    elif it.label == "cash_bag":
        c.rect(img, it.bbox, (45, 145, 45))
        c.rect(img, it.bbox, (20, 80, 20), 2)
        cv2.line(img, c.p(x1, (y1 + y2) / 2), c.p(x2, (y1 + y2) / 2), (25, 25, 25), c.r(0.008))  # ремень
        cv2.circle(img, c.p(x2 - 0.015, y1 + 0.01), c.r(0.008), (40, 40, 200), -1)  # пломба
    elif it.label == "cell_phone":
        c.rect(img, it.bbox, (25, 25, 25))
        c.rect(img, (x1 + 0.004, y1 + 0.004, x2 - 0.004, y2 - 0.004), (140, 95, 50))
    elif it.label == "cup":
        cx, cy = c.p((x1 + x2) / 2, (y1 + y2) / 2)
        cv2.circle(img, (cx, cy), c.r((x2 - x1) / 2), (235, 235, 235), -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), c.r((x2 - x1) / 3), (40, 70, 110), -1, cv2.LINE_AA)
    elif it.label == "food":
        c.rect(img, it.bbox, (120, 180, 220))
        c.rect(img, it.bbox, (70, 120, 170), 2)
    else:
        c.rect(img, it.bbox, (200, 200, 200))


def _draw_person(img, c: _Canvas, p: PersonState) -> None:
    cv2 = _cv2()
    color = ROLE_COLORS.get(p.role, (128, 128, 128))
    center = c.p(p.x, p.y)
    a = math.radians(p.facing)
    fx, fy = math.sin(a), -math.cos(a)
    axes = (c.r(BODY_A), c.r(BODY_B))
    if p.hands_visible:
        rx, ry = -fy, fx
        for side, (hx, hy) in zip((-1, 1), p.hands):
            sh = (int(center[0] + side * rx * axes[0] * 0.8), int(center[1] + side * ry * axes[0] * 0.8))
            cv2.line(img, sh, c.p(hx, hy), color, c.r(0.03), cv2.LINE_AA)
    cv2.ellipse(img, center, axes, p.facing % 180, 0, 360, color, -1, cv2.LINE_AA)
    cv2.ellipse(img, center, axes, p.facing % 180, 0, 360, tuple(int(v * 0.6) for v in color), 2, cv2.LINE_AA)
    if p.role == "collector":  # светоотражающая полоса на форме
        cv2.ellipse(img, center, (int(axes[0] * 0.9), max(1, axes[1] // 4)), p.facing % 180, 0, 360,
                    (60, 200, 220), -1, cv2.LINE_AA)
    hr = c.r(HEAD_R)
    cv2.circle(img, center, hr, HAIR, -1, cv2.LINE_AA)
    face = (int(center[0] + fx * hr * 0.55), int(center[1] + fy * hr * 0.55))
    cv2.circle(img, face, max(1, int(hr * 0.55)), SKIN, -1, cv2.LINE_AA)
    if p.hands_visible:
        for hx, hy in p.hands:
            cv2.circle(img, c.p(hx, hy), c.r(HAND_R), SKIN, -1, cv2.LINE_AA)


def render_snapshot(snap: Snapshot, size: tuple[int, int] = DEFAULT_SIZE):
    """Кадр BGR uint8 базового ракурса для снимка сцены."""
    img = _background(size)
    c = _Canvas(size)
    _draw_safe(img, c, snap.safe_open)
    _draw_door(img, c, snap.door_open)
    for it in snap.items:
        if it.holder is None:
            _draw_item(img, c, it)
    for p in sorted(snap.persons, key=lambda q: q.y):
        _draw_person(img, c, p)
    for it in snap.items:
        if it.holder is not None:
            _draw_item(img, c, it)
    return img


def _apply_view(img, view: str, size: tuple[int, int]):
    if view == "default":
        return img
    cv2 = _cv2()
    if view == "mirror":
        return cv2.flip(img, 1)
    if view == "rot180":
        return cv2.rotate(img, cv2.ROTATE_180)
    if view == "landscape":
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    np = _np()
    m, out = view_transform(view, size)
    return cv2.warpPerspective(img, np.array(m, dtype=np.float64), out, borderValue=(45, 45, 48))


class SynthFrames(Sequence):
    """Ленивая последовательность кадров сценария (рендер по запросу + LRU) — экономит память в тестах."""

    def __init__(self, script: SceneScript, fps: float, view: str = "default", seed: int = 0,
                 noise_sigma: float = 2.0, osd: bool = False, cache: int = 32):
        np = _np()
        self.script, self.fps, self.view, self.osd = script, float(fps), view, osd
        self.n = max(1, int(round(script.duration * self.fps)))
        rng = np.random.default_rng(seed)
        w, h = script.size
        self._noise = [rng.normal(0, noise_sigma, (h, w, 3)).astype(np.int16) for _ in range(3)] \
            if noise_sigma > 0 else None
        self._cache: OrderedDict[int, Any] = OrderedDict()
        self._max = cache

    def __len__(self) -> int:
        return self.n

    def render(self, i: int):
        np, cv2 = _np(), _cv2()
        t = i / self.fps
        img = render_snapshot(self.script.at(t), self.script.size)
        if self._noise is not None:
            img = np.clip(img.astype(np.int16) + self._noise[i % len(self._noise)], 0, 255).astype(np.uint8)
        if self.osd:
            cv2.putText(img, f"CAM-01 2026-10-17 10:00:{int(t) % 60:02d}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return _apply_view(img, self.view, self.script.size)

    def __getitem__(self, i):  # type: ignore[override]
        if isinstance(i, slice):
            return [self[k] for k in range(*i.indices(self.n))]
        if i < 0:
            i += self.n
        if not 0 <= i < self.n:
            raise IndexError(i)
        if i in self._cache:
            self._cache.move_to_end(i)
            return self._cache[i]
        frame = self.render(i)
        self._cache[i] = frame
        if len(self._cache) > self._max:
            self._cache.popitem(last=False)
        return frame


def _gt_dict(script: SceneScript, fps: float, view: str, n_frames: int, seed: int) -> dict:
    _, out_size = view_transform(view, script.size)
    gt = script.gt.as_dict()
    gt.update({"fps": float(fps), "size": list(out_size), "n_frames": n_frames, "view": view, "seed": seed,
               "rotation": 0, "synthetic": True})
    return gt


def render_frames(scenario: str, duration: float = 20.0, fps: float = 30.0, size: tuple[int, int] = DEFAULT_SIZE,
                  view: str = "default", seed: int = 0, noise_sigma: float = 2.0, osd: bool = False
                  ) -> tuple[SynthFrames, dict]:
    """Ленивые кадры сценария + ground truth (без записи файла)."""
    script = SceneScript(scenario, duration, size)
    frames = SynthFrames(script, fps, view=view, seed=seed, noise_sigma=noise_sigma, osd=osd)
    return frames, _gt_dict(script, fps, view, len(frames), seed)


def _open_writer(path: Path, fps: float, size: tuple[int, int]):
    cv2 = _cv2()
    ext = path.suffix.lower()
    fourccs = ["MJPG", "XVID"] if ext == ".avi" else ["mp4v", "avc1", "MJPG"]
    for cc in fourccs:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*cc), fps, size)
        if writer.isOpened():
            return writer, cc
        writer.release()
    raise RuntimeError(f"OpenCV не смог открыть VideoWriter для {path} (пробовали {fourccs})")


def make_video(path: str | Path, scenario: str, duration: float = 20, fps: float = 30,
               size: tuple[int, int] = DEFAULT_SIZE, *, view: str = "default", seed: int = 0,
               noise_sigma: float = 2.0, osd: bool = False) -> dict:
    """Записать синтетический ролик сценария и вернуть ground truth.

    Возвращает dict: ``scenario, verdict, type_id, t, t_start, t_end, duration, description, fps, size [w, h],
    n_frames, view, seed, rotation, synthetic, path, fourcc``.
    """
    if view not in VIEWS:
        raise ValueError(f"неизвестный ракурс {view!r}; доступны: {', '.join(VIEWS)}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames, gt = render_frames(scenario, duration, fps, size, view, seed, noise_sigma, osd)
    out_size = tuple(gt["size"])
    writer, fourcc = _open_writer(path, fps, out_size)  # type: ignore[arg-type]
    try:
        for i in range(len(frames)):
            writer.write(frames.render(i))
    finally:
        writer.release()
    gt.update({"path": str(path), "fourcc": fourcc})
    return gt


def add_rotation_metadata(src: str | Path, dst: str | Path, degrees: int = 90) -> Optional[Path]:
    """Копия ролика с тегом поворота в метаданных (нужен ffmpeg в PATH); None — если ffmpeg недоступен."""
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-display_rotation", str(degrees), "-i", str(src), "-c", "copy",
           str(dst)]
    if subprocess.run(cmd, capture_output=True).returncode != 0:
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(src), "-c", "copy", "-metadata:s:v:0",
               f"rotate={degrees}", str(dst)]
        if subprocess.run(cmd, capture_output=True).returncode != 0:
            return None
    return Path(dst)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Синтетические ролики кассы (камера сверху)")
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--scenario", action="append", choices=SCENARIOS)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--view", default="default", choices=VIEWS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--osd", action="store_true", help="рисовать «часы камеры» поверх кадра")
    args = ap.parse_args(argv)
    scenarios = list(SCENARIOS) if args.all or not args.scenario else args.scenario
    out = Path(args.out)
    gts = []
    for sc in scenarios:
        suffix = "" if args.view == "default" else f"_{args.view}"
        gt = make_video(out / f"{sc}{suffix}.mov", sc, args.duration, args.fps, view=args.view, seed=args.seed,
                        osd=args.osd)
        gts.append(gt)
        print(f"{gt['path']}: {gt['verdict']} {gt['type_id'] or ''} t={gt['t']}")
    (out / "gt.json").write_text(json.dumps(gts, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
