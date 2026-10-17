"""Детерминированные сценарии синтетической кассы — единый источник истины тест-кита (A01).

Из одного ``SceneScript`` строятся и синтетическое видео (``synth_video``), и ряды
Detection/Track/Zone/SceneState/PoseFrame (``fixtures``), поэтому видео и «идеальные» выходы CV
согласованы между собой. Только numpy-free python: модуль импортируется без cv2/numpy.

Координаты — нормированные 0..1 в «базовом» ракурсе (портрет, камера сверху, x вправо, y вниз).
Другие ракурсы (``VIEWS``) получаются геометрическим преобразованием кадра и координат — см. ``view_transform``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

BBoxN = tuple[float, float, float, float]

SCENARIOS: tuple[str, ...] = ("clean", "clean_collector", "V01", "V02", "V03", "V04", "V05", "V06", "V07", "V08")
VIOLATION_SCENARIOS: tuple[str, ...] = tuple(s for s in SCENARIOS if s.startswith("V"))
VIEWS: tuple[str, ...] = ("default", "mirror", "rot180", "landscape", "perspective")

# --- геометрия базового ракурса (нормированные координаты) ---
LAYOUT: dict[str, BBoxN] = {
    "safe_stand": (0.04, 0.04, 0.29, 0.22),
    "safe": (0.06, 0.05, 0.27, 0.20),
    "desk": (0.06, 0.40, 0.68, 0.54),
    "laptop": (0.44, 0.415, 0.60, 0.505),
    "desk_phone": (0.09, 0.42, 0.17, 0.475),
    "money_counter": (0.19, 0.42, 0.28, 0.50),
    "cash_tray": (0.29, 0.42, 0.40, 0.51),
    "document": (0.61, 0.43, 0.67, 0.50),
    "partition": (0.00, 0.553, 0.70, 0.567),
    "door": (0.70, 0.548, 0.97, 0.572),
}
DOOR_HINGE = (0.97, 0.56)
DOOR_LEN = 0.27
PARTITION_Y = 0.56
SEAT = (0.40, 0.30)
WINDOW = (0.35, 0.70)
SAFE_FRONT = (0.17, 0.30)
OFF_BOTTOM_DOOR = (0.84, 1.12)
OFF_BOTTOM_WINDOW = (0.35, 1.12)
OFF_RIGHT = (1.15, 0.28)
CASH_ON_TRAY: BBoxN = (0.305, 0.435, 0.385, 0.495)
CASH_IN_SAFE: BBoxN = (0.12, 0.15, 0.20, 0.19)

# --- размеры человека (в долях ШИРИНЫ кадра, чтобы фигуры не растягивались в портрете) ---
BODY_A = 0.10   # полуось «плечи»
BODY_B = 0.055  # полуось «грудь–спина»
HEAD_R = 0.045
HAND_R = 0.018
ARM_LEN = 0.13

ROLE_COLORS = {  # BGR
    "cashier": (150, 95, 40),      # синяя форма
    "visitor": (60, 60, 170),      # бордовое пальто
    "collector": (70, 110, 85),    # оливковая форма инкассатора
}

DEFAULT_SIZE = (464, 848)  # (width, height), портрет как в датасете


@dataclass
class PersonState:
    pid: int
    role: str                       # cashier | visitor | collector
    x: float
    y: float
    facing: float                   # градусы: 180 — лицом «вниз» (к столу/клиенту), 0 — «вверх» (спиной)
    hands: list[tuple[float, float]] = field(default_factory=list)
    hands_visible: bool = True
    speed: float = 0.0              # нормированная скорость (для motion_energy)

    @property
    def center(self) -> tuple[float, float]:
        return self.x, self.y


@dataclass
class ItemState:
    label: str                      # из LABELS: cash, cash_bag, cell_phone, cup, food, ...
    bbox: BBoxN
    holder: Optional[int] = None    # pid держащего, None — лежит
    near_head: bool = False         # у уха/рта


@dataclass
class Snapshot:
    t: float
    persons: list[PersonState]
    items: list[ItemState]
    safe_open: float                # 0..1
    door_open: float                # 0..1
    handover: float = 0.0           # 0..1 — идёт передача ценностей

    def person(self, role: str) -> Optional[PersonState]:
        for p in self.persons:
            if p.role == role:
                return p
        return None


@dataclass
class GroundTruth:
    scenario: str
    verdict: str                    # violation | no_violation
    type_id: Optional[str]
    t: Optional[float]              # «секунда события», как в разметке датасета
    t_start: Optional[float]
    t_end: Optional[float]
    duration: float
    description: str

    def as_dict(self) -> dict:
        return {
            "scenario": self.scenario, "verdict": self.verdict, "type_id": self.type_id, "t": self.t,
            "t_start": self.t_start, "t_end": self.t_end, "duration": self.duration,
            "description": self.description,
        }


# ----------------------------------------------------------------------------- utils

def _lerp(a: float, b: float, u: float) -> float:
    return a + (b - a) * u


def _smooth(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


def path_at(keys: list[tuple[float, float, float]], t: float) -> tuple[float, float]:
    """Кусочно-линейная (со сглаживанием) траектория по ключам (t, x, y)."""
    if t <= keys[0][0]:
        return keys[0][1], keys[0][2]
    for (t0, x0, y0), (t1, x1, y1) in zip(keys, keys[1:]):
        if t0 <= t <= t1:
            u = _smooth((t - t0) / (t1 - t0)) if t1 > t0 else 1.0
            return _lerp(x0, x1, u), _lerp(y0, y1, u)
    return keys[-1][1], keys[-1][2]


def ramp(t: float, t_on: float, t_off: float, fade: float = 0.6) -> float:
    """0 -> 1 на [t_on, t_on+fade], 1 до t_off, 1 -> 0 на [t_off, t_off+fade]."""
    if t < t_on or t > t_off + fade:
        return 0.0
    if t < t_on + fade:
        return _smooth((t - t_on) / fade)
    if t <= t_off:
        return 1.0
    return 1.0 - _smooth((t - t_off) / fade)


def inside(p: tuple[float, float], margin: float = 0.0) -> bool:
    return -margin <= p[0] <= 1 + margin and -margin <= p[1] <= 1 + margin


def box_around(cx: float, cy: float, w: float, h: float, aspect: float) -> BBoxN:
    """Бокс w×h (в долях ширины) вокруг точки; aspect = width/height кадра."""
    hw, hh = w / 2, h / 2 * aspect
    return (cx - hw, cy - hh, cx + hw, cy + hh)


def clip_box(b: BBoxN) -> BBoxN:
    return (min(1.0, max(0.0, b[0])), min(1.0, max(0.0, b[1])), min(1.0, max(0.0, b[2])), min(1.0, max(0.0, b[3])))


# ----------------------------------------------------------------------------- script

class SceneScript:
    """Сценарий: ``at(t) -> Snapshot`` + ground truth. Детерминирован по (scenario, duration, size)."""

    def __init__(self, scenario: str, duration: float = 20.0, size: tuple[int, int] = DEFAULT_SIZE):
        if scenario not in SCENARIOS:
            raise ValueError(f"неизвестный сценарий {scenario!r}; доступны: {', '.join(SCENARIOS)}")
        self.scenario = scenario
        self.duration = float(duration)
        self.size = size
        self.aspect = size[0] / size[1]
        d = self.duration
        self.s = min(3.0, max(0.25, d / 20.0))  # масштаб длительностей событий
        self.T = round(min(max(0.45 * d, 1.5 * self.s + 0.5), max(0.5, d - 2.0 * self.s)), 1)

    # длительность в «секундах сценария» (масштабируется под короткие ролики)
    def _o(self, sec: float) -> float:
        return self.T + sec * self.s

    # ------------------------------------------------------------------ ground truth
    @property
    def gt(self) -> GroundTruth:
        sc, T, o = self.scenario, self.T, self._o
        windows = {
            "V01": (o(-1.5), o(4.0), "посторонний человек проходит за кассовый стол"),
            "V02": (T, o(5.0), "кассир уходит, наличные остаются на столе без присмотра"),
            "V03": (o(-1.0), o(5.0), "сейф открыт, кассир отошёл, у окна клиент"),
            "V04": (T, o(6.0), "дверь кассового помещения открыта, в проёме человек"),
            "V05": (o(-0.5), o(4.0), "кассир держит мобильный телефон у уха при наличных на столе"),
            "V06": (T, o(6.0), "кассир пьёт/ест на рабочем месте при наличных на столе"),
            "V07": (o(-1.0), o(6.0), "инкассаторская сумка оставлена на полу без присмотра"),
            "V08": (T, o(4.0), "кассир отворачивается, руки с наличными скрыты от камеры"),
        }
        if sc in windows:
            a, b, desc = windows[sc]
            return GroundTruth(sc, "violation", sc, round(T, 1), round(max(0.0, a), 1),
                               round(min(self.duration, b), 1), self.duration, desc)
        desc = {"clean": "обычное обслуживание клиента через окно",
                "clean_collector": "инкассатор передаёт сумку кассиру из рук в руки"}[sc]
        return GroundTruth(sc, "no_violation", None, None, None, None, self.duration, desc)

    # ------------------------------------------------------------------ people
    def _cashier(self, t: float) -> Optional[PersonState]:
        sc, o = self.scenario, self._o
        keys = [(0.0, *SEAT)]
        facing = 180.0
        if sc == "V02":
            keys = [(0.0, *SEAT), (o(-2.0), *SEAT), (o(-1.0), 0.80, 0.30), (self.T, *OFF_RIGHT),
                    (o(5.0), *OFF_RIGHT), (o(6.0), 0.80, 0.30), (o(7.0), *SEAT)]
        elif sc == "V03":
            keys = [(0.0, *SEAT), (o(-5.0), *SEAT), (o(-4.0), *SAFE_FRONT), (o(-2.0), *SAFE_FRONT),
                    (o(-1.0), *SEAT), (o(4.5), *SEAT), (o(5.0), *SAFE_FRONT), (o(6.5), *SAFE_FRONT),
                    (o(7.0), *SEAT)]
        x, y = path_at(keys, t)
        if sc == "V03" and abs(x - SAFE_FRONT[0]) < 0.03 and abs(y - SAFE_FRONT[1]) < 0.03:
            facing = 0.0  # лицом к сейфу
        if sc == "V08":
            facing = 180.0 - 180.0 * ramp(t, o(-1.0), o(4.0), fade=0.8)
        p = PersonState(1, "cashier", x, y, facing)
        if not inside((x, y), margin=0.02):
            return None
        p.speed = self._speed(keys, t)
        # руки: перед собой, при пересчёте — небольшое покачивание
        counting = 0.015 * math.sin(2 * math.pi * 1.5 * t) if self._counting(t) else 0.0
        p.hands = self._hands(p, spread=0.08, reach=0.09 + counting)
        if sc == "V08" and ramp(t, o(-0.2), o(4.0), fade=0.01) > 0.5:
            p.hands_visible = False
        return p

    def _counting(self, t: float) -> bool:
        return self.scenario in ("clean", "V05", "V06") and 0.15 * self.duration < t < 0.65 * self.duration

    def _hands(self, p: PersonState, spread: float, reach: float) -> list[tuple[float, float]]:
        a = math.radians(p.facing)
        fx, fy = math.sin(a), -math.cos(a)  # единичный вектор «вперёд» (facing=180 -> вниз)
        rx, ry = -fy, fx                    # «вправо» относительно взгляда
        out = []
        for side in (-1, 1):
            hx = p.x + (fx * reach + rx * spread * side)
            hy = p.y + (fy * reach + ry * spread * side) * self.aspect
            out.append((hx, hy))
        return out

    def _speed(self, keys: list[tuple[float, float, float]], t: float, dt: float = 0.2) -> float:
        x0, y0 = path_at(keys, t - dt)
        x1, y1 = path_at(keys, t + dt)
        return math.hypot(x1 - x0, (y1 - y0) / max(self.aspect, 1e-6) * self.aspect) / (2 * dt)

    def _visitor(self, t: float) -> Optional[PersonState]:
        sc, o, d = self.scenario, self._o, self.duration
        keys: list[tuple[float, float, float]]
        if sc in ("clean", "V03", "V05", "V06", "V08"):
            keys = [(0.0, *OFF_BOTTOM_WINDOW), (0.10 * d, *OFF_BOTTOM_WINDOW), (0.22 * d, *WINDOW),
                    (0.72 * d, *WINDOW), (0.86 * d, *OFF_BOTTOM_WINDOW)]
            if sc == "V03":  # клиент у окна всё время, пока сейф открыт
                keys = [(0.0, *OFF_BOTTOM_WINDOW), (0.05 * d, *OFF_BOTTOM_WINDOW), (0.15 * d, *WINDOW),
                        (max(0.80 * d, o(5.5)), *WINDOW), (d, *OFF_BOTTOM_WINDOW)]
        elif sc == "V01":
            keys = [(0.0, *OFF_BOTTOM_DOOR), (o(-4.5), *OFF_BOTTOM_DOOR), (o(-3.0), 0.84, 0.66),
                    (o(-1.5), 0.84, 0.46), (self.T, 0.64, 0.33), (o(3.0), 0.64, 0.33), (o(4.0), 0.84, 0.46),
                    (o(5.0), 0.84, 0.66), (o(6.5), *OFF_BOTTOM_DOOR)]
        elif sc == "V04":
            keys = [(0.0, *OFF_BOTTOM_DOOR), (o(-1.0), *OFF_BOTTOM_DOOR), (o(0.5), 0.84, 0.60),
                    (o(5.0), 0.84, 0.60), (o(6.5), *OFF_BOTTOM_DOOR)]
        else:
            return None
        x, y = path_at(keys, t)
        if not inside((x, y), margin=0.02):
            return None
        facing = 0.0  # лицом к кассе (вверх)
        p = PersonState(2, "visitor", x, y, facing, speed=self._speed(keys, t))
        p.hands = self._hands(p, spread=0.075, reach=0.06)
        return p

    def _collector(self, t: float) -> Optional[PersonState]:
        sc, o = self.scenario, self._o
        if sc == "V07":
            keys = [(0.0, *OFF_BOTTOM_DOOR), (o(-5.0), *OFF_BOTTOM_DOOR), (o(-4.0), 0.84, 0.66),
                    (o(-3.0), 0.84, 0.46), (o(-2.0), 0.80, 0.42), (o(-1.0), 0.80, 0.42), (o(0.0), 0.84, 0.46),
                    (o(1.0), 0.84, 0.66), (o(2.5), *OFF_BOTTOM_DOOR)]
        elif sc == "clean_collector":
            keys = [(0.0, *OFF_BOTTOM_DOOR), (o(-5.0), *OFF_BOTTOM_DOOR), (o(-4.0), 0.84, 0.66),
                    (o(-3.0), 0.84, 0.46), (o(-2.0), 0.58, 0.36), (o(1.0), 0.58, 0.36), (o(2.0), 0.84, 0.46),
                    (o(3.0), 0.84, 0.66), (o(4.5), *OFF_BOTTOM_DOOR)]
        else:
            return None
        x, y = path_at(keys, t)
        if not inside((x, y), margin=0.02):
            return None
        p = PersonState(3, "collector", x, y, 270.0 if sc == "clean_collector" else 0.0, speed=self._speed(keys, t))
        p.hands = self._hands(p, spread=0.075, reach=0.07)
        return p

    # ------------------------------------------------------------------ objects
    def _door_open(self, t: float) -> float:
        sc, o = self.scenario, self._o
        if sc == "V01":
            return ramp(t, o(-3.5), o(5.5))
        if sc == "V04":
            return ramp(t, self.T - 0.6, o(6.0))
        if sc == "V07":
            return ramp(t, o(-4.5), o(1.5))
        if sc == "clean_collector":
            return ramp(t, o(-4.5), o(3.5))
        return 0.0

    def _safe_open(self, t: float) -> float:
        if self.scenario == "V03":
            return ramp(t, self._o(-3.5), self._o(6.0), fade=0.5)
        return 0.0

    def _items(self, t: float, cashier: Optional[PersonState], visitor: Optional[PersonState],
               collector: Optional[PersonState]) -> tuple[list[ItemState], float]:
        sc, o, d, a = self.scenario, self._o, self.duration, self.aspect
        items: list[ItemState] = []
        handover = 0.0
        hand = cashier.hands[1] if cashier and cashier.hands else None
        head = (cashier.x + 0.06, cashier.y - 0.005) if cashier else None

        # --- наличные ---
        if sc == "clean":
            if 0.15 * d <= t < 0.45 * d:
                items.append(ItemState("cash", CASH_ON_TRAY))
            elif 0.45 * d <= t < 0.52 * d and visitor:  # передача клиенту через окно
                u = (t - 0.45 * d) / (0.07 * d)
                cx, cy = _lerp(0.345, WINDOW[0], u), _lerp(0.465, WINDOW[1] - 0.06, u)
                items.append(ItemState("cash", box_around(cx, cy, 0.07, 0.05, a), holder=1))
                handover = 1.0
        elif sc in ("V02", "V05", "V06"):
            items.append(ItemState("cash", CASH_ON_TRAY))
        elif sc == "V01":
            if t < o(5.0):
                items.append(ItemState("cash", CASH_ON_TRAY))
        elif sc == "V08":
            if t < o(-1.0) or t >= o(5.0):
                items.append(ItemState("cash", CASH_ON_TRAY))
            elif (t < o(-0.2) or t >= o(4.0)) and hand:  # пачка в руке
                items.append(ItemState("cash", box_around(*hand, 0.07, 0.05, a), holder=1))
            # T-0.2..T+4: пачка и руки скрыты от камеры
        elif sc == "V03":
            if self._safe_open(t) > 0.5:
                items.append(ItemState("cash", CASH_IN_SAFE))
        elif sc == "clean_collector" and cashier:
            items.append(ItemState("cash", CASH_ON_TRAY))

        # --- телефон (V05) ---
        if sc == "V05" and cashier and hand and o(-1.5) <= t < o(5.0):
            to_ear = ramp(t, o(-0.5), o(4.0), fade=0.4)
            px, py = _lerp(hand[0], head[0], to_ear), _lerp(hand[1], head[1], to_ear)
            items.append(ItemState("cell_phone", box_around(px, py, 0.035, 0.06, a), holder=1, near_head=to_ear > 0.5))

        # --- еда/напитки (V06) ---
        if sc == "V06" and cashier and hand:
            if o(-1.0) <= t < o(2.5):
                to_mouth = ramp(t, o(-0.5), o(2.0), fade=0.4)
                cx, cy = _lerp(hand[0], head[0] - 0.03, to_mouth), _lerp(hand[1], head[1] + 0.01, to_mouth)
                items.append(ItemState("cup", box_around(cx, cy, 0.05, 0.05, a), holder=1, near_head=to_mouth > 0.5))
            elif t >= o(2.5):
                items.append(ItemState("cup", box_around(0.62, 0.395, 0.05, 0.05, a)))
            if o(3.5) <= t < o(6.0):
                to_mouth = ramp(t, o(4.0), o(5.5), fade=0.4)
                cx, cy = _lerp(hand[0], head[0] - 0.03, to_mouth), _lerp(hand[1], head[1] + 0.01, to_mouth)
                items.append(ItemState("food", box_around(cx, cy, 0.06, 0.04, a), holder=1, near_head=to_mouth > 0.5))

        # --- инкассаторская сумка ---
        if sc == "V07":
            drop_t = o(-1.0)
            if collector and t < drop_t:
                h = collector.hands[0]
                items.append(ItemState("cash_bag", box_around(h[0], h[1] + 0.02, 0.10, 0.07, a), holder=3))
            elif t >= drop_t and (collector or t > o(-0.5)):
                items.append(ItemState("cash_bag", box_around(0.78, 0.49, 0.10, 0.07, a)))
        elif sc == "clean_collector":
            give_t = o(0.0)
            if collector and t < give_t:
                h = collector.hands[0]
                items.append(ItemState("cash_bag", box_around(h[0], h[1] + 0.02, 0.10, 0.07, a), holder=3))
                if t > o(-1.0):
                    handover = 1.0
            elif t >= give_t and cashier:
                items.append(ItemState("cash_bag", box_around(0.52, 0.36, 0.10, 0.07, a), holder=1))
                if t < o(0.8):
                    handover = 1.0
        return items, handover

    # ------------------------------------------------------------------ public
    def at(self, t: float) -> Snapshot:
        cashier = self._cashier(t)
        visitor = self._visitor(t)
        collector = self._collector(t)
        persons = [p for p in (cashier, visitor, collector) if p is not None]
        items, handover = self._items(t, cashier, visitor, collector)
        return Snapshot(t=round(t, 4), persons=persons, items=items, safe_open=self._safe_open(t),
                        door_open=self._door_open(t), handover=handover)


# ----------------------------------------------------------------------------- views

def _solve_homography(src: list[tuple[float, float]], dst: list[tuple[float, float]]) -> list[list[float]]:
    """3x3 гомография по 4 точкам (метод Гаусса, без numpy)."""
    rows, rhs = [], []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        rhs.append(u)
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        rhs.append(v)
    n = 8
    m = [r[:] + [b] for r, b in zip(rows, rhs)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(m[r][c]))
        m[c], m[piv] = m[piv], m[c]
        for r in range(n):
            if r != c and m[r][c] != 0:
                f = m[r][c] / m[c][c]
                m[r] = [a - f * b for a, b in zip(m[r], m[c])]
    h = [m[i][n] / m[i][i] for i in range(n)]
    return [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]]


def view_transform(view: str, size: tuple[int, int] = DEFAULT_SIZE) -> tuple[list[list[float]], tuple[int, int]]:
    """Гомография (в пикселях) базового кадра -> кадр ракурса ``view`` и размер выходного кадра (w, h)."""
    w, h = size
    if view == "default":
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], (w, h)
    if view == "mirror":
        return [[-1.0, 0.0, w - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], (w, h)
    if view == "rot180":
        return [[-1.0, 0.0, w - 1.0], [0.0, -1.0, h - 1.0], [0.0, 0.0, 1.0]], (w, h)
    if view == "landscape":  # поворот на 90° по часовой: (x, y) -> (h-1-y, x)
        return [[0.0, -1.0, h - 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], (h, w)
    if view == "perspective":  # камера наклонена и чуть повёрнута
        src = [(0.0, 0.0), (w - 1.0, 0.0), (w - 1.0, h - 1.0), (0.0, h - 1.0)]
        dst = [(0.14 * w, 0.05 * h), (0.93 * w, 0.0), (w - 1.0, h - 1.0), (0.04 * w, 0.93 * h)]
        return _solve_homography(src, dst), (w, h)
    raise ValueError(f"неизвестный ракурс {view!r}; доступны: {', '.join(VIEWS)}")


def map_point(pt: tuple[float, float], view: str, size: tuple[int, int] = DEFAULT_SIZE) -> tuple[float, float]:
    """Нормированная точка базового ракурса -> нормированная точка ракурса ``view``."""
    if view == "default":
        return pt
    m, (ow, oh) = view_transform(view, size)
    w, h = size
    x, y = pt[0] * (w - 1), pt[1] * (h - 1)
    u = m[0][0] * x + m[0][1] * y + m[0][2]
    v = m[1][0] * x + m[1][1] * y + m[1][2]
    s = m[2][0] * x + m[2][1] * y + m[2][2]
    return (u / s) / (ow - 1), (v / s) / (oh - 1)


def map_box(b: BBoxN, view: str, size: tuple[int, int] = DEFAULT_SIZE) -> BBoxN:
    if view == "default":
        return clip_box(b)
    pts = [map_point(p, view, size) for p in ((b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3]))]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return clip_box((min(xs), min(ys), max(xs), max(ys)))


def map_polygon(poly: list[tuple[float, float]], view: str, size: tuple[int, int] = DEFAULT_SIZE
                ) -> list[tuple[float, float]]:
    return [tuple(min(1.0, max(0.0, c)) for c in map_point(p, view, size)) for p in poly]  # type: ignore[misc]


def person_box(p: PersonState, aspect: float) -> BBoxN:
    """Бокс человека: повёрнутый эллипс тела + голова + видимые руки."""
    a = math.radians(p.facing)
    ex = math.hypot(BODY_A * math.cos(a), BODY_B * math.sin(a))
    ey = math.hypot(BODY_A * math.sin(a), BODY_B * math.cos(a)) * aspect
    x1, y1, x2, y2 = p.x - ex, p.y - ey, p.x + ex, p.y + ey
    if p.hands_visible:
        for hx, hy in p.hands:
            x1, x2 = min(x1, hx - HAND_R), max(x2, hx + HAND_R)
            y1, y2 = min(y1, hy - HAND_R * aspect), max(y2, hy + HAND_R * aspect)
    return clip_box((x1, y1, x2, y2))


def bbox_of_layout(name: str) -> BBoxN:
    return LAYOUT[name]
