"""Тест-кит v2 под спецификацию поведения команды (A01). Синтетика проверяет МЕХАНИЗМЫ, а не качество модели.

Кейсы — пары «норма / нарушение» из спецификации (``CASES``), две планировки (``A``: горизонтальная перегородка;
``B``: вертикальная перегородка, стол вдоль неё, сейф внизу) и ориентации ``portrait``/``landscape``.

* Длительность и момент события берутся из ОДНОГО распределения для всех кейсов (``sample_timing`` не знает
  класс): длительность, позиция в ролике и имя файла не должны быть признаками.
* GT — список событий (``events``) с таймкодом по началу действия (переход: отошла, вошёл, поднесла);
  у одного кейса может быть несколько событий, дребезг одного события — одно событие.
* Неизвестно ≠ нет: сигнал, который нельзя наблюдать (темно, человека нет в кадре), в ``SceneState.signals``
  отсутствует; ``scene_visibility`` (EXTRA_SIGNALS) показывает освещённость.
* Никаких пиксельных координат в логике: стороны кассы/клиента — полигоны планировки (``zones``).

    from testkit.spec import make_suite, build_case, case_bundle, make_case_video
    for spec in make_suite(seed=0):
        case = build_case(spec)            # .gt, .at(t)
        b = case_bundle(spec)              # детекции/треки/зоны/позы/сигналы/ожидаемые события
"""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from contracts.extensions import EXTRA_SIGNALS, TYPE_REGISTRY
from contracts.sentinel_contracts import SIGNALS, Detection, Event, PoseFrame, SceneState, Track, Zone

from .fake_clip import sample_times
from .scenarios import (
    BODY_A,
    BODY_B,
    CASH_IN_SAFE,
    DEFAULT_SIZE,
    DOOR_HINGE,
    DOOR_LEN,
    HAND_R,
    LAYOUT,
    ItemState,
    PersonState,
    Snapshot,
    map_box,
    map_point,
    path_at,
    ramp,
)

__all__ = ["LAYOUTS", "CASES", "CASE_ORDER", "CLEAN_CASES", "CaseSpec", "SpecSnapshot", "SpecCase", "sample_timing", "make_suite", "build_case",
           "spec_signals", "case_bundle", "make_case_video", "ORIENTATIONS"]

ORIENTATIONS = {"portrait": "default", "landscape": "landscape"}


# ----------------------------------------------------------------------------- планировки

def _layout_a() -> dict[str, Any]:
    p = LAYOUT["partition"]
    return {**LAYOUT, "client_area": (0.0, p[3], 1.0, 1.0), "window": (0.30, p[1], 0.42, p[3]),
            "door_hinge": DOOR_HINGE, "door_len": DOOR_LEN, "door_closed_deg": 180.0, "safe_front": "bottom",
            "seat": (0.40, 0.30), "cashier_facing": 180.0, "client_spot": (0.35, 0.70), "client_facing": 0.0,
            "client2_spot": (0.62, 0.80), "safe_spot": (0.17, 0.30), "safe_facing": 0.0,
            "door_in": (0.84, 0.46), "door_out": (0.84, 0.66), "off_client": (0.84, 1.12), "off_back": (1.15, 0.28),
            "cashier_poly": [(0.0, 0.0), (1.0, 0.0), (1.0, p[1]), (0.0, p[1])],
            "client_poly": [(0.0, p[3]), (1.0, p[3]), (1.0, 1.0), (0.0, 1.0)],
            "tray_box": (0.305, 0.435, 0.385, 0.495), "cash_in_safe": CASH_IN_SAFE,
            "desk_bag_box": (0.50, 0.42, 0.62, 0.51), "jacket_box": (0.07, 0.405, 0.20, 0.53),
            "floor_bag_box": (0.73, 0.455, 0.83, 0.525), "handset_box": (0.10, 0.425, 0.16, 0.47)}


def _layout_b() -> dict[str, Any]:
    return {"safe_stand": (0.03, 0.82, 0.26, 0.97), "safe": (0.05, 0.83, 0.24, 0.96), "safe_front": "top",
            "desk": (0.32, 0.30, 0.48, 0.86), "laptop": (0.34, 0.50, 0.46, 0.62),
            "money_counter": (0.35, 0.37, 0.45, 0.44), "cash_tray": (0.34, 0.64, 0.46, 0.72),
            "desk_phone": (0.36, 0.76, 0.44, 0.82), "document": (0.35, 0.31, 0.45, 0.36),
            "partition": (0.495, 0.25, 0.505, 1.0), "door": (0.49, 0.04, 0.51, 0.25),
            "client_area": (0.505, 0.0, 1.0, 1.0), "window": (0.49, 0.55, 0.51, 0.65),
            "door_hinge": (0.50, 0.04), "door_len": 0.38, "door_closed_deg": 90.0,
            "seat": (0.20, 0.58), "cashier_facing": 90.0, "client_spot": (0.70, 0.60), "client_facing": 270.0,
            "client2_spot": (0.84, 0.40), "safe_spot": (0.15, 0.74), "safe_facing": 180.0,
            "door_in": (0.38, 0.14), "door_out": (0.64, 0.14), "off_client": (1.15, 0.30), "off_back": (-0.15, 0.40),
            "cashier_poly": [(0.0, 0.0), (0.495, 0.0), (0.495, 1.0), (0.0, 1.0)],
            "client_poly": [(0.505, 0.0), (1.0, 0.0), (1.0, 1.0), (0.505, 1.0)],
            "tray_box": (0.35, 0.65, 0.45, 0.71), "cash_in_safe": (0.10, 0.90, 0.18, 0.94),
            "desk_bag_box": (0.33, 0.44, 0.47, 0.52), "jacket_box": (0.33, 0.74, 0.47, 0.85),
            "floor_bag_box": (0.36, 0.17, 0.46, 0.23), "handset_box": (0.37, 0.77, 0.43, 0.81)}


LAYOUTS: dict[str, dict[str, Any]] = {"A": _layout_a(), "B": _layout_b()}


def _inside_poly(pt: tuple[float, float], poly: list[tuple[float, float]]) -> bool:
    x, y = pt
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def _box_center(b: tuple) -> tuple[float, float]:
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def _box_around(cx: float, cy: float, w: float, h: float, aspect: float) -> tuple:
    return (cx - w / 2, cy - h / 2 * aspect, cx + w / 2, cy + h / 2 * aspect)


# ----------------------------------------------------------------------------- план кейса

@dataclass
class GTEvent:
    type_id: str
    t_onset: float                    # начало действия (переход) — таймкод для жюри
    t_end: float
    alt_type_ids: list[str] = field(default_factory=list)  # допустимые альтернативные типы
    note: str = ""


@dataclass
class ItemPlan:
    label: str
    t0: float
    t1: float
    where: tuple                      # ("box", key|bbox) | ("hand", pid) | ("head", pid)
    near_head: bool = False


@dataclass
class ActorPlan:
    pid: int
    role: str
    keys: list[tuple[float, Any]]     # (t, spot-key | (x, y))
    facing: list[tuple[float, float, float]] = field(default_factory=list)  # (t0, t1, deg) переопределения
    hidden_hands: list[tuple[float, float]] = field(default_factory=list)
    hand_raise: list[tuple[float, float]] = field(default_factory=list)     # рука у лица (без предмета)


@dataclass
class Plan:
    verdict: str
    events: list[GTEvent]
    actors: list[ActorPlan]
    items: list[ItemPlan] = field(default_factory=list)
    safe_open: list[tuple[float, float]] = field(default_factory=list)
    door_open: list[tuple[float, float]] = field(default_factory=list)
    lights_off: list[tuple[float, float]] = field(default_factory=list)
    handover: list[tuple[float, float]] = field(default_factory=list)
    description: str = ""


@dataclass
class CaseSpec:
    case: str
    layout: str = "A"
    orientation: str = "portrait"
    duration: float = 30.0
    onset: float = 12.0
    seed: int = 0

    @property
    def view(self) -> str:
        return ORIENTATIONS[self.orientation]


# ----------------------------------------------------------------------------- кейсы спецификации

def _seated(pid: int = 1) -> ActorPlan:
    return ActorPlan(pid, "cashier", [(0.0, "seat")])


def _leave(T: float, D: float, ret: Optional[float] = None) -> ActorPlan:
    keys: list[tuple[float, Any]] = [(0.0, "seat"), (T, "seat"), (T + 2.2, "off_back")]
    if ret is not None and ret + 2.5 < D:
        keys += [(ret, "off_back"), (ret + 2.2, "seat")]
    return ActorPlan(1, "cashier", keys)


def _client(t_in: float, t_out: float, D: float, pid: int = 2, spot: str = "client_spot") -> ActorPlan:
    return ActorPlan(pid, "visitor", [(0.0, "off_client"), (max(0.0, t_in - 2.0), "off_client"), (t_in, spot),
                                      (min(D, t_out), spot), (min(D, t_out) + 2.5, "off_client")])


def _put_cash_away(T: float) -> tuple[ActorPlan, list[ItemPlan], list[tuple[float, float]]]:
    """Убрала наличные в сейф и закрыла его к моменту T-0.5 (стол пуст)."""
    cashier = ActorPlan(1, "cashier", [(0.0, "seat"), (T - 6.0, "seat"), (T - 4.5, "safe_spot"),
                                       (T - 2.0, "safe_spot"), (T - 0.6, "seat")],
                        facing=[(T - 4.6, T - 1.9, -1.0)])  # -1 => лицом к сейфу
    items = [ItemPlan("cash", 0.0, T - 6.0, ("box", "tray_box")), ItemPlan("cash", T - 6.0, T - 3.0, ("hand", 1)),
             ItemPlan("cash", T - 3.0, T - 2.2, ("box", "cash_in_safe"))]
    return cashier, items, [(T - 4.0, T - 2.4)]


def _case_plans() -> dict[str, Any]:
    def leave_clean(T, D):
        c, items, safe = _put_cash_away(T)
        c.keys += [(T, "seat"), (T + 2.2, "off_back")]
        return Plan("no_violation", [], [c], items, safe_open=safe,
                    description="убрала наличные в сейф, закрыла его и ушла — стол пуст (норма)")

    def leave_cash(T, D):
        return Plan("violation", [GTEvent("V02", T, D, note="ушла, наличные на столе")], [_leave(T, D)],
                    [ItemPlan("cash", 0.0, D, ("box", "tray_box"))],
                    description="ушла, на столе остались наличные")

    def leave_bag(T, D):
        return Plan("violation", [GTEvent("V02", T, D, ["V07"], "ушла, на столе инкассаторская сумка")],
                    [_leave(T, D)], [ItemPlan("cash_bag", 0.0, D, ("box", "desk_bag_box"))],
                    description="ушла, на столе осталась инкассаторская сумка")

    def leave_safe_open(T, D):
        c = ActorPlan(1, "cashier", [(0.0, "seat"), (T - 5.0, "seat"), (T - 3.5, "safe_spot"), (T - 2.0, "safe_spot"),
                                     (T - 0.6, "seat"), (T, "seat"), (T + 2.2, "off_back")],
                      facing=[(T - 3.6, T - 1.9, -1.0)])
        return Plan("violation", [GTEvent("V03", T, D, ["V02"], "ушла, сейф открыт")], [c],
                    safe_open=[(T - 3.0, D)], description="ушла, сейф остался открытым")

    def client_opposite(T, D):
        return Plan("no_violation", [], [_seated(), _client(T, T + 8.0, D)],
                    [ItemPlan("cash", 0.0, T + 2.0, ("box", "tray_box"))], handover=[(T + 2.0, T + 3.0)],
                    description="клиент напротив с клиентской стороны, обслуживание (норма)")

    def outsider_cashier_side(T, D):
        o = ActorPlan(2, "visitor", [(0.0, "off_client"), (T - 3.5, "off_client"), (T - 1.5, "door_out"),
                                     (T, "door_in"), (T + 1.5, "seat_near"), (T + 5.0, "seat_near"),
                                     (T + 6.5, "door_in"), (T + 7.5, "door_out"), (T + 9.5, "off_client")])
        return Plan("violation", [GTEvent("V01", T, T + 7.0, note="посторонний на стороне кассира")],
                    [_seated(), o], [ItemPlan("cash", 0.0, D, ("box", "tray_box"))], door_open=[(T - 2.0, T + 8.0)],
                    description="посторонний прошёл через дверь на сторону кассира")

    def second_outsider(T, D):
        return Plan("violation", [GTEvent("V01", T, T + 6.0, note="второй посторонний во время обслуживания")],
                    [_seated(), _client(1.0, D, D), _client(T, T + 6.0, D, pid=3, spot="client2_spot")],
                    [ItemPlan("cash", 0.0, D, ("box", "tray_box"))],
                    description="во время обслуживания подошёл второй посторонний")

    def cashier_enters_first(T, D):
        c = ActorPlan(1, "cashier", [(0.0, "off_back"), (0.5, "off_back"), (3.0, "seat")])
        return Plan("no_violation", [], [c, _client(T, T + 6.0, D)],
                    description="кассир входит первой в первые секунды, затем клиент (норма)")

    def client_first(T, D):
        c = ActorPlan(1, "cashier", [(0.0, "off_back"), (T, "off_back"), (T + 2.5, "seat")])
        return Plan("no_violation", [], [_client(1.0, D, D), c],
                    description="первым появляется клиент, кассир приходит позже (норма: роль — по зоне)")

    def lights_off_clean(T, D):
        p = leave_clean(T, D)
        p.lights_off = [(min(D, T + 3.5), D)]
        p.description = "убрала всё, закрыла сейф, ушла, свет выключили (норма)"
        return p

    def lights_off_cash(T, D):
        p = leave_cash(T, D)
        p.lights_off = [(min(D, T + 3.5), D)]
        p.description = "ушла, наличные на столе, затем свет выключили"
        return p

    def drinking(T, D):
        return Plan("violation", [GTEvent("V06", T, T + 3.0, note="поднесла стакан ко рту")], [_seated()],
                    [ItemPlan("cash", 0.0, D, ("box", "tray_box")), ItemPlan("cup", T - 1.5, T, ("hand", 1)),
                     ItemPlan("cup", T, T + 3.0, ("head", 1), near_head=True),
                     ItemPlan("cup", T + 3.0, D, ("box", "handset_box"))],
                    description="пьёт на рабочем месте при наличных")

    def hand_to_face(T, D):
        c = _seated()
        c.hand_raise = [(T, T + 3.0)]
        return Plan("no_violation", [], [c], [ItemPlan("cash", 0.0, D, ("box", "tray_box"))],
                    description="рука у лица без телефона и стакана (не сигнал)")

    def mobile(T, D):
        return Plan("violation", [GTEvent("V05", T, T + 5.0, note="мобильный в руке кассира")], [_seated()],
                    [ItemPlan("cash", 0.0, D, ("box", "tray_box")), ItemPlan("cell_phone", T, T + 1.0, ("hand", 1)),
                     ItemPlan("cell_phone", T + 1.0, T + 5.0, ("head", 1), near_head=True)],
                    description="мобильный телефон в руке и у уха при наличных")

    def desk_phone(T, D):
        return Plan("no_violation", [], [_seated()],
                    [ItemPlan("cash", 0.0, D, ("box", "tray_box")), ItemPlan("handset", T, T + 5.0, ("head", 1),
                                                                              near_head=True)],
                    description="разговор по стационарному телефону (норма)")

    def jacket_on_desk(T, D):
        return Plan("violation", [GTEvent("V09", T, D, note="личные вещи на кассовом столе")], [_seated()],
                    [ItemPlan("cash", 0.0, D, ("box", "tray_box")), ItemPlan("personal_item", T, D,
                                                                              ("box", "jacket_box"))],
                    description="куртка на кассовом столе при наличных")

    def bag_handover_proper(T, D):
        col = ActorPlan(3, "collector", [(0.0, "off_client"), (T - 4.0, "off_client"), (T - 2.5, "door_out"),
                                         (T - 1.0, "door_in"), (T, "seat_near"), (T + 1.0, "seat_near"),
                                         (T + 2.0, "door_in"), (T + 3.0, "door_out"), (T + 5.0, "off_client")])
        c = ActorPlan(1, "cashier", [(0.0, "seat"), (T + 0.5, "seat"), (T + 2.0, "safe_spot"), (T + 3.5, "safe_spot"),
                                     (T + 4.8, "seat")], facing=[(T + 1.9, T + 3.6, -1.0)])
        return Plan("no_violation", [], [c, col],
                    [ItemPlan("cash_bag", 0.0, T, ("hand", 3)), ItemPlan("cash_bag", T, T + 2.6, ("hand", 1)),
                     ItemPlan("cash_bag", T + 2.6, D, ("box", "cash_in_safe"))],
                    safe_open=[(T + 2.0, T + 3.4)], door_open=[(T - 3.0, T + 3.5)], handover=[(T - 0.3, T + 0.5)],
                    description="сумка из рук в руки и сразу в сейф (норма)")

    def bag_left(T, D):
        col = ActorPlan(3, "collector", [(0.0, "off_client"), (T - 4.0, "off_client"), (T - 2.5, "door_out"),
                                         (T - 1.0, "door_in"), (T, "door_in"), (T + 1.0, "door_out"),
                                         (T + 3.0, "off_client")])
        return Plan("violation", [GTEvent("V07", T, D, note="инкассаторская сумка оставлена без присмотра")],
                    [_seated(), col], [ItemPlan("cash_bag", 0.0, T, ("hand", 3)),
                                       ItemPlan("cash_bag", T, D, ("box", "floor_bag_box"))],
                    door_open=[(T - 3.0, T + 2.0)], description="инкассатор оставил сумку и ушёл")

    def safe_open_working(T, D):
        c = ActorPlan(1, "cashier", [(0.0, "seat"), (T - 3.0, "seat"), (T - 1.8, "safe_spot"), (T - 1.0, "safe_spot"),
                                     (T, "seat")], facing=[(T - 1.9, T - 0.9, -1.0)])
        return Plan("violation", [GTEvent("V03", T, D, note="работает за столом при открытом сейфе за спиной")],
                    [c], [ItemPlan("cash", 0.0, D, ("box", "tray_box"))], safe_open=[(T - 1.5, D)],
                    description="работа за столом при открытом сейфе за спиной")

    def multi_event(T, D):
        t2 = min(D - 3.0, T + 7.0)
        return Plan("violation", [GTEvent("V05", T, T + 4.0, note="мобильный"),
                                  GTEvent("V02", t2, D, note="затем ушла, наличные на столе")],
                    [_leave(t2, D)], [ItemPlan("cash", 0.0, D, ("box", "tray_box")),
                                      ItemPlan("cell_phone", T, T + 4.0, ("head", 1), near_head=True)],
                    description="два события: мобильный, затем уход с наличными на столе")

    def debounce_phone(T, D):
        return Plan("violation", [GTEvent("V05", T, T + 6.0, note="одно событие с кратким перерывом")],
                    [_seated()], [ItemPlan("cash", 0.0, D, ("box", "tray_box")),
                                  ItemPlan("cell_phone", T, T + 3.0, ("head", 1), near_head=True),
                                  ItemPlan("cell_phone", T + 3.0, T + 3.8, ("hand", 1)),
                                  ItemPlan("cell_phone", T + 3.8, T + 6.0, ("head", 1), near_head=True)],
                    description="телефон у уха, на миг опущен и снова у уха — одно событие")

    fns = {k: v for k, v in locals().items() if callable(v)}
    return {name: fns[name] for name in CASE_ORDER}


# Пары «норма / нарушение» спецификации (порядок фиксирован — воспроизводимость набора).
CASE_ORDER = ("leave_clean", "leave_cash", "leave_bag", "leave_safe_open", "client_opposite", "outsider_cashier_side",
              "second_outsider", "cashier_enters_first", "client_first", "lights_off_clean", "lights_off_cash",
              "drinking", "hand_to_face", "mobile", "desk_phone", "jacket_on_desk", "bag_handover_proper", "bag_left",
              "safe_open_working", "multi_event", "debounce_phone")
CASES: dict[str, Any] = _case_plans()
CLEAN_CASES = tuple(n for n in CASE_ORDER if CASES[n](20.0, 40.0).verdict == "no_violation")


# ----------------------------------------------------------------------------- тайминги и набор

def sample_timing(rng: random.Random, duration_range: tuple[float, float] = (14.0, 60.0)) -> tuple[float, float]:
    """(длительность, момент события) — ОДНО распределение для всех кейсов (класс сюда не передаётся)."""
    d = round(rng.uniform(*duration_range), 1)
    lo, hi = max(7.0, 0.3 * d), max(max(7.0, 0.3 * d) + 0.1, min(d - 6.5, 0.75 * d))
    return d, round(rng.uniform(lo, hi), 1)


def make_suite(seed: int = 0, cases: Optional[list[str]] = None, layouts: tuple[str, ...] = ("A", "B"),
               orientations: tuple[str, ...] = ("portrait", "landscape"), n_per_case: int = 1,
               duration_range: tuple[float, float] = (14.0, 60.0)) -> list[CaseSpec]:
    rng = random.Random(seed)
    out = []
    for name in cases or list(CASES):
        for lay in layouts:
            for ori in orientations:
                for _ in range(n_per_case):
                    d, t = sample_timing(rng, duration_range)
                    out.append(CaseSpec(name, lay, ori, d, t, rng.randrange(1 << 30)))
    return out


# ----------------------------------------------------------------------------- исполнение плана

@dataclass
class SpecSnapshot(Snapshot):
    lights: float = 1.0
    layout: Optional[dict] = None


class _GT:
    def __init__(self, data: dict):
        self._d = data

    def as_dict(self) -> dict:
        return dict(self._d)


class SpecCase:
    """Исполняемый кейс: ``at(t) -> SpecSnapshot``, ``gt`` (dict), ``size``, ``duration`` (совместим с SynthFrames)."""

    def __init__(self, spec: CaseSpec, size: tuple[int, int] = DEFAULT_SIZE):
        if spec.case not in CASES:
            raise ValueError(f"неизвестный кейс {spec.case!r}; доступны: {', '.join(CASES)}")
        if spec.layout not in LAYOUTS:
            raise ValueError(f"неизвестная планировка {spec.layout!r}")
        self.spec, self.size, self.duration = spec, size, float(spec.duration)
        self.aspect = size[0] / size[1]
        self.layout = dict(LAYOUTS[spec.layout])
        self.layout["seat_near"] = self._near_seat()
        self.plan: Plan = CASES[spec.case](float(spec.onset), self.duration)

    def _near_seat(self) -> tuple[float, float]:
        sx, sy = self.layout["seat"]
        a = math.radians(self.layout["cashier_facing"])
        return sx - 0.20 * math.cos(a) * 0 + 0.20 * (1 if sx < 0.5 else -1) * abs(math.cos(a)), \
            sy + 0.06 * (1 if abs(math.cos(a)) > 0.5 else -1) * abs(math.sin(a)) + 0.03 * abs(math.cos(a))

    @property
    def gt(self) -> _GT:
        p, s = self.plan, self.spec
        events = [{"type_id": e.type_id, "alt_type_ids": e.alt_type_ids, "t": round(e.t_onset, 1),
                   "t_start": round(e.t_onset, 1), "t_end": round(min(self.duration, e.t_end), 1), "note": e.note}
                  for e in p.events]
        first = events[0] if events else None
        return _GT({"scenario": s.case, "case": s.case, "layout": s.layout, "orientation": s.orientation,
                    "duration": self.duration, "verdict": p.verdict, "type_id": first["type_id"] if first else None,
                    "t": first["t"] if first else None, "t_start": first["t_start"] if first else None,
                    "t_end": first["t_end"] if first else None, "events": events,
                    "unknown_spans": [[round(a, 1), round(min(b, self.duration), 1)] for a, b in p.lights_off],
                    "description": p.description, "seed": s.seed})

    def _pos(self, key: Any) -> tuple[float, float]:
        return tuple(self.layout[key]) if isinstance(key, str) else tuple(key)  # type: ignore[return-value]

    def _actor(self, a: ActorPlan, t: float) -> Optional[PersonState]:
        keys = [(kt, *self._pos(k)) for kt, k in a.keys]
        x, y = path_at(keys, t)
        if not (-0.02 <= x <= 1.02 and -0.02 <= y <= 1.02):
            return None
        lay = self.layout
        default = {"cashier": lay["cashier_facing"], "visitor": lay["client_facing"],
                   "collector": lay["client_facing"]}.get(a.role, 0.0)
        facing = default
        for t0, t1, deg in a.facing:
            if t0 <= t <= t1:
                facing = lay["safe_facing"] if deg < 0 else deg
        x0, y0 = path_at(keys, t - 0.2)
        x1, y1 = path_at(keys, t + 0.2)
        p = PersonState(a.pid, a.role, x, y, facing, speed=math.hypot(x1 - x0, y1 - y0) / 0.4)
        p.hands = self._hands(p)
        if any(t0 <= t <= t1 for t0, t1 in a.hidden_hands):
            p.hands_visible = False
        for t0, t1 in a.hand_raise:
            u = ramp(t, t0, t1, fade=0.4)
            if u > 0:
                hx, hy = p.hands[1]
                head = self._head(p)
                p.hands[1] = (hx + (head[0] - hx) * u, hy + (head[1] - hy) * u)
        return p

    def _hands(self, p: PersonState) -> list[tuple[float, float]]:
        a = math.radians(p.facing)
        fx, fy = math.sin(a), -math.cos(a)
        rx, ry = -fy, fx
        return [(p.x + fx * 0.09 + rx * 0.08 * s, p.y + (fy * 0.09 + ry * 0.08 * s) * self.aspect) for s in (-1, 1)]

    def _head(self, p: PersonState) -> tuple[float, float]:
        a = math.radians(p.facing)
        rx, ry = math.cos(a), math.sin(a)
        return p.x + rx * 0.05, p.y + ry * 0.05 * self.aspect

    def at(self, t: float) -> SpecSnapshot:
        persons = [q for q in (self._actor(a, t) for a in self.plan.actors) if q is not None]
        by_pid = {q.pid: q for q in persons}
        items: list[ItemState] = []
        for it in self.plan.items:
            if not it.t0 <= t < it.t1:
                continue
            kind = it.where[0]
            if kind == "box":
                box = self.layout[it.where[1]] if isinstance(it.where[1], str) else it.where[1]
                items.append(ItemState(it.label, tuple(box)))
            else:
                holder = by_pid.get(it.where[1])
                if holder is None:
                    continue
                cx, cy = self._head(holder) if kind == "head" else holder.hands[1]
                w, h = {"cash": (0.07, 0.05), "cash_bag": (0.10, 0.07), "cell_phone": (0.035, 0.06),
                        "cup": (0.05, 0.05), "handset": (0.06, 0.03)}.get(it.label, (0.05, 0.05))
                items.append(ItemState(it.label, _box_around(cx, cy, w, h, self.aspect), holder=holder.pid,
                                       near_head=kind == "head"))
        safe = max((ramp(t, a, b, fade=0.5) for a, b in self.plan.safe_open), default=0.0)
        door = max((ramp(t, a, b, fade=0.6) for a, b in self.plan.door_open), default=0.0)
        lights = 1.0 - max((ramp(t, a, b, fade=0.3) for a, b in self.plan.lights_off), default=0.0)
        handover = 1.0 if any(a <= t <= b for a, b in self.plan.handover) else 0.0
        return SpecSnapshot(t=round(t, 4), persons=persons, items=items, safe_open=safe, door_open=door,
                            handover=handover, lights=lights, layout=self.layout)


def build_case(spec: CaseSpec, size: tuple[int, int] = DEFAULT_SIZE) -> SpecCase:
    return SpecCase(spec, size)


# ----------------------------------------------------------------------------- сигналы (неизвестно ≠ нет)

CASHIER_SIGNALS = ("phone_in_hand", "phone_to_ear", "eating_drinking", "hands_hidden", "cashier_facing_away")


def spec_signals(snap: SpecSnapshot) -> dict[str, float]:
    """Сигналы снимка. Не наблюдаемое (темно; кассира нет в кадре) — ОТСУТСТВУЕТ, а не 0."""
    lay = snap.layout or LAYOUTS["A"]
    out: dict[str, float] = {"scene_visibility": round(snap.lights, 3)}
    if snap.lights < 0.3:
        return out  # темно: всё остальное неизвестно
    cashier = snap.person("cashier")
    sx, sy = lay["seat"]
    in_work = bool(cashier and abs(cashier.x - sx) < 0.15 and abs(cashier.y - sy) < 0.10)
    outsiders = [p for p in snap.persons if p.role == "visitor"]
    on_desk = [i for i in snap.items if i.holder is None and _inside_poly(_box_center(i.bbox), lay["cashier_poly"])
               and _box_center(i.bbox)[0] >= lay["desk"][0] - 0.02 and _box_center(i.bbox)[0] <= lay["desk"][2] + 0.02
               and lay["desk"][1] - 0.02 <= _box_center(i.bbox)[1] <= lay["desk"][3] + 0.02]
    values_on_desk = [i for i in on_desk if i.label in ("cash", "cash_bag")]
    phone = next((i for i in snap.items if i.label == "cell_phone" and i.holder == 1), None)
    eat = [i for i in snap.items if i.label in ("cup", "food", "bottle") and i.holder == 1]
    out.update({
        "cashier_present": 1.0 if cashier else 0.0,
        "cashier_in_workplace": 1.0 if in_work else 0.0,
        "n_persons": float(len(snap.persons)),
        "outsider_in_cashier_area": 1.0 if any(_inside_poly((p.x, p.y), lay["cashier_poly"]) for p in outsiders)
        else 0.0,
        "second_outsider": 1.0 if len(outsiders) >= 2 else 0.0,
        "collector_present": 1.0 if snap.person("collector") else 0.0,
        "safe_open": round(snap.safe_open, 3),
        "door_open": round(snap.door_open, 3),
        "cash_visible": 1.0 if any(i.label == "cash" for i in snap.items) else 0.0,
        "cash_bag_visible": 1.0 if any(i.label == "cash_bag" for i in snap.items) else 0.0,
        "personal_items_on_desk": 1.0 if any(i.label == "personal_item" for i in on_desk) else 0.0,
        "workplace_unattended_with_cash": 1.0 if values_on_desk and not in_work else 0.0,
        "handover_motion": round(snap.handover, 3),
        "motion_energy": round(min(1.0, 2.0 * sum(p.speed for p in snap.persons)), 3),
    })
    if cashier is not None:  # иначе — неизвестно (сигналов нет)
        work = lay["cashier_facing"]
        diff = abs((cashier.facing - work + 180) % 360 - 180)
        out.update({
            "phone_in_hand": 1.0 if phone else 0.0,
            "phone_to_ear": 1.0 if phone and phone.near_head else 0.0,
            "eating_drinking": 1.0 if any(i.near_head for i in eat) else (0.3 if eat else 0.0),
            "hands_hidden": 0.0 if cashier.hands_visible else 1.0,
            "cashier_facing_away": 1.0 if diff > 120 else 0.0,
        })
    assert set(out) <= set(SIGNALS) | set(EXTRA_SIGNALS)
    return out


# ----------------------------------------------------------------------------- фикстуры кейса

@dataclass
class CaseBundle:
    spec: CaseSpec
    gt: dict
    zones: list[Zone]
    detections: list[Detection]
    tracks: list[Track]
    poses: list[PoseFrame]
    states: list[SceneState]
    events: list[Event]


def _person_box(p: PersonState, aspect: float) -> tuple:
    a = math.radians(p.facing)
    ex = math.hypot(BODY_A * math.cos(a), BODY_B * math.sin(a))
    ey = math.hypot(BODY_A * math.sin(a), BODY_B * math.cos(a)) * aspect
    x1, y1, x2, y2 = p.x - ex, p.y - ey, p.x + ex, p.y + ey
    if p.hands_visible:
        for hx, hy in p.hands:
            x1, x2 = min(x1, hx - HAND_R), max(x2, hx + HAND_R)
            y1, y2 = min(y1, hy - HAND_R * aspect), max(y2, hy + HAND_R * aspect)
    return tuple(min(1.0, max(0.0, v)) for v in (x1, y1, x2, y2))


def case_bundle(spec: CaseSpec, sample_fps: float = 5.0, size: tuple[int, int] = DEFAULT_SIZE) -> CaseBundle:
    """«Идеальные» выходы стадий для кейса: при темноте детекций нет (неизвестно), сигналы — без неизвестных."""
    case = SpecCase(spec, size)
    lay, view = case.layout, spec.view
    zones = [Zone(name, [tuple(min(1.0, max(0.0, c)) for c in map_point(pt, view, size)) for pt in poly], 0.9)
             for name, poly in (("cashier_area", lay["cashier_poly"]), ("client_area", lay["client_poly"]),
                                ("desk", _rect(lay["desk"])), ("safe", _rect(lay["safe_stand"])),
                                ("door", _rect(lay["door"])), ("cash_tray", _rect(lay["cash_tray"])))]
    dets: list[Detection] = []
    tracks: dict[int, Track] = {}
    poses: list[PoseFrame] = []
    states: list[SceneState] = []
    for t in sample_times(case.duration, sample_fps):
        snap = case.at(t)
        states.append(SceneState(t=t, signals=spec_signals(snap)))
        if snap.lights < 0.3:
            continue
        boxes = [("person", _person_box(p, case.aspect), p.pid, p.role) for p in snap.persons]
        boxes += [("safe_open" if snap.safe_open > 0.5 else "safe", lay["safe"], 101, None),
                  ("door_open" if snap.door_open > 0.5 else "door", lay["door"], 102, None),
                  ("laptop", lay["laptop"], 103, None), ("money_counter", lay["money_counter"], 104, None)]
        boxes += [(i.label, i.bbox, None, None) for i in snap.items if i.label != "handset"]
        for label, box, tid, role in boxes:
            b = map_box(tuple(box), view, size)
            if b[2] - b[0] <= 1e-4 or b[3] - b[1] <= 1e-4:
                continue
            dets.append(Detection(t=t, label=label, bbox=b, score=0.9, track_id=tid, source="testkit-spec"))
            if tid is not None:
                base = {"safe_open": "safe", "door_open": "door"}.get(label, label)
                tr = tracks.setdefault(tid, Track(tid, base, role=role or "unknown", role_conf=0.9 if role else 0.0))
                tr.boxes.append((t, b))
        for p in snap.persons:
            kps = [(min(1.0, max(0.0, map_point((x, y), view, size)[0])),
                    min(1.0, max(0.0, map_point((x, y), view, size)[1])), s) for x, y, s in _kps(p, case.aspect)]
            poses.append(PoseFrame(t=t, track_id=p.pid, keypoints=kps))
    events = [Event(type_id=e["type_id"], t_start=e["t_start"], t_end=e["t_end"], t_peak=e["t"], confidence=0.9,
                    source="testkit", facts=[e["note"]], extra={"alt_type_ids": e["alt_type_ids"], "case": spec.case})
              for e in case.gt.as_dict()["events"]]
    return CaseBundle(spec, case.gt.as_dict(), zones, dets, sorted(tracks.values(), key=lambda x: x.track_id), poses,
                      states, events)


def _rect(b: tuple) -> list[tuple[float, float]]:
    return [(b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])]


def _kps(p: PersonState, aspect: float) -> list[tuple[float, float, float]]:
    a = math.radians(p.facing)
    fx, fy = math.sin(a), -math.cos(a)
    rx, ry = -fy, fx
    hs = 0.85 if p.hands_visible else 0.05

    def pt(df: float, dr: float, s: float) -> tuple[float, float, float]:
        return (p.x + fx * df + rx * dr, p.y + (fy * df + ry * dr) * aspect, s)

    (lx, ly), (rx2, ry2) = p.hands
    ls, rs = pt(0, -BODY_A * 0.8, 0.9), pt(0, BODY_A * 0.8, 0.9)
    return [pt(0.04, 0, 0.6), pt(0.03, -0.015, 0.5), pt(0.03, 0.015, 0.5), pt(0, -0.045, 0.5), pt(0, 0.045, 0.5),
            ls, rs, ((ls[0] + lx) / 2, (ls[1] + ly) / 2, hs), ((rs[0] + rx2) / 2, (rs[1] + ry2) / 2, hs),
            (lx, ly, hs), (rx2, ry2, hs), pt(-0.02, -0.05, 0.3), pt(-0.02, 0.05, 0.3), pt(0.04, -0.04, 0.1),
            pt(0.04, 0.04, 0.1), pt(0.06, -0.04, 0.05), pt(0.06, 0.04, 0.05)]


def make_case_video(path: str | Path, spec: CaseSpec, fps: float = 10.0, size: tuple[int, int] = DEFAULT_SIZE,
                    noise_sigma: float = 2.0) -> dict:
    """Записать ролик кейса (кадры рендерятся тем же рендером, что v1) и вернуть GT."""
    from .synth_video import SynthFrames, _open_writer, view_transform

    case = SpecCase(spec, size)
    frames = SynthFrames(case, fps, view=spec.view, seed=spec.seed % 1000, noise_sigma=noise_sigma)  # type: ignore[arg-type]
    _, out_size = view_transform(spec.view, size)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer, fourcc = _open_writer(path, fps, out_size)
    try:
        for i in range(len(frames)):
            writer.write(frames.render(i))
    finally:
        writer.release()
    gt = case.gt.as_dict()
    gt.update({"path": str(path), "fps": fps, "size": list(out_size), "fourcc": fourcc, "spec": asdict(spec),
               "type_name": TYPE_REGISTRY.get(gt["type_id"] or "", None)})
    return gt
