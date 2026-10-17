"""Безопасный векторный парсер выражений правил (никакого ``eval``).

Выражение из ``rules.yaml`` разбирается через :mod:`ast`, проверяется по белому списку
узлов и компилируется в дерево замыканий, которое считает значение сразу для всего
временного ряда (numpy-массивы формы ``(N,)``).

Семантика (нечёткая логика, значения сигналов 0..1):

* ``a and b`` → ``min(a, b)``, ``a or b`` → ``max(a, b)``, ``not a`` → ``1 - a``
  (операнды логических операций обрезаются в 0..1);
* сравнения (``<``, ``<=``, ``>``, ``>=``, ``==``, ``!=``, цепочки ``0.2 < x < 0.8``) → 0.0/1.0;
* арифметика ``+ - * /`` (деление на 0 → 0), унарный минус;
* точечные функции: ``min``, ``max``, ``abs``, ``clip(x, lo, hi)``, ``ramp(x, lo, hi)``,
  ``step(x, thr)``;
* временные функции (окно — константа в секундах):
  ``was(x, w)`` — максимум за последние ``w`` с (было ли), ``held(x, w)`` — минимум за ``w`` с
  (держится ли непрерывно; при истории короче ``w`` → 0), ``avg(x, w)`` — среднее за ``w`` с,
  ``ahead(x, w)`` — максимум на ``w`` с вперёд, ``ever(x)`` — максимум с начала ролика.

Имена — сигналы из ``SIGNALS`` (или любые другие ключи ``SceneState.signals``) и
константы ``true``/``false``. Атрибуты, индексы, строки, лямбды, comprehension-ы,
вызовы чего-либо кроме белого списка — запрещены (``ExprError`` на этапе компиляции).
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional

import numpy as np

__all__ = [
    "ExprError",
    "EvalContext",
    "CompiledExpr",
    "compile_expr",
    "evaluate_scalar",
    "POINTWISE_FUNCS",
    "TEMPORAL_FUNCS",
]

MAX_EXPR_LEN = 2000
MAX_NODES = 300

POINTWISE_FUNCS = frozenset({"min", "max", "abs", "clip", "ramp", "step"})
TEMPORAL_FUNCS = frozenset({"was", "held", "avg", "ahead", "ever"})
_CONSTANT_NAMES = {"true": 1.0, "false": 0.0, "True": 1.0, "False": 0.0}

_CMP_SYMBOL = {
    ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!=",
}
_ALLOWED_NODES = (
    ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.USub, ast.UAdd,
    ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Compare, ast.Lt, ast.LtE, ast.Gt,
    ast.GtE, ast.Eq, ast.NotEq, ast.Name, ast.Load, ast.Constant, ast.Call,
)


class ExprError(ValueError):
    """Ошибка разбора/проверки выражения правила."""


@dataclass
class EvalContext:
    """Контекст вычисления: общая временная ось и ряды сигналов одинаковой длины."""

    times: np.ndarray
    signals: Mapping[str, np.ndarray]
    default: float = 0.0
    _window_cache: dict = field(default_factory=dict, repr=False)

    @property
    def n(self) -> int:
        return int(self.times.shape[0])

    def get(self, name: str) -> np.ndarray:
        arr = self.signals.get(name)
        if arr is None:
            return np.full(self.n, float(self.default))
        return arr

    def left_index(self, window_s: float) -> np.ndarray:
        key = ("L", float(window_s))
        if key not in self._window_cache:
            self._window_cache[key] = np.searchsorted(self.times, self.times - window_s - 1e-9, side="left")
        return self._window_cache[key]

    def right_index(self, window_s: float) -> np.ndarray:
        key = ("R", float(window_s))
        if key not in self._window_cache:
            self._window_cache[key] = np.searchsorted(self.times, self.times + window_s + 1e-9, side="right")
        return self._window_cache[key]


@dataclass(frozen=True)
class Comparison:
    """Сравнение вида ``signal <op> константа`` — для объяснимости (trace)."""

    name: str
    op: str
    threshold: float

    def check(self, value: float) -> bool:
        return bool(_apply_cmp(self.op, np.asarray([value], float), np.asarray([self.threshold], float))[0] > 0.5)


@dataclass
class CompiledExpr:
    source: str
    names: frozenset[str]
    funcs: frozenset[str]
    comparisons: tuple[Comparison, ...]
    max_window_s: float
    _fn: Callable[[EvalContext], Any] = field(repr=False)

    def __call__(self, ctx: EvalContext) -> np.ndarray:
        res = self._fn(ctx)
        arr = np.asarray(res, dtype=float)
        if arr.ndim == 0:
            arr = np.full(ctx.n, float(arr))
        arr = np.where(np.isfinite(arr), arr, 0.0)
        return arr


def _truth(x: Any) -> Any:
    return np.clip(x, 0.0, 1.0)


def _apply_cmp(op: str, a: Any, b: Any) -> Any:
    if op == "<":
        r = np.less(a, b)
    elif op == "<=":
        r = np.less_equal(a, b)
    elif op == ">":
        r = np.greater(a, b)
    elif op == ">=":
        r = np.greater_equal(a, b)
    elif op == "==":
        r = np.isclose(a, b, atol=1e-9)
    elif op == "!=":
        r = ~np.isclose(a, b, atol=1e-9)
    else:  # pragma: no cover - защищено белым списком
        raise ExprError(f"Недопустимый оператор сравнения: {op}")
    return np.asarray(r, dtype=float)


def _safe_div(a: Any, b: Any) -> Any:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    shape = np.broadcast(a, b).shape
    out = np.zeros(shape, dtype=float)
    np.divide(a, b, out=out, where=np.broadcast_to(b != 0, shape))
    return out


def _as_array(x: Any, n: int) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        return np.full(n, float(arr))
    return arr


def _window_reduce(ctx: EvalContext, x: Any, window_s: float, kind: str) -> np.ndarray:
    n = ctx.n
    arr = _as_array(x, n)
    if n == 0:
        return arr
    if kind == "ever":
        return np.maximum.accumulate(arr)
    if kind == "ahead":
        right = ctx.right_index(window_s)
        out = np.empty(n)
        for i in range(n):
            out[i] = arr[i:max(i + 1, int(right[i]))].max()
        return out
    left = ctx.left_index(window_s)
    if kind == "avg":
        cs = np.concatenate([[0.0], np.cumsum(arr)])
        idx = np.arange(n)
        cnt = idx + 1 - left
        return (cs[idx + 1] - cs[left]) / np.maximum(cnt, 1)
    out = np.empty(n)
    if kind == "was":
        for i in range(n):
            out[i] = arr[int(left[i]):i + 1].max()
        return out
    if kind == "held":
        t0 = float(ctx.times[0])
        for i in range(n):
            if float(ctx.times[i]) - t0 < window_s - 1e-6:
                out[i] = 0.0  # истории меньше окна — «держится» не доказано
            else:
                out[i] = arr[int(left[i]):i + 1].min()
        return out
    raise ExprError(f"Неизвестная временная функция: {kind}")  # pragma: no cover


class _Compiler:
    def __init__(self, allowed_names: Optional[set[str]]) -> None:
        self.allowed_names = allowed_names
        self.names: set[str] = set()
        self.funcs: set[str] = set()
        self.comparisons: list[Comparison] = []
        self.max_window = 0.0

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _const_number(node: ast.AST, what: str) -> float:
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -_Compiler._const_number(node.operand, what)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        raise ExprError(f"{what}: ожидалась числовая константа")

    # -- main ------------------------------------------------------------
    def compile(self, node: ast.AST) -> Callable[[EvalContext], Any]:
        if isinstance(node, ast.Expression):
            return self.compile(node.body)
        if isinstance(node, ast.Constant):
            v = node.value
            if isinstance(v, bool):
                fv = 1.0 if v else 0.0
            elif isinstance(v, (int, float)):
                fv = float(v)
            else:
                raise ExprError(f"Недопустимая константа: {v!r}")
            return lambda ctx, fv=fv: fv
        if isinstance(node, ast.Name):
            name = node.id
            if name in _CONSTANT_NAMES:
                fv = _CONSTANT_NAMES[name]
                return lambda ctx, fv=fv: fv
            if name.startswith("_"):
                raise ExprError(f"Недопустимое имя: {name}")
            if self.allowed_names is not None and name not in self.allowed_names:
                raise ExprError(f"Неизвестный сигнал: {name}")
            self.names.add(name)
            return lambda ctx, name=name: ctx.get(name)
        if isinstance(node, ast.BoolOp):
            parts = [self.compile(v) for v in node.values]
            if isinstance(node.op, ast.And):
                def _and(ctx: EvalContext, parts=parts) -> Any:
                    res = _truth(parts[0](ctx))
                    for p in parts[1:]:
                        res = np.minimum(res, _truth(p(ctx)))
                    return res
                return _and

            def _or(ctx: EvalContext, parts=parts) -> Any:
                res = _truth(parts[0](ctx))
                for p in parts[1:]:
                    res = np.maximum(res, _truth(p(ctx)))
                return res
            return _or
        if isinstance(node, ast.UnaryOp):
            inner = self.compile(node.operand)
            if isinstance(node.op, ast.Not):
                return lambda ctx, inner=inner: 1.0 - _truth(inner(ctx))
            if isinstance(node.op, ast.USub):
                return lambda ctx, inner=inner: -np.asarray(inner(ctx), dtype=float)
            if isinstance(node.op, ast.UAdd):
                return inner
            raise ExprError("Недопустимый унарный оператор")
        if isinstance(node, ast.BinOp):
            a, b = self.compile(node.left), self.compile(node.right)
            if isinstance(node.op, ast.Add):
                return lambda ctx, a=a, b=b: np.add(a(ctx), b(ctx))
            if isinstance(node.op, ast.Sub):
                return lambda ctx, a=a, b=b: np.subtract(a(ctx), b(ctx))
            if isinstance(node.op, ast.Mult):
                return lambda ctx, a=a, b=b: np.multiply(a(ctx), b(ctx))
            if isinstance(node.op, ast.Div):
                return lambda ctx, a=a, b=b: _safe_div(a(ctx), b(ctx))
            raise ExprError("Недопустимый арифметический оператор")
        if isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
            fns = [self.compile(o) for o in operands]
            ops = []
            for op_node, left, right in zip(node.ops, operands[:-1], operands[1:]):
                sym = _CMP_SYMBOL.get(type(op_node))
                if sym is None:
                    raise ExprError("Недопустимый оператор сравнения")
                ops.append(sym)
                self._record_comparison(left, sym, right)

            def _cmp(ctx: EvalContext, fns=fns, ops=ops) -> Any:
                vals = [f(ctx) for f in fns]
                res: Any = 1.0
                for i, sym in enumerate(ops):
                    res = np.minimum(res, _apply_cmp(sym, vals[i], vals[i + 1]))
                return res
            return _cmp
        if isinstance(node, ast.Call):
            return self._compile_call(node)
        raise ExprError(f"Недопустимая конструкция: {type(node).__name__}")

    def _record_comparison(self, left: ast.AST, sym: str, right: ast.AST) -> None:
        flip = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "==": "==", "!=": "!="}
        try:
            if isinstance(left, ast.Name) and left.id not in _CONSTANT_NAMES:
                self.comparisons.append(Comparison(left.id, sym, self._const_number(right, "")))
            elif isinstance(right, ast.Name) and right.id not in _CONSTANT_NAMES:
                self.comparisons.append(Comparison(right.id, flip[sym], self._const_number(left, "")))
        except ExprError:
            pass  # сравнение двух сигналов — в trace не выносим

    def _compile_call(self, node: ast.Call) -> Callable[[EvalContext], Any]:
        if not isinstance(node.func, ast.Name):
            raise ExprError("Вызывать можно только функции из белого списка")
        fname = node.func.id
        if node.keywords:
            raise ExprError(f"{fname}(): именованные аргументы не поддерживаются")
        if any(isinstance(a, ast.Starred) for a in node.args):
            raise ExprError(f"{fname}(): *args не поддерживаются")
        args = node.args
        if fname in POINTWISE_FUNCS:
            self.funcs.add(fname)
            fns = [self.compile(a) for a in args]
            if fname in ("min", "max"):
                if not fns:
                    raise ExprError(f"{fname}(): нужен хотя бы один аргумент")
                red = np.minimum if fname == "min" else np.maximum

                def _mm(ctx: EvalContext, fns=fns, red=red) -> Any:
                    res = fns[0](ctx)
                    for f in fns[1:]:
                        res = red(res, f(ctx))
                    return res
                return _mm
            if fname == "abs":
                if len(fns) != 1:
                    raise ExprError("abs(x): ровно один аргумент")
                return lambda ctx, f=fns[0]: np.abs(f(ctx))
            if fname == "clip":
                if len(fns) != 3:
                    raise ExprError("clip(x, lo, hi): ровно три аргумента")
                return lambda ctx, f=fns: np.clip(f[0](ctx), f[1](ctx), f[2](ctx))
            if fname == "ramp":
                if len(fns) != 3:
                    raise ExprError("ramp(x, lo, hi): ровно три аргумента")
                return lambda ctx, f=fns: np.clip(
                    _safe_div(np.subtract(f[0](ctx), f[1](ctx)), np.subtract(f[2](ctx), f[1](ctx))), 0.0, 1.0)
            if fname == "step":
                if len(fns) != 2:
                    raise ExprError("step(x, thr): ровно два аргумента")
                return lambda ctx, f=fns: _apply_cmp(">=", f[0](ctx), f[1](ctx))
        if fname in TEMPORAL_FUNCS:
            self.funcs.add(fname)
            if fname == "ever":
                if len(args) != 1:
                    raise ExprError("ever(x): ровно один аргумент")
                inner = self.compile(args[0])
                return lambda ctx, inner=inner: _window_reduce(ctx, inner(ctx), 0.0, "ever")
            if len(args) != 2:
                raise ExprError(f"{fname}(x, окно_с): ровно два аргумента")
            inner = self.compile(args[0])
            window = self._const_number(args[1], f"{fname}(): окно")
            if not (0.0 <= window <= 3600.0):
                raise ExprError(f"{fname}(): окно должно быть в диапазоне 0..3600 с")
            self.max_window = max(self.max_window, window)
            return lambda ctx, inner=inner, w=window, k=fname: _window_reduce(ctx, inner(ctx), w, k)
        raise ExprError(f"Функция не разрешена: {fname}")


def _check_tree(tree: ast.AST) -> None:
    count = 0
    for node in ast.walk(tree):
        count += 1
        if count > MAX_NODES:
            raise ExprError("Выражение слишком сложное")
        if isinstance(node, ast.Starred):
            raise ExprError("*args не поддерживаются")
        if not isinstance(node, _ALLOWED_NODES):
            raise ExprError(f"Недопустимая конструкция: {type(node).__name__}")


def compile_expr(source: Any, allowed_names: Optional[Iterable[str]] = None) -> CompiledExpr:
    """Скомпилировать выражение правила. Число/bool допускаются как константное выражение."""
    if isinstance(source, bool):
        source = "true" if source else "false"
    elif isinstance(source, (int, float)):
        source = repr(float(source))
    if not isinstance(source, str):
        raise ExprError(f"Выражение должно быть строкой, получено {type(source).__name__}")
    text = " ".join(source.split())
    if not text:
        raise ExprError("Пустое выражение")
    if len(text) > MAX_EXPR_LEN:
        raise ExprError("Выражение слишком длинное")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ExprError(f"Синтаксическая ошибка в выражении {text!r}: {exc.msg}") from None
    _check_tree(tree)
    comp = _Compiler(set(allowed_names) if allowed_names is not None else None)
    fn = comp.compile(tree)
    return CompiledExpr(
        source=text,
        names=frozenset(comp.names),
        funcs=frozenset(comp.funcs),
        comparisons=tuple(comp.comparisons),
        max_window_s=comp.max_window,
        _fn=fn,
    )


def evaluate_scalar(expr: CompiledExpr | str, variables: Mapping[str, float]) -> float:
    """Вычислить выражение для одного набора скалярных переменных (формулы confidence)."""
    compiled = compile_expr(expr) if isinstance(expr, str) else expr
    ctx = EvalContext(
        times=np.zeros(1),
        signals={k: np.asarray([float(v)], dtype=float) for k, v in variables.items()},
    )
    missing = sorted(n for n in compiled.names if n not in variables)
    if missing:
        raise ExprError(f"Нет переменных для формулы: {', '.join(missing)}")
    return float(compiled(ctx)[0])
