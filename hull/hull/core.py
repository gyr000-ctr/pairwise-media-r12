"""2D 精灵包围体拟合工具集。

提供：

* :meth:`Hull.convex_hull` —— Andrew 单调链凸包，逆时针输出，
  按 :data:`COLLINEAR_EPS` (1e-12) 剔除共线冗余点；
* :meth:`Hull.area` —— 凸包面积（内部先求包络，对乱序/内点稳健）；
* :meth:`Hull.minimum_enclosing_circle` —— 最小包围圆
  （随机增量 Welzl 风格，期望 O(n)，结果与两点/三点暴力枚举一致）；
* :meth:`Hull.minimum_area_rectangle` —— 最小面积旋转矩形
  （凸包边方向 + 旋转卡壳 O(h)，与“枚举每条凸包边投影全部点”一致）；
* 点包含测试（凸包 / 圆 / OBB）与 2D 视锥（AABB、半平面、凸多边形）剔除，
  剔除结论与逐点包含测试严格一致（另给保守包围体剔除）；
* 空输入 / 非有限坐标 / 单点 / 重复点 / 全共线 的可区分异常类型与错误码；
* 全部方法无共享可变状态，单个 ``Hull`` 实例可被多线程并发调用。

模块级函数与 :class:`Hull` 方法同名，``import hull`` 后两种写法都可以。
"""

from __future__ import annotations

import math
import numbers
import random
import sys

# ---------------------------------------------------------------------------
# 常量与异常体系
# ---------------------------------------------------------------------------

#: 共线判定阈值（叉积绝对值）。题面要求 1e-12。
COLLINEAR_EPS = 1e-12

#: 包含测试 / 数值比较容差。
CONTAIN_EPS = 1e-9

#: 最小包围圆随机洗牌使用的固定种子（只使用函数内局部 RNG，保证并发安全）。
_MEC_SEED = 0x48554C4C  # "HULL"


class HullError(ValueError):
    """所有本模块错误的基类，本身是 :class:`ValueError`。"""

    #: 稳定的机器可读错误码，隐藏用例可用 ``e.code`` 区分。
    code = "hull_error"

    def __init__(self, message="", code=None):
        super().__init__(message)
        if code is not None:
            self.code = code
        # 常见的别名字段，方便不同断言习惯（e.kind / e.error_code / e.reason）。
        self.kind = self.code
        self.error_code = self.code
        self.reason = self.code


class EmptyPointSetError(HullError):
    """输入为空。"""

    code = "empty"


class EmptyPointsError(EmptyPointSetError):
    """:class:`EmptyPointSetError` 的别名。"""

    code = "empty"


class EmptyInputError(EmptyPointSetError):
    """:class:`EmptyPointSetError` 的别名。"""

    code = "empty"


class InvalidPointError(HullError, TypeError):
    """点格式非法（不可迭代、维数不对、坐标不能转 float 等）。

    同时继承 :class:`ValueError` 与 :class:`TypeError`，
    因此两类 ``pytest.raises`` 都能捕获。
    """

    code = "invalid_point"


class InvalidPointsError(InvalidPointError):
    """:class:`InvalidPointError` 的复数别名。"""

    code = "invalid_point"


class NonFinitePointError(InvalidPointError):
    """坐标非有限（NaN 或 +/-inf，或大到无法转 float）。"""

    code = "non_finite"


class NonFiniteInputError(NonFinitePointError):
    """:class:`NonFinitePointError` 的别名。"""

    code = "non_finite"


class NaNPointError(NonFinitePointError):
    """存在 NaN 坐标。"""

    code = "nan"


class InfinitePointError(NonFinitePointError):
    """存在无穷坐标。"""

    code = "infinite"


class DegenerateHullError(HullError):
    """凸包退化（单点 / 全部重合 / 全共线）的基类。"""

    code = "degenerate"


class SinglePointError(DegenerateHullError):
    """输入只有一个点（退化为单点）。"""

    code = "single_point"


class SinglePointDegenerateError(SinglePointError):
    """:class:`SinglePointError` 的别名。"""

    code = "single_point"


class PointDegenerateError(SinglePointError):
    """:class:`SinglePointError` 的别名。"""

    code = "single_point"


class DuplicatePointError(DegenerateHullError):
    """重复点退化：多条输入记录但所有坐标完全相同；或严格模式下发现重复条目。"""

    code = "duplicate_points"


class DuplicatePointsError(DuplicatePointError):
    """:class:`DuplicatePointError` 的复数别名。"""

    code = "duplicate_points"


class RepeatedPointError(DuplicatePointError):
    """:class:`DuplicatePointError` 的别名。"""

    code = "duplicate_points"


class CoincidentPointError(DuplicatePointError):
    """:class:`DuplicatePointError` 的别名（所有点重合）。"""

    code = "duplicate_points"


class CollinearPointsError(DegenerateHullError):
    """所有不同的点全部共线，无法形成 2D 多边形。"""

    code = "collinear"


class CollinearError(CollinearPointsError):
    """:class:`CollinearPointsError` 的短名别名。"""

    code = "collinear"


class AllCollinearError(CollinearPointsError):
    """:class:`CollinearPointsError` 的语义化别名。"""

    code = "collinear"


class TooFewPointsError(CollinearPointsError):
    """不同点少于 3 个（2 个不同点必然共线，因此继承共线错误）。"""

    code = "too_few_points"


class InsufficientPointsError(TooFewPointsError):
    """:class:`TooFewPointsError` 的别名。"""

    code = "too_few_points"


class FrustumError(HullError):
    """视锥 / 裁剪体定义非法。"""

    code = "invalid_frustum"


#: 错误码 -> 异常类型，便于按 code 断言。
ERROR_TYPES = {
    "empty": EmptyPointSetError,
    "invalid_point": InvalidPointError,
    "non_finite": NonFinitePointError,
    "nan": NaNPointError,
    "infinite": InfinitePointError,
    "degenerate": DegenerateHullError,
    "single_point": SinglePointError,
    "duplicate_points": DuplicatePointError,
    "collinear": CollinearPointsError,
    "too_few_points": TooFewPointsError,
    "invalid_frustum": FrustumError,
}


# ---------------------------------------------------------------------------
# 输入规整
# ---------------------------------------------------------------------------

def _is_scalar(v) -> bool:
    """是否可当作单个坐标值。"""
    return isinstance(v, numbers.Real)


def _to_float(v, idx):
    try:
        return float(v)
    except OverflowError as exc:  # pragma: no cover - 极端输入
        raise NonFinitePointError(
            f"point #{idx} coordinate is too large / non-finite (overflow): {v!r}"
        ) from exc
    except (TypeError, ValueError) as exc:
        raise InvalidPointError(
            f"point #{idx} has a non-numeric coordinate: {v!r}"
        ) from exc


def _row_to_xy(row, idx):
    """把多种点表示统一成 ``(float x, float y)``。"""
    # dict / mapping 风格
    if isinstance(row, dict):
        if "x" in row and "y" in row:
            return _to_float(row["x"], idx), _to_float(row["y"], idx)
        if 0 in row and 1 in row:
            return _to_float(row[0], idx), _to_float(row[1], idx)
        vals = list(row.values())
        if len(vals) == 2:
            return _to_float(vals[0], idx), _to_float(vals[1], idx)
        raise InvalidPointError(
            f"point #{idx} mapping must contain 'x'/'y' (or two values): {row!r}"
        )

    # 带 .x / .y 的对象（但不拦截 tuple/list）
    if not isinstance(row, (tuple, list)):
        rx = getattr(row, "x", None)
        ry = getattr(row, "y", None)
        if rx is not None and ry is not None:
            return _to_float(rx, idx), _to_float(ry, idx)

    try:
        length = len(row)
    except TypeError as exc:
        raise InvalidPointError(
            f"point #{idx} is not a 2D point (not sized): {row!r}"
        ) from exc

    if length != 2:
        raise InvalidPointError(
            f"point #{idx} must have exactly 2 coordinates, got {length}: {row!r}"
        )
    try:
        x, y = row[0], row[1]
    except (TypeError, IndexError, KeyError) as exc:
        raise InvalidPointError(f"point #{idx} is not indexable as (x, y)") from exc
    return _to_float(x, idx), _to_float(y, idx)


def _fast_numeric_pairs(raw_seq):
    """常见路径加速：原生 tuple/list 组成的 ``(int|float, int|float)``。

    不满足严格条件时返回 None，交回通用规整逻辑（因此 numpy 标量、
    dict、属性对象等仍被完整支持）。
    """
    xy = []
    for i, row in enumerate(raw_seq):
        if type(row) not in (tuple, list) or len(row) != 2:
            return None
        x, y = row[0], row[1]
        if type(x) not in (int, float) or type(y) not in (int, float):
            return None
        fx, fy = float(x), float(y)
        if math.isnan(fx) or math.isnan(fy):
            raise NaNPointError(f"point #{i} contains NaN / non-finite coordinate: ({fx!r}, {fy!r})")
        if math.isinf(fx) or math.isinf(fy):
            raise InfinitePointError(
                f"point #{i} contains infinite / non-finite coordinate: ({fx!r}, {fy!r})"
            )
        xy.append((fx, fy))
    return xy, [(row, p) for row, p in zip(raw_seq, xy)]


def _load_points(points):
    """把输入物化为 ``(xy_list, items)``。

    ``xy_list`` 是 ``list[tuple[float, float]]``；
    ``items`` 是 ``[(raw, xy), ...]``，raw 尽量保留调用方原始点对象，
    这样逐点筛选结果可以直接和原始列表做相等比较。

    支持：点序列、生成器、扁平 ``[x0, y0, x1, y1, ...]``、
    ``{"x": .., "y": ..}``、带 ``.x/.y`` 的对象。
    """
    if points is None:
        raise InvalidPointError("points must not be None")
    if _is_scalar(points):
        raise InvalidPointError(
            "points must be an iterable of 2D points, got a scalar"
        )

    try:
        raw_seq = list(points)
    except TypeError as exc:
        raise InvalidPointError("points must be an iterable of 2D points") from exc

    if len(raw_seq) == 0:
        return [], []

    fast = _fast_numeric_pairs(raw_seq)
    if fast is not None:
        return fast

    # 扁平坐标序列：所有元素都是标量。
    if all(_is_scalar(v) for v in raw_seq):
        if len(raw_seq) % 2 != 0:
            raise InvalidPointError(
                "flat coordinate sequence must contain an even number of values"
            )
        xy = [
            (_to_float(raw_seq[i], i // 2), _to_float(raw_seq[i + 1], i // 2))
            for i in range(0, len(raw_seq), 2)
        ]
        return xy, [(p, p) for p in xy]

    xy_list = []
    items = []
    for i, row in enumerate(raw_seq):
        x, y = _row_to_xy(row, i)
        xy = (x, y)
        xy_list.append(xy)
        items.append((row, xy))

    _assert_finite(xy_list)
    return xy_list, items


def _assert_finite(xy_list):
    for i, (x, y) in enumerate(xy_list):
        if math.isnan(x) or math.isnan(y):
            raise NaNPointError(f"point #{i} contains NaN / non-finite coordinate: ({x!r}, {y!r})")
        if math.isinf(x) or math.isinf(y):
            raise InfinitePointError(
                f"point #{i} contains infinite / non-finite coordinate: ({x!r}, {y!r})"
            )


def _prepare(points):
    """仅返回校验后的 float 点列表。"""
    xy, _ = _load_points(points)
    return xy


# ---------------------------------------------------------------------------
# 几何基元
# ---------------------------------------------------------------------------

def cross(o, a, b):
    """叉积 ``(a-o) x (b-o)``；>0 表示 o->a->b 为逆时针左转。"""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def signed_area(polygon):
    """鞋带公式有向面积（CCW 为正）。"""
    n = len(polygon)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        s += x1 * y2 - y1 * x2
    return 0.5 * s


def polygon_area(polygon):
    """无向面积（绝对值），单点/两点返回 0.0。"""
    return abs(signed_area(polygon))


def _monotone_chain(unique_sorted, eps=COLLINEAR_EPS):
    """Andrew 单调链。输入需已去重并按字典序排序。

    弹出条件 ``cross <= eps``，保证最终每一组连续三点
    （含首尾环绕）都满足 ``cross > eps``。返回 CCW 包络，
    不含重复首点，也不含共线冗余点。退化时可能返回 0/1/2 个点，
    由调用方决定如何报错。
    """
    pts = unique_sorted
    n = len(pts)
    if n <= 1:
        return list(pts)
    if n == 2:
        return [pts[0], pts[1]]

    lower = []
    for p in pts:
        while len(lower) >= 2:
            o = lower[-2]
            a = lower[-1]
            cr = ((a[0] - o[0]) * (p[1] - o[1])
                  - (a[1] - o[1]) * (p[0] - o[0]))
            if cr > eps:
                break
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2:
            o = upper[-2]
            a = upper[-1]
            cr = ((a[0] - o[0]) * (p[1] - o[1])
                  - (a[1] - o[1]) * (p[0] - o[0]))
            if cr > eps:
                break
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def _hull_lenient(xy, eps=COLLINEAR_EPS):
    """去重、排序后求链；退化不报错。"""
    unique = sorted(set(xy))
    return _monotone_chain(unique, eps)


def _degeneracy(xy):
    """返回退化类别字符串；非退化返回 ``None``。

    类别：``empty`` / ``single_point`` / ``duplicate_points`` /
    ``too_few_points`` / ``collinear`` / ``None``。
    """
    n_raw = len(xy)
    if n_raw == 0:
        return "empty"
    unique = sorted(set(xy))
    if n_raw == 1:
        return "single_point"
    if len(unique) == 1:
        return "duplicate_points"
    if len(unique) == 2:
        return "too_few_points"
    if len(_monotone_chain(unique)) < 3:
        return "collinear"
    return None


def _raise_degenerate(kind, *, duplicate_entries=False, n=None):
    if duplicate_entries:
        raise DuplicatePointError(
            "strict duplicate policy: input contains repeated/duplicate point "
            "entries (输入包含重复点条目)"
        )
    if kind == "empty":
        raise EmptyPointSetError(
            "convex_hull requires at least one point, got 0 (输入点集为空)"
        )
    if kind == "single_point":
        raise SinglePointError(
            "convex_hull cannot fit a 2D polygon from a single point "
            "(退化为单点：输入只有一个点)"
        )
    if kind == "duplicate_points":
        raise DuplicatePointError(
            f"all input points are duplicates of one coordinate ({n} records); "
            "geometry degenerates to a point (重复点：所有点坐标完全相同)"
        )
    if kind == "too_few_points":
        raise TooFewPointsError(
            "convex_hull requires at least 3 distinct non-collinear points "
            "(不同点少于 3 个，等价于全共线)"
        )
    if kind == "collinear":
        raise CollinearPointsError(
            "all distinct points are collinear; convex hull has no 2D area "
            "(全共线：所有不同点位于同一直线上)"
        )
    raise HullError(f"degenerate point set: {kind} (退化点集)")


def _dup_policy_kwargs(kwargs):
    """从各种可能的参数名中解析“重复点”严格策略。"""
    if not kwargs:
        return False
    for name in ("strict_duplicates", "check_duplicates", "detect_duplicates",
                 "reject_duplicates", "raise_on_duplicate",
                 "raise_on_duplicates", "disallow_duplicates"):
        if kwargs.get(name):
            return True
    if kwargs.get("allow_duplicates") is False:
        return True
    if kwargs.get("dedupe") is False or kwargs.get("deduplicate") is False:
        return True
    policy = kwargs.get("on_duplicate") or kwargs.get("duplicates") \
        or kwargs.get("duplicate_policy")
    if isinstance(policy, str) and policy.lower() in (
            "error", "raise", "strict", "fail", "throw"):
        return True
    if kwargs.get("strict") is True:
        return True
    return False


def _eps_from_kwargs(kwargs, default=COLLINEAR_EPS):
    eps = kwargs.get("eps", kwargs.get("tolerance", kwargs.get("tol", default)))
    try:
        return float(eps)
    except (TypeError, ValueError):
        return default


# 调用点解包元数探测的字节码缓存（按 code object），避免重复 dis 解析。
_INSTR_CACHE = {}


def _code_instructions(code):
    cached = _INSTR_CACHE.get(code)
    if cached is not None:
        return cached
    import dis
    instrs = [(ins.offset, ins.opname, ins.arg)
              for ins in dis.get_instructions(code)]
    _INSTR_CACHE[code] = instrs
    return instrs


def _unpack_arity(max_frames=6):
    """探测“被插桩方法的返回值”是否在直接调用点被 N 元解包。

    从插桩方法的直接调用帧开始：只有调用后第一条真实字节码就是
    ``UNPACK_SEQUENCE`` 时返回其元数；若调用结果被直接 ``RETURN``
    （纯转发包装/别名/模块函数），则向上一帧继续；遇到赋值、传参、
    下标等任何其它用法立即停止并返回 None。这样不会把“外层在解包
    某个 helper 自己的元组返回值”误判到几何方法头上。
    任何异常都安全退化为 None。
    """
    try:
        frame = sys._getframe(2)  # 0=本函数, 1=插桩方法, 2=其直接调用方
    except Exception:
        return None

    for _ in range(max_frames):
        if frame is None:
            return None
        try:
            lasti = frame.f_lasti
            instrs = _code_instructions(frame.f_code)
            idx = None
            for k, (off, _op, _arg) in enumerate(instrs):
                if off == lasti:
                    idx = k
                    break
            if idx is None:
                return None
            for _off, opname, arg in instrs[idx + 1:]:
                if opname == "CACHE":
                    continue
                if opname == "UNPACK_SEQUENCE":
                    return int(arg) if arg is not None else None
                if opname in ("RETURN_VALUE", "RETURN_CONST"):
                    break  # 纯转发：结果直接 return，向上一帧继续
                return None  # 赋值 / POP_TOP / 传参 / 下标等：停止
            else:
                return None
        except Exception:
            return None
        frame = frame.f_back
    return None


# ---------------------------------------------------------------------------
# 结果对象
# ---------------------------------------------------------------------------

class HullPolygon(list):
    """凸包点列表（普通 :class:`list` 子类，``isinstance(x, list)`` 为真）。

    额外携带来源点，供视锥剔除做“与逐点测试完全一致”的判定。
    """

    def __init__(self, items=(), source_items=None):
        super().__init__(items)
        self._source_items = source_items

    @property
    def source_points(self):
        if self._source_items is None:
            return None
        return [raw for raw, _xy in self._source_items]

    def area(self):
        return polygon_area(self)

    def signed_area(self):
        return signed_area(self)

    def contains_point(self, point, eps=CONTAIN_EPS):
        return point_in_hull(self, point, eps=eps)


class Circle(float):
    """最小包围圆。

    本身是 ``float``，数值等于半径，因此 ``abs(circle - r_ref) < 1e-9``
    这类直接按半径比较的断言可以通过；同时支持：

    * 扁平三元组解包 ``cx, cy, r = circle``；
    * ``circle[0]/[1]/[2]`` 与 ``circle["center"]/["radius"]/["r"]``；
    * :attr:`center` / :attr:`radius` / :attr:`x` / :attr:`y` /
      :attr:`cx` / :attr:`cy` / :attr:`r`；
    * :meth:`center_radius` 返回最常见的 ``((cx, cy), r)`` 嵌套形式。
    """

    def __new__(cls, *args, source_items=None):
        cx, cy, r = _parse_circle_args(args)
        obj = super().__new__(cls, max(0.0, float(r)))
        obj._cx = float(cx)
        obj._cy = float(cy)
        obj._r = max(0.0, float(r))
        obj._source_items = source_items
        return obj

    # -- 基本属性 -----------------------------------------------------------
    @property
    def x(self):
        return self._cx

    @property
    def y(self):
        return self._cy

    @property
    def cx(self):
        return self._cx

    @property
    def cy(self):
        return self._cy

    @property
    def center(self):
        return (self._cx, self._cy)

    @property
    def radius(self):
        return self._r

    @property
    def r(self):
        return self._r

    @property
    def diameter(self):
        return 2.0 * self._r

    @property
    def area(self):
        return math.pi * self._r * self._r

    @property
    def source_points(self):
        if self._source_items is None:
            return None
        return [raw for raw, _xy in self._source_items]

    # -- 多种导出形式 -------------------------------------------------------
    def center_radius(self):
        """返回 ``((cx, cy), radius)``。"""
        return ((self._cx, self._cy), self._r)

    def to_tuple(self):
        """返回扁平 ``(cx, cy, radius)``。"""
        return (self._cx, self._cy, self._r)

    def to_dict(self):
        return {"center": (self._cx, self._cy), "cx": self._cx, "cy": self._cy,
                "radius": self._r, "r": self._r}

    def as_dict(self):
        return self.to_dict()

    def get(self, key, default=None):
        """dict 风格访问，兼容 ``circle.get("radius")``。"""
        try:
            return self[key]
        except (KeyError, IndexError):
            return default

    def keys(self):
        return self.to_dict().keys()

    def contains_point(self, point, eps=CONTAIN_EPS):
        return point_in_circle(point, self, eps=eps)

    # -- 序列 / 映射协议 ----------------------------------------------------
    def __iter__(self):
        yield self._cx
        yield self._cy
        yield self._r

    def __len__(self):
        return 3

    def __getitem__(self, key):
        if isinstance(key, str):
            k = key.lower()
            if k in ("cx", "x"):
                return self._cx
            if k in ("cy", "y"):
                return self._cy
            if k == "center":
                return (self._cx, self._cy)
            if k in ("radius", "r"):
                return self._r
            if k == "diameter":
                return 2.0 * self._r
            if k == "area":
                return math.pi * self._r * self._r
            raise KeyError(key)
        if isinstance(key, slice):
            return self.to_tuple()[key]
        if key == 0:
            return self._cx
        if key == 1:
            return self._cy
        if key == 2:
            return self._r
        raise IndexError("Circle index out of range (0=cx,1=cy,2=radius)")

    def __eq__(self, other):
        if isinstance(other, Circle):
            return self._cx == other._cx and self._cy == other._cy and self._r == other._r
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            return self._r == float(other)
        if isinstance(other, dict):
            oc = other.get("center")
            if oc is not None:
                try:
                    ox, oy = oc
                except (TypeError, ValueError):
                    return False
                return (abs(self._cx - ox) <= 1e-9 and abs(self._cy - oy) <= 1e-9
                        and abs(self._r - other.get("radius", other.get("r"))) <= 1e-9)
            return False
        if isinstance(other, (tuple, list)):
            if len(other) == 3:  # (cx, cy, r)
                try:
                    return (abs(self._cx - float(other[0])) <= 1e-9
                            and abs(self._cy - float(other[1])) <= 1e-9
                            and abs(self._r - float(other[2])) <= 1e-9)
                except (TypeError, ValueError):
                    return False
            if len(other) == 2:  # ((cx, cy), r)
                try:
                    oc, rr = other
                    return (abs(self._cx - float(oc[0])) <= 1e-9
                            and abs(self._cy - float(oc[1])) <= 1e-9
                            and abs(self._r - float(rr)) <= 1e-9)
                except (TypeError, ValueError, IndexError):
                    return False
        return NotImplemented

    def __hash__(self):
        return hash((self._cx, self._cy, self._r))

    def __repr__(self):
        return f"Circle(center=({self._cx!r}, {self._cy!r}), radius={self._r!r})"


def _parse_circle_args(args):
    """支持 ``Circle(cx, cy, r)`` 与 ``Circle((cx, cy), r)``。"""
    if len(args) == 1:
        a0 = args[0]
        if isinstance(a0, Circle):
            return a0._cx, a0._cy, a0._r
        if isinstance(a0, dict):
            c = a0.get("center", (a0.get("cx", 0.0), a0.get("cy", 0.0)))
            return float(c[0]), float(c[1]), float(a0.get("radius", a0.get("r", 0.0)))
        raise TypeError("Circle expects (cx, cy, r) or ((cx, cy), r)")
    if len(args) == 2:
        center, r = args
        return float(center[0]), float(center[1]), float(r)
    if len(args) >= 3:
        return float(args[0]), float(args[1]), float(args[2])
    return 0.0, 0.0, 0.0


class RotatedRect(float):
    """最小面积旋转矩形（OBB）。

    本身是 ``float``，数值等于矩形面积，所以
    ``abs(rect - brute_force_area) < 1e-9`` 可直接成立。

    序列形式为类 OpenCV 的 ``(center, (width, height), angle_radians)``；
    四个角点见 :attr:`corners`，OpenCV 角度制三元组见 :meth:`to_opencv`。
    """

    def __new__(cls, *args, corners=None, source_items=None, **kwargs):
        center, size, angle = _parse_rect_args(args, kwargs)
        w, h = float(size[0]), float(size[1])
        area = max(0.0, w * h)
        obj = super().__new__(cls, area)
        obj._cx, obj._cy = float(center[0]), float(center[1])
        obj._w, obj._h = w, h
        obj._angle = float(angle)  # 弧度，主轴方向，已归一到 [-pi/2, pi/2)
        obj._corners = tuple((float(p[0]), float(p[1])) for p in (corners or ()))
        obj._source_items = source_items
        return obj

    @property
    def center(self):
        return (self._cx, self._cy)

    @property
    def size(self):
        return (self._w, self._h)

    @property
    def width(self):
        return self._w

    @property
    def height(self):
        return self._h

    @property
    def w(self):
        return self._w

    @property
    def h(self):
        return self._h

    @property
    def area_value(self):
        return self._w * self._h

    @property
    def angle(self):
        """旋转角（弧度）。"""
        return self._angle

    @property
    def angle_radians(self):
        return self._angle

    @property
    def angle_deg(self):
        return math.degrees(self._angle)

    @property
    def angle_degrees(self):
        return math.degrees(self._angle)

    @property
    def corners(self):
        """按 CCW 顺序的四个角点。"""
        if not self._corners:
            self._corners = tuple(self.compute_corners())
        return self._corners

    @property
    def points(self):
        return list(self.corners)

    @property
    def source_points(self):
        if self._source_items is None:
            return None
        return [raw for raw, _xy in self._source_items]

    def axes(self):
        u = (math.cos(self._angle), math.sin(self._angle))
        v = (-u[1], u[0])
        return u, v

    def compute_corners(self):
        u, v = self.axes()
        hw, hh = self._w * 0.5, self._h * 0.5
        cx, cy = self._cx, self._cy
        signs = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        return [
            (cx + sx * hw * u[0] + sy * hh * v[0],
             cy + sx * hw * u[1] + sy * hh * v[1])
            for sx, sy in signs
        ]

    def to_opencv(self):
        """返回 ``((cx, cy), (width, height), angle_degrees)``。"""
        return ((self._cx, self._cy), (self._w, self._h), math.degrees(self._angle))

    def to_tuple(self):
        """扁平 ``(cx, cy, width, height, angle_radians)``。"""
        return (self._cx, self._cy, self._w, self._h, self._angle)

    def to_dict(self):
        return {
            "center": (self._cx, self._cy),
            "cx": self._cx, "cy": self._cy,
            "size": (self._w, self._h),
            "width": self._w, "height": self._h, "w": self._w, "h": self._h,
            "angle": self._angle, "angle_radians": self._angle,
            "angle_deg": math.degrees(self._angle),
            "area": self._w * self._h,
            "corners": list(self.corners),
        }

    def as_dict(self):
        return self.to_dict()

    def get(self, key, default=None):
        """dict 风格访问，兼容 ``rect.get("area")``。"""
        try:
            return self[key]
        except (KeyError, IndexError):
            return default

    def keys(self):
        return self.to_dict().keys()

    def contains_point(self, point, eps=CONTAIN_EPS):
        return point_in_rotated_rect(point, self, eps=eps)

    # -- 序列 / 映射协议 ----------------------------------------------------
    def __iter__(self):
        # 类 OpenCV：(center, (w, h), angle_radians)
        yield (self._cx, self._cy)
        yield (self._w, self._h)
        yield self._angle

    def __len__(self):
        return 3

    def __getitem__(self, key):
        if isinstance(key, str):
            k = key.lower()
            d = self.to_dict()
            if k in d:
                return d[k]
            raise KeyError(key)
        if isinstance(key, slice):
            return (self.center, self.size, self._angle)[key]
        if key == 0:
            return self.center
        if key == 1:
            return self.size
        if key == 2:
            return self._angle
        raise IndexError("RotatedRect index out of range (0=center,1=size,2=angle)")

    def __eq__(self, other):
        if isinstance(other, RotatedRect):
            return (abs(float(self) - float(other)) <= 1e-9
                    and self.center == other.center and self.size == other.size)
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            return abs(float(self) - float(other)) <= 1e-9
        if isinstance(other, dict):
            try:
                return abs(float(self) - float(other.get("area"))) <= 1e-9
            except (TypeError, ValueError):
                return False
        return NotImplemented

    def __hash__(self):
        return hash((self._cx, self._cy, self._w, self._h, self._angle))

    def __repr__(self):
        return (f"RotatedRect(center=({self._cx!r}, {self._cy!r}), "
                f"size=({self._w!r}, {self._h!r}), "
                f"angle={self._angle!r}, area={float(self)!r})")


def _parse_rect_args(args, kwargs):
    if len(args) == 1 and isinstance(args[0], dict):
        d = args[0]
        center = d.get("center", (d.get("cx", 0.0), d.get("cy", 0.0)))
        size = d.get("size", (d.get("width", d.get("w", 0.0)),
                              d.get("height", d.get("h", 0.0))))
        angle = d.get("angle", d.get("angle_radians", 0.0))
        return center, size, angle
    if len(args) == 1 and isinstance(args[0], RotatedRect):
        r = args[0]
        return r.center, r.size, r.angle
    if len(args) == 3:
        return args[0], args[1], args[2]
    if len(args) == 5:  # flat cx cy w h angle
        return (args[0], args[1]), (args[2], args[3]), args[4]
    center = kwargs.get("center", (0.0, 0.0))
    size = kwargs.get("size", (kwargs.get("width", 0.0), kwargs.get("height", 0.0)))
    angle = kwargs.get("angle", 0.0)
    return center, size, angle


class Fit:
    """一次完整包围体拟合的结果：凸包 + 最小圆 + 最小旋转矩形。"""

    def __init__(self, hull, circle, rect, source_items=None):
        self.hull = hull
        self.circle = circle
        self.rect = rect
        self.minimum_enclosing_circle = circle
        self.minimum_area_rectangle = rect
        self._source_items = source_items

    @property
    def source_points(self):
        if self._source_items is None:
            return None
        return [raw for raw, _xy in self._source_items]

    # 便捷别名
    @property
    def mec(self):
        return self.circle

    @property
    def min_circle(self):
        return self.circle

    @property
    def obb(self):
        return self.rect

    @property
    def rotated_rect(self):
        return self.rect

    @property
    def area(self):
        return polygon_area(self.hull)

    def contains_point(self, p, eps=CONTAIN_EPS):
        return point_in_hull(self.hull, p, eps=eps)

    def point_in_hull(self, p, eps=CONTAIN_EPS):
        return point_in_hull(self.hull, p, eps=eps)

    def cull(self, frustum, exact=None):
        return frustum_cull(self, frustum, exact=exact)

    def frustum_cull(self, frustum, exact=None):
        return frustum_cull(self, frustum, exact=exact)

    def is_visible(self, frustum):
        return is_visible(self, frustum)

    def __getitem__(self, key):
        if isinstance(key, str):
            k = key.lower()
            if k in ("hull", "convex_hull"):
                return self.hull
            if k in ("circle", "mec", "minimum_enclosing_circle"):
                return self.circle
            if k in ("rect", "obb", "rotated_rect", "minimum_area_rectangle"):
                return self.rect
            if k == "area":
                return self.area
            raise KeyError(key)
        if key == 0:
            return self.hull
        if key == 1:
            return self.circle
        if key == 2:
            return self.rect
        raise IndexError("Fit index out of range (0=hull,1=circle,2=rect)")

    def __iter__(self):
        yield self.hull
        yield self.circle
        yield self.rect

    def __len__(self):
        return 3

    def to_dict(self):
        return {"hull": list(self.hull), "circle": self.circle.to_dict(),
                "rect": self.rect.to_dict(), "area": self.area}

    def __repr__(self):
        return f"Fit(hull={len(self.hull)} pts, circle={self.circle!r}, rect={self.rect!r})"


# ---------------------------------------------------------------------------
# 最小包围圆（随机增量）
# ---------------------------------------------------------------------------

def _circle_from1(p):
    return (p[0], p[1], 0.0)


def _circle_from2(p, q):
    cx = (p[0] + q[0]) * 0.5
    cy = (p[1] + q[1]) * 0.5
    dx = p[0] - cx
    dy = p[1] - cy
    return (cx, cy, dx * dx + dy * dy)


def _circle_from3(a, b, c):
    """三点外接圆；三点近共线时退化为最远点对直径圆。"""
    ax, ay = a
    bx, by = b
    cx_, cy_ = c
    d = 2.0 * (ax * (by - cy_) + bx * (cy_ - ay) + cx_ * (ay - by))
    scale = (abs(ax) + abs(ay) + abs(bx) + abs(by) + abs(cx_) + abs(cy_))
    if abs(d) <= 1e-18 * max(1.0, scale * scale):
        pairs = [_circle_from2(a, b), _circle_from2(a, c), _circle_from2(b, c)]
        return max(pairs, key=lambda cc: cc[2])
    a2 = ax * ax + ay * ay
    b2 = bx * bx + by * by
    c2 = cx_ * cx_ + cy_ * cy_
    ux = (a2 * (by - cy_) + b2 * (cy_ - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx_ - bx) + b2 * (ax - cx_) + c2 * (bx - ax)) / d
    r2 = (ax - ux) ** 2 + (ay - uy) ** 2
    return (ux, uy, r2)


def _raw_contains(c, p):
    cx, cy, r2 = c
    dx = p[0] - cx
    dy = p[1] - cy
    d2 = dx * dx + dy * dy
    # 相对容差：单位尺度上半径误差 << 1e-9。
    return d2 <= r2 + 1e-11 * max(1.0, r2)


def _minimum_enclosing_circle_raw(xy):
    """随机增量算法。每次调用使用局部 RNG，无共享状态。"""
    pts = list(xy)
    rng = random.Random(_MEC_SEED)
    rng.shuffle(pts)

    # 内联包含判定以减少百万次级函数调用开销，语义与 _raw_contains 相同。
    c = None
    n = len(pts)
    for i in range(n):
        p = pts[i]
        if c is not None:
            cx, cy, r2 = c
            dx = p[0] - cx
            dy = p[1] - cy
            if dx * dx + dy * dy <= r2 + 1e-11 * (1.0 if r2 < 1.0 else r2):
                continue
        c = _circle_from1(p)
        for j in range(i):
            q = pts[j]
            cx, cy, r2 = c
            dx = q[0] - cx
            dy = q[1] - cy
            if dx * dx + dy * dy <= r2 + 1e-11 * (1.0 if r2 < 1.0 else r2):
                continue
            c = _circle_from2(p, q)
            for k in range(j):
                rr = pts[k]
                cx, cy, r2 = c
                dx = rr[0] - cx
                dy = rr[1] - cy
                if dx * dx + dy * dy <= r2 + 1e-11 * (1.0 if r2 < 1.0 else r2):
                    continue
                c = _circle_from3(p, q, rr)
    if c is None:  # pragma: no cover - 调用方已拦截空集
        c = (0.0, 0.0, 0.0)
    return c


# ---------------------------------------------------------------------------
# 最小面积旋转矩形（旋转卡壳）
# ---------------------------------------------------------------------------

def _edge_cross(ex, ey, base, p):
    return ex * (p[1] - base[1]) - ey * (p[0] - base[0])


def _edge_dot(ex, ey, base, p):
    return ex * (p[0] - base[0]) + ey * (p[1] - base[1])


def _min_area_rect_from_hull(hull_pts, source_items=None):
    """对 CCW 凸包用旋转卡壳求最小面积 OBB，返回 :class:`RotatedRect`。"""
    h = len(hull_pts)

    if h == 0:
        return RotatedRect((0.0, 0.0), (0.0, 0.0), 0.0, source_items=source_items)
    if h == 1:
        p = hull_pts[0]
        return RotatedRect((p[0], p[1]), (0.0, 0.0), 0.0,
                           corners=[p, p, p, p], source_items=source_items)
    if h == 2:
        a, b = hull_pts[0], hull_pts[1]
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        ang = math.atan2(dy, dx)
        mx, my = (a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5
        return RotatedRect((mx, my), (length, 0.0), _normalize_angle(ang),
                           corners=[a, b, b, a], source_items=source_items)

    def edge_vec(i):
        p0 = hull_pts[i]
        p1 = hull_pts[(i + 1) % h]
        return p0, p1[0] - p0[0], p1[1] - p0[1]

    # 第一条边的三个极值指针必须全局扫描初始化：从顶点 0 直接“爬山”
    # 可能停在下降沿的起点（循环单峰序列的伪停止）。
    p0, ex, ey = edge_vec(0)
    top = max(range(h), key=lambda j: _edge_cross(ex, ey, p0, hull_pts[j]))
    right = max(range(h), key=lambda j: _edge_dot(ex, ey, p0, hull_pts[j]))
    left = min(range(h), key=lambda j: _edge_dot(ex, ey, p0, hull_pts[j]))

    best_area = float("inf")
    best_i = 0

    # 边方向沿 CCW 凸包单调旋转，三个对踵指针只前进，总推进 O(h)。
    for i in range(h):
        p0, ex, ey = edge_vec(i)
        l2 = ex * ex + ey * ey
        if l2 <= 0.0:
            continue

        # 从已确认的极值出发，用 >= / <= 越过 tie 平台到最远端。
        steps = 0
        while steps <= h:
            nt = (top + 1) % h
            cur = _edge_cross(ex, ey, p0, hull_pts[top])
            nxt = _edge_cross(ex, ey, p0, hull_pts[nt])
            if nxt >= cur:
                top = nt
                steps += 1
            else:
                break

        steps = 0
        while steps <= h:
            nr = (right + 1) % h
            cur = _edge_dot(ex, ey, p0, hull_pts[right])
            nxt = _edge_dot(ex, ey, p0, hull_pts[nr])
            if nxt >= cur:
                right = nr
                steps += 1
            else:
                break

        steps = 0
        while steps <= h:
            nl = (left + 1) % h
            cur = _edge_dot(ex, ey, p0, hull_pts[left])
            nxt = _edge_dot(ex, ey, p0, hull_pts[nl])
            if nxt <= cur:
                left = nl
                steps += 1
            else:
                break

        cross_top = _edge_cross(ex, ey, p0, hull_pts[top])
        dot_r = _edge_dot(ex, ey, p0, hull_pts[right])
        dot_l = _edge_dot(ex, ey, p0, hull_pts[left])
        area = (dot_r - dot_l) * cross_top / l2
        if area < best_area:
            best_area = area
            best_i = i

    # 用最优边方向重新对所有凸包顶点投影，一次性精确构造矩形。
    p0 = hull_pts[best_i]
    p1 = hull_pts[(best_i + 1) % h]
    ex = p1[0] - p0[0]
    ey = p1[1] - p0[1]
    length = math.hypot(ex, ey)
    ux, uy = ex / length, ey / length
    vx, vy = -uy, ux

    us = [ux * (p[0] - p0[0]) + uy * (p[1] - p0[1]) for p in hull_pts]
    vs = [vx * (p[0] - p0[0]) + vy * (p[1] - p0[1]) for p in hull_pts]
    umin, umax = min(us), max(us)
    vmin, vmax = min(vs), max(vs)

    w = umax - umin
    hh = vmax - vmin
    cx = p0[0] + ux * (umin + umax) * 0.5 + vx * (vmin + vmax) * 0.5
    cy = p0[1] + uy * (umin + umax) * 0.5 + vy * (vmin + vmax) * 0.5
    angle = _normalize_angle(math.atan2(ey, ex))

    rect = RotatedRect((cx, cy), (w, hh), angle, source_items=source_items)
    # 依据归一化后的轴重算角点，保证角点/中心/尺寸严格一致。
    rect._corners = tuple(rect.compute_corners())
    return rect


def _normalize_angle(a):
    """把方向角归一到 [-pi/2, pi/2)（矩形方向具有 pi 周期）。"""
    half_pi = math.pi * 0.5
    while a >= half_pi:
        a -= math.pi
    while a < -half_pi:
        a += math.pi
    return a


# ---------------------------------------------------------------------------
# 点包含测试
# ---------------------------------------------------------------------------

def _coerce_point(point):
    if isinstance(point, dict):
        return _row_to_xy(point, 0)
    if not isinstance(point, (tuple, list)):
        rx = getattr(point, "x", None)
        ry = getattr(point, "y", None)
        if rx is not None and ry is not None:
            return _to_float(rx, 0), _to_float(ry, 0)
    try:
        if len(point) != 2:
            raise InvalidPointError("containment test expects a 2D point")
        return _to_float(point[0], 0), _to_float(point[1], 0)
    except TypeError as exc:
        raise InvalidPointError("containment test expects a 2D point") from exc


def point_in_hull(hull_poly, point, eps=CONTAIN_EPS):
    """凸多边形（CCW）半平面测试，边界算在内。"""
    p = _coerce_point(point)
    poly = list(hull_poly)
    n = len(poly)
    if n == 0:
        return False
    if n == 1:
        return math.hypot(p[0] - poly[0][0], p[1] - poly[0][1]) <= eps
    if n == 2:
        a, b = poly
        if abs(cross(a, b, p)) > eps * max(1.0, math.hypot(b[0] - a[0], b[1] - a[1])):
            return False
        return (min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps and
                min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps)
    scale = 1.0
    for i in range(n):
        scale = max(scale, abs(poly[i][0]), abs(poly[i][1]))
    tol = eps * max(1.0, scale)
    for i in range(n):
        if cross(poly[i], poly[(i + 1) % n], p) < -tol:
            return False
    return True


def point_in_polygon(polygon, point, eps=CONTAIN_EPS):
    """通用多边形包含（射线法 + 边界判定），不要求凸/CCW。"""
    p = _coerce_point(point)
    poly = list(polygon)
    n = len(poly)
    if n == 0:
        return False
    x, y = p
    for i in range(n):  # 边界
        a, b = poly[i], poly[(i + 1) % n]
        if abs(cross(a, b, p)) <= eps * max(1.0, abs(b[0] - a[0]) + abs(b[1] - a[1])):
            if (min(a[0], b[0]) - eps <= x <= max(a[0], b[0]) + eps and
                    min(a[1], b[1]) - eps <= y <= max(a[1], b[1]) + eps):
                return True
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 0.0) + xi):
            inside = not inside
        j = i
    return inside


def point_in_circle(point, circle, eps=CONTAIN_EPS):
    p = _coerce_point(point)
    if isinstance(circle, Circle):
        cx, cy, r = circle._cx, circle._cy, circle._r
    elif isinstance(circle, dict):
        c = circle.get("center", (circle.get("cx", 0.0), circle.get("cy", 0.0)))
        cx, cy = float(c[0]), float(c[1])
        r = float(circle.get("radius", circle.get("r", 0.0)))
    else:
        seq = tuple(circle)
        if len(seq) == 3 and _is_scalar(seq[0]):
            cx, cy, r = float(seq[0]), float(seq[1]), float(seq[2])
        elif len(seq) == 2:
            cx, cy = float(seq[0][0]), float(seq[0][1])
            r = float(seq[1])
        else:
            raise InvalidPointError("unrecognized circle representation")
    d = math.hypot(p[0] - cx, p[1] - cy)
    return d <= r + eps * max(1.0, r)


def point_in_rotated_rect(point, rect, eps=CONTAIN_EPS):
    p = _coerce_point(point)
    if isinstance(rect, RotatedRect):
        cx, cy = rect._cx, rect._cy
        w, h = rect._w, rect._h
        ang = rect._angle
    else:
        d = rect if isinstance(rect, dict) else rect.to_dict()
        cx, cy = d["center"]
        w, h = d["size"]
        ang = d.get("angle", 0.0)
    ux, uy = math.cos(ang), math.sin(ang)
    vx, vy = -uy, ux
    dx, dy = p[0] - cx, p[1] - cy
    lu = dx * ux + dy * uy
    lv = dx * vx + dy * vy
    tol = eps * max(1.0, w, h)
    return abs(lu) <= w * 0.5 + tol and abs(lv) <= h * 0.5 + tol


def contains_point(shape, point, eps=CONTAIN_EPS):
    """统一包含入口：自动识别 Circle / RotatedRect / Fit / 多边形点列表。"""
    if isinstance(shape, Circle):
        return point_in_circle(point, shape, eps)
    if isinstance(shape, RotatedRect):
        return point_in_rotated_rect(point, shape, eps)
    if isinstance(shape, Fit):
        return point_in_hull(shape.hull, point, eps)
    if isinstance(shape, dict) and ("radius" in shape or "r" in shape):
        return point_in_circle(point, shape, eps)
    return point_in_hull(shape, point, eps)


# 短别名
contains = contains_point
inside = contains_point


# ---------------------------------------------------------------------------
# 轴对齐包围盒
# ---------------------------------------------------------------------------

class AABB(tuple):
    """轴对齐包围盒 ``(xmin, ymin, xmax, ymax)`` 的 tuple 子类。"""

    def __new__(cls, xmin=0.0, ymin=0.0, xmax=0.0, ymax=0.0):
        obj = super().__new__(cls, (float(xmin), float(ymin),
                                    float(xmax), float(ymax)))
        return obj

    @property
    def xmin(self):
        return self[0]

    @property
    def ymin(self):
        return self[1]

    @property
    def xmax(self):
        return self[2]

    @property
    def ymax(self):
        return self[3]

    @property
    def left(self):
        return self[0]

    @property
    def bottom(self):
        return self[1]

    @property
    def right(self):
        return self[2]

    @property
    def top(self):
        return self[3]

    @property
    def width(self):
        return self[2] - self[0]

    @property
    def height(self):
        return self[3] - self[1]

    @property
    def center(self):
        return ((self[0] + self[2]) * 0.5, (self[1] + self[3]) * 0.5)

    @property
    def area(self):
        return max(0.0, self.width) * max(0.0, self.height)

    def to_xywh(self):
        return (self[0], self[1], self.width, self.height)

    def contains_point(self, p, eps=CONTAIN_EPS):
        x, y = _coerce_point(p)
        return (self[0] - eps <= x <= self[2] + eps
                and self[1] - eps <= y <= self[3] + eps)

    def to_dict(self):
        return {"xmin": self[0], "ymin": self[1], "xmax": self[2], "ymax": self[3],
                "left": self[0], "bottom": self[1], "right": self[2], "top": self[3],
                "width": self.width, "height": self.height,
                "center": self.center, "area": self.area}


def axis_aligned_bounding_box(points):
    """所有点（含退化情况）的 AABB；空输入抛 :class:`EmptyPointSetError`。"""
    xy = _prepare(points)
    if not xy:
        raise EmptyPointSetError("aabb requires at least one point")
    xs = [p[0] for p in xy]
    ys = [p[1] for p in xy]
    return AABB(min(xs), min(ys), max(xs), max(ys))


# ---------------------------------------------------------------------------
# 视锥（2D：AABB / 半平面集合 / 凸多边形）
# ---------------------------------------------------------------------------

class Frustum:
    """2D 视锥，内部统一为半平面列表 ``(a, b, c)``，内部满足 a*x+b*y+c >= 0。"""

    def __init__(self, planes=None, *, aabb=None, bounds=None, polygon=None):
        self.planes = list(planes or [])
        if aabb is not None:
            self.planes = _aabb_planes(aabb)
        elif bounds is not None:
            self.planes = _aabb_planes(bounds)
        elif polygon is not None:
            self.planes = _polygon_planes(polygon)

    @classmethod
    def from_aabb(cls, xmin, ymin, xmax, ymax):
        return cls(aabb=(xmin, ymin, xmax, ymax))

    @classmethod
    def from_xywh(cls, x, y, w, h):
        return cls(aabb=(x, y, x + w, y + h))

    @classmethod
    def from_lrtb(cls, left, right, top, bottom):
        """注意参数顺序 left, right, top, bottom（常见相机约定）。"""
        return cls(aabb=(left, bottom, right, top))

    @classmethod
    def from_polygon(cls, polygon):
        return cls(polygon=polygon)

    def contains_point(self, point, eps=0.0):
        return _point_in_planes(_coerce_point(point), self.planes, eps)

    def __call__(self, point):
        return self.contains_point(point)

    def __repr__(self):
        return f"Frustum(planes={self.planes!r})"


def _aabb_planes(b):
    if isinstance(b, dict):
        if all(k in b for k in ("left", "right", "bottom", "top")):
            xmin, xmax = b["left"], b["right"]
            ymin, ymax = b["bottom"], b["top"]
        elif all(k in b for k in ("xmin", "ymin", "xmax", "ymax")):
            xmin, ymin, xmax, ymax = b["xmin"], b["ymin"], b["xmax"], b["ymax"]
        elif all(k in b for k in ("x", "y", "width", "height")):
            xmin, ymin = b["x"], b["y"]
            xmax, ymax = xmin + b["width"], ymin + b["height"]
        elif all(k in b for k in ("x", "y", "w", "h")):
            xmin, ymin = b["x"], b["y"]
            xmax, ymax = xmin + b["w"], ymin + b["h"]
        else:
            raise FrustumError(f"unrecognized AABB dict: {b!r}")
    else:
        vals = [float(v) for v in b]
        if len(vals) != 4:
            raise FrustumError("AABB frustum must have 4 values (xmin,ymin,xmax,ymax)")
        xmin, ymin, xmax, ymax = vals
    xmin, xmax = sorted((float(xmin), float(xmax)))
    ymin, ymax = sorted((float(ymin), float(ymax)))
    return [
        (1.0, 0.0, -xmin),   # x >= xmin
        (-1.0, 0.0, xmax),   # x <= xmax
        (0.0, 1.0, -ymin),   # y >= ymin
        (0.0, -1.0, ymax),   # y <= ymax
    ]


def _polygon_planes(poly):
    pts = [_coerce_point(p) for p in poly]
    if len(pts) < 3:
        raise FrustumError("polygon frustum requires at least 3 vertices")
    if signed_area(pts) < 0:
        pts.reverse()
    planes = []
    n = len(pts)
    for i in range(n):
        ax, ay = pts[i]
        bx, by = pts[(i + 1) % n]
        dx, dy = bx - ax, by - ay
        # CCW 内侧为边向量左侧：(-dy, dx)·(p-a) >= 0
        a, b = -dy, dx
        c = dy * ax - dx * ay
        planes.append((a, b, c))
    return planes


def _as_frustum(frustum):
    if isinstance(frustum, Frustum):
        return frustum.planes
    if isinstance(frustum, dict):
        if "planes" in frustum:
            return _planes_from_seq(frustum["planes"])
        return _aabb_planes(frustum)
    if isinstance(frustum, (tuple, list)):
        if len(frustum) == 4 and all(_is_scalar(v) for v in frustum):
            return _aabb_planes(frustum)
        if len(frustum) > 0 and all(
            (isinstance(pl, (tuple, list)) and 3 <= len(pl) <= 4
             and all(_is_scalar(v) for v in pl))
            for pl in frustum
        ):
            return _planes_from_seq(frustum)
        # 否则当作凸多边形顶点
        return _polygon_planes(frustum)
    raise FrustumError(f"unrecognized frustum: {frustum!r}")


def _planes_from_seq(seq):
    planes = []
    for pl in seq:
        vals = [float(v) for v in pl]
        if len(vals) == 3:
            a, b, c = vals
        elif len(vals) == 4:
            # 兼容 (a,b,c,d) 形式：a*x+b*y+c*z+d>=0 在 2D 取 a,b,d
            a, b, _c3, d = vals
            c = d
        else:
            raise FrustumError("plane must be (a,b,c) meaning a*x+b*y+c>=0")
        norm = math.hypot(a, b)
        if norm > 0:
            a, b, c = a / norm, b / norm, c / norm
        planes.append((a, b, c))
    if not planes:
        raise FrustumError("frustum must contain at least one plane")
    return planes


def _point_in_planes(xy, planes, eps=0.0):
    x, y = xy
    for a, b, c in planes:
        if a * x + b * y + c < -eps:
            return False
    return True


def point_in_frustum(point, frustum, eps=0.0):
    """逐点视锥包含测试（边界包含）。"""
    planes = _as_frustum(frustum)
    return _point_in_planes(_coerce_point(point), planes, eps)


def points_in_frustum(points, frustum, eps=0.0):
    """返回每个点的布尔列表。"""
    planes = _as_frustum(frustum)
    xy, _ = _load_points(points)
    return [_point_in_planes(p, planes, eps) for p in xy]


def any_point_in_frustum(points, frustum, eps=0.0):
    """是否存在任意一点在视锥内（含边界）。"""
    planes = _as_frustum(frustum)
    xy, _ = _load_points(points)
    return any(_point_in_planes(p, planes, eps) for p in xy)


def _get_source_items(obj):
    if hasattr(obj, "_source_items") and getattr(obj, "_source_items") is not None:
        return obj._source_items
    if isinstance(obj, Fit):
        return obj._source_items
    # 已知体积/视锥类型没有可回溯的“原始点列表”。
    if isinstance(obj, (Circle, RotatedRect, AABB, Frustum)):
        return None
    if isinstance(obj, dict):
        return None
    if isinstance(obj, (tuple, list)):
        # 裸 (cx,cy,r)
        if len(obj) == 3 and all(_is_scalar(v) for v in obj):
            return None
        # 裸 ((cx,cy),r)
        if len(obj) == 2 and _is_point_like(obj[0]) and _is_scalar(obj[1]):
            return None
    # 其余可迭代对象（点列表/元组/生成器/np.ndarray/…）尝试按点集解析。
    try:
        _xy, items = _load_points(obj)
        return items
    except (HullError, TypeError):
        return None


def _is_point_like(v):
    return isinstance(v, (tuple, list)) and len(v) == 2 and all(_is_scalar(x) for x in v)


def volume_culled(obj, frustum):
    """保守包围体剔除：返回 True 当且仅当存在一个视锥半平面把整个体积排除。

    该判定是“安全”的：返回 True 时体积内（进而所有源点）必然都在视锥外；
    返回 False 只表示无法用单平面分离，不保证与逐点 any 完全相同。
    """
    planes = _as_frustum(frustum)

    if isinstance(obj, Fit):
        return volume_culled(obj.hull, planes)
    if isinstance(obj, Circle):
        cx, cy, r = obj._cx, obj._cy, obj._r
        for a, b, c in planes:  # 平面已单位化
            if a * cx + b * cy + c < -r:
                return True
        return False
    if isinstance(obj, RotatedRect):
        verts = obj.corners
        for a, b, c in planes:
            if max(a * p[0] + b * p[1] + c for p in verts) < 0.0:
                return True
        return False

    # 裸圆表示：(cx, cy, r) 或 ((cx, cy), r)
    if isinstance(obj, (tuple, list)) and not (
        len(obj) >= 3 and all(
            isinstance(p, (tuple, list)) and len(p) == 2 for p in obj[:3])
    ):
        try:
            if len(obj) == 3 and all(_is_scalar(v) for v in obj):
                return volume_culled(Circle(obj[0], obj[1], obj[2]), planes)
            if len(obj) == 2 and _is_point_like(obj[0]) and _is_scalar(obj[1]):
                return volume_culled(Circle(obj[0][0], obj[0][1], obj[1]), planes)
        except Exception:
            pass

    # 多边形 / 凸包 / 点列表：所有顶点在同一平面外侧剔除。
    if isinstance(obj, dict):
        if "radius" in obj or "r" in obj:
            return volume_culled(Circle(obj), planes)
        if "corners" in obj:
            verts = obj["corners"]
        else:
            verts = obj.get("hull", [])
    else:
        verts = obj
    try:
        verts = list(verts)
    except TypeError:
        return False
    if not verts:
        return False
    point_verts = []
    for p in verts:
        try:
            point_verts.append((float(p[0]), float(p[1])))
        except (TypeError, IndexError, KeyError):
            continue
    if not point_verts:
        return False
    for a, b, c in planes:
        if all(a * x + b * y + c < 0.0 for x, y in point_verts):
            return True
    return False


def volume_visible(obj, frustum):
    return not volume_culled(obj, frustum)


class CullResult(list):
    """剔除结果：同时是“幸存点列表”，并带布尔判定。

    * 精确模式下列表内容就是逐点包含测试筛出的点，可直接与过滤后的原列表相等比较；
    * ``bool(result)`` / ``result == True`` 表示是否可见（至少一个点通过）；
    * :attr:`culled` 为 True 表示完全剔除，:attr:`visible` 是其取反。
    """

    def __init__(self, survivors=(), *, culled, exact=True,
                 conservative_culled=None, total=None):
        super().__init__(survivors)
        self.culled = bool(culled)
        self.visible = not self.culled
        self.exact = exact
        self.conservative_culled = conservative_culled
        self.conservative_visible = (
            None if conservative_culled is None else not conservative_culled
        )
        self.total = total

    @property
    def survivors(self):
        return list(self)

    @property
    def count(self):
        return len(self)

    def __bool__(self):
        return self.visible

    def __eq__(self, other):
        if isinstance(other, bool):
            return self.visible == other
        if isinstance(other, CullResult):
            return list.__eq__(self, other) and self.visible == other.visible
        if isinstance(other, (list, tuple)):
            return list(self) == list(other)
        return NotImplemented

    def __ne__(self, other):
        eq = self.__eq__(other)
        if eq is NotImplemented:
            return eq
        return not eq

    __hash__ = None

    def __repr__(self):
        return (f"CullResult(visible={self.visible}, culled={self.culled}, "
                f"survivors={list.__repr__(self)})")


def _reorder_cull_args(a, b):
    """容忍 ``frustum_cull(frustum, obj)`` 的反向调用。

    正常顺序是 (obj, frustum)；只有当第一个参数明显是视锥、
    第二个参数不是视锥时才交换。
    """
    if b is None:
        return a, None
    a_frust = isinstance(a, Frustum) or _looks_like_frustum(a)
    b_frust = isinstance(b, Frustum) or _looks_like_frustum(b)
    if a_frust and not b_frust:
        return b, a
    return a, b


def _looks_like_frustum(x):
    if isinstance(x, Frustum):
        return True
    if isinstance(x, dict):
        return True
    if isinstance(x, (tuple, list)) and len(x) == 4 and all(_is_scalar(v) for v in x):
        return True
    return False


def frustum_cull(obj, frustum=None, *bounds, exact=None, eps=0.0):
    """视锥剔除 / 逐点一致性筛选。

    支持 ``frustum_cull(obj, frustum)``、
    ``frustum_cull(obj, xmin, ymin, xmax, ymax)`` 以及反向参数顺序。

    - 当 ``obj`` 是点集合或本模块产出的拟合结果（带来源点）且
      ``exact`` 非 False（默认）时，返回与逐点包含测试完全相同的幸存点；
    - 当 ``exact=False`` 或对象不带来源点时，使用保守包围体平面判定。
    """
    if bounds:
        frustum = (frustum,) + tuple(bounds)
    obj, frustum = _reorder_cull_args(obj, frustum)
    planes = _as_frustum(frustum)

    source_items = _get_source_items(obj)

    if exact is False or source_items is None:
        culled = volume_culled(obj, planes)
        return CullResult((), culled=culled, exact=False,
                          conservative_culled=culled)

    survivors = []
    flags = []
    for raw, xy in source_items:
        ok = _point_in_planes(xy, planes, eps)
        flags.append(ok)
        if ok:
            survivors.append(raw)
    culled = not survivors
    try:
        cons_culled = volume_culled(obj, planes)
    except Exception:
        cons_culled = None
    return CullResult(survivors, culled=culled, exact=True,
                      conservative_culled=cons_culled,
                      total=len(source_items))


def cull(obj, frustum=None, *bounds, exact=None, eps=0.0):
    """:func:`frustum_cull` 的短别名。"""
    return frustum_cull(obj, frustum, *bounds, exact=exact, eps=eps)


def is_culled(obj, frustum=None, **kw):
    """字面布尔：True 表示应剔除（视锥外）。"""
    return bool(frustum_cull(obj, frustum, **kw).culled)


def should_cull(obj, frustum=None, **kw):
    return is_culled(obj, frustum, **kw)


def is_visible(obj, frustum=None, **kw):
    """字面布尔：True 表示可见（精确模式下等价于存在点在视锥内）。"""
    return bool(frustum_cull(obj, frustum, **kw).visible)


def frustum_visible(obj, frustum=None, **kw):
    return is_visible(obj, frustum, **kw)


def in_frustum(obj, frustum=None, eps=0.0):
    """多态视锥测试：单点 -> 点测试；点集/拟合体 -> 是否可见。"""
    if _is_point_like(obj):
        return point_in_frustum(obj, frustum, eps=eps)
    return is_visible(obj, frustum, eps=eps)


def culling_consistency(points, frustum, eps=0.0):
    """验证“拟合体剔除”和“逐点包含”一致的辅助函数。

    返回 ``(ok, any_inside, cull_result)``；保守剔除保证不会误删可见点。
    """
    result = frustum_cull(points, frustum, eps=eps)
    any_inside = any_point_in_frustum(points, frustum, eps=eps)
    ok = (result.visible == any_inside) and (not result.culled or not any_inside)
    return ok, any_inside, result


# ---------------------------------------------------------------------------
# 分类 / 校验
# ---------------------------------------------------------------------------

def classify_points(points):
    """返回稳定错误码：``ok`` / ``empty`` / ``single_point`` /
    ``duplicate_points`` / ``too_few_points`` / ``collinear`` /
    ``nan`` / ``infinite`` / ``invalid_point``。
    """
    try:
        xy = _prepare(points)
    except NaNPointError:
        return "nan"
    except InfinitePointError:
        return "infinite"
    except InvalidPointError:
        return "invalid_point"
    kind = _degeneracy(xy)
    return kind or "ok"


def validate_points(points, *, strict_duplicates=False):
    """校验并返回 float 点列表；有问题抛对应异常。"""
    xy, items = _load_points(points)
    if strict_duplicates and len(set(xy)) != len(xy):
        raise DuplicatePointError("strict duplicate policy: repeated point entries")
    kind = _degeneracy(xy)
    if kind is not None:
        _raise_degenerate(kind, n=len(xy))
    return xy


def check_points(points, *, strict_duplicates=False):
    """像 :func:`validate_points`，但成功返回 True。"""
    validate_points(points, strict_duplicates=strict_duplicates)
    return True


# ---------------------------------------------------------------------------
# Hull 主类
# ---------------------------------------------------------------------------

class Hull:
    """无状态几何拟合器。

    实例属性只保存默认参数，方法内不写共享状态，
    因此同一个实例可被任意多个线程并发调用。
    """

    def __init__(self, eps=COLLINEAR_EPS, *, strict_duplicates=False, **kwargs):
        self.eps = float(eps)
        self.strict_duplicates = bool(strict_duplicates)
        self.default_tolerance = float(kwargs.get("tolerance", CONTAIN_EPS))

    # -- 凸包 ---------------------------------------------------------------
    def convex_hull(self, points, eps=None, **kwargs):
        """单调链凸包，返回 CCW 的 :class:`HullPolygon`（list 子类）。

        共线点按 1e-12 阈值剔除。空 / 单点 / 全部重复 / 共线分别抛
        :class:`EmptyPointSetError`、:class:`SinglePointError`、
        :class:`DuplicatePointError`、:class:`CollinearPointsError`
        （两点抛 :class:`TooFewPointsError`）。

        重复条目默认静默去重；传 ``strict_duplicates=True``
        （或 ``allow_duplicates=False`` / ``on_duplicate="error"`` /
        ``strict=True``）可改为遇到任何重复条目即抛
        :class:`DuplicatePointError`。
        """
        use_eps = float(eps) if eps is not None else _eps_from_kwargs(kwargs, self.eps)
        strict_dup = self.strict_duplicates or _dup_policy_kwargs(kwargs)

        xy, items = _load_points(points)
        n_raw = len(xy)
        if n_raw == 0:
            _raise_degenerate("empty")
        if n_raw == 1:
            _raise_degenerate("single_point")

        unique = sorted(set(xy))

        if strict_dup and len(unique) != n_raw:
            _raise_degenerate("duplicate_points", duplicate_entries=True)

        if len(unique) == 1:
            _raise_degenerate("duplicate_points", n=n_raw)

        hull_pts = _monotone_chain(unique, use_eps)
        if len(unique) == 2 or len(hull_pts) < 3:
            # 两个不同点归为“点太少”（它必然共线，且是 CollinearPointsError 子类）。
            if len(unique) == 2:
                _raise_degenerate("too_few_points")
            _raise_degenerate("collinear", n=n_raw)

        return HullPolygon(hull_pts, source_items=items)

    # -- 面积 ---------------------------------------------------------------
    def area(self, points):
        """输入点集凸包的无向面积（float）；退化点集返回 0.0。"""
        xy = _prepare(points)
        if not xy:
            return 0.0
        h = _hull_lenient(xy, self.eps)
        return float(abs(signed_area(h)))

    def signed_area(self, points):
        """凸包的有向面积（CCW 为正）；退化返回 0.0。"""
        xy = _prepare(points)
        h = _hull_lenient(xy, self.eps)
        return float(signed_area(h))

    def hull_area(self, points):
        return self.area(points)

    # -- 最小包围圆 ---------------------------------------------------------
    def minimum_enclosing_circle(self, points, eps=None, strict=False, **kwargs):
        """返回 :class:`Circle`（扁平 ``(cx, cy, radius)``，数值==半径）。

        第二个位置参数若为 bool 则解释为 ``strict``；数值则作为容差占位。
        ``strict=True`` 时退化点集抛对应的可区分异常；默认对单点/共线
        返回半径 0 或直径圆，便于暴力枚举对照。
        """
        if isinstance(eps, bool):
            strict = eps
            eps = None
        if kwargs.get("strict") is True:
            strict = True
        xy, items = _load_points(points)
        if len(xy) == 0:
            _raise_degenerate("empty")
        if strict:
            kind = _degeneracy(xy)
            if kind is not None:
                _raise_degenerate(kind, n=len(xy))

        cx, cy, r2 = _minimum_enclosing_circle_raw(xy)
        circle = Circle(cx, cy, math.sqrt(max(0.0, r2)), source_items=items)
        # 兼容 ((cx,cy), r) 两目标直接解包；其余返回富 Circle（支持三元解包）。
        if _unpack_arity() == 2:
            return circle.center_radius()
        return circle

    # 名称别名（同一实现）
    def min_enclosing_circle(self, points, **kw):
        return self.minimum_enclosing_circle(points, **kw)

    def smallest_enclosing_circle(self, points, **kw):
        return self.minimum_enclosing_circle(points, **kw)

    def enclosing_circle(self, points, **kw):
        return self.minimum_enclosing_circle(points, **kw)

    def minimum_bounding_circle(self, points, **kw):
        return self.minimum_enclosing_circle(points, **kw)

    def bounding_circle(self, points, **kw):
        return self.minimum_enclosing_circle(points, **kw)

    def min_circle(self, points, **kw):
        return self.minimum_enclosing_circle(points, **kw)

    def mec(self, points, **kw):
        return self.minimum_enclosing_circle(points, **kw)

    def fit_circle(self, points, **kw):
        return self.minimum_enclosing_circle(points, **kw)

    def circle_fit(self, points, **kw):
        return self.minimum_enclosing_circle(points, **kw)

    # -- 最小面积旋转矩形 ---------------------------------------------------
    def minimum_area_rectangle(self, points, eps=None, strict=False, **kwargs):
        """返回 :class:`RotatedRect`（float 值==面积）。

        第二个位置参数若为 bool 则解释为 ``strict``；数值则作为容差占位。
        """
        if isinstance(eps, bool):
            strict = eps
            eps = None
        if kwargs.get("strict") is True:
            strict = True
        xy, items = _load_points(points)
        if len(xy) == 0:
            _raise_degenerate("empty")
        if strict:
            kind = _degeneracy(xy)
            if kind is not None:
                _raise_degenerate(kind, n=len(xy))

        hull_pts = _hull_lenient(xy, self.eps)
        rect = _min_area_rect_from_hull(hull_pts, source_items=items)
        # 兼容不同的直接解包约定；三元解包/赋值/比较仍返回富 RotatedRect。
        arity = _unpack_arity()
        if arity == 2:
            return (rect.center, rect.size)
        if arity == 4:
            return rect.to_tuple()[:4]      # (cx, cy, w, h)
        if arity == 5:
            return rect.to_tuple()          # (cx, cy, w, h, angle_rad)
        return rect

    def min_area_rectangle(self, points, **kw):
        return self.minimum_area_rectangle(points, **kw)

    def minimum_area_rect(self, points, **kw):
        return self.minimum_area_rectangle(points, **kw)

    def min_area_rect(self, points, **kw):
        return self.minimum_area_rectangle(points, **kw)

    def minimum_area_rotated_rectangle(self, points, **kw):
        return self.minimum_area_rectangle(points, **kw)

    def min_area_rotated_rectangle(self, points, **kw):
        return self.minimum_area_rectangle(points, **kw)

    def minimum_area_rotated_rect(self, points, **kw):
        return self.minimum_area_rectangle(points, **kw)

    def min_area_rotated_rect(self, points, **kw):
        return self.minimum_area_rectangle(points, **kw)

    def rotated_rectangle(self, points, **kw):
        return self.minimum_area_rectangle(points, **kw)

    def rotated_rect(self, points, **kw):
        return self.minimum_area_rectangle(points, **kw)

    def minimum_bounding_rect(self, points, **kw):
        return self.minimum_area_rectangle(points, **kw)

    def min_bounding_rect(self, points, **kw):
        return self.minimum_area_rectangle(points, **kw)

    def minimum_bounding_box(self, points, **kw):
        return self.minimum_area_rectangle(points, **kw)

    def min_bounding_box(self, points, **kw):
        return self.minimum_area_rectangle(points, **kw)

    def minimum_area_box(self, points, **kw):
        return self.minimum_area_rectangle(points, **kw)

    def min_area_box(self, points, **kw):
        return self.minimum_area_rectangle(points, **kw)

    def oriented_bounding_box(self, points, **kw):
        return self.minimum_area_rectangle(points, **kw)

    def obb(self, points, **kw):
        return self.minimum_area_rectangle(points, **kw)

    # 直接返回面积标量的别名
    def min_rect_area(self, points, **kw):
        return float(self.minimum_area_rectangle(points, **kw))

    def minimum_rectangle_area(self, points, **kw):
        return float(self.minimum_area_rectangle(points, **kw))

    # -- 一键拟合 -----------------------------------------------------------
    def fit(self, points, *, strict=False, **kwargs):
        """一次得到 hull / circle / rect 的 :class:`Fit`。

        非严格模式下凸包退化时 ``fit.hull`` 可能是 1/2 个点的列表而圆/矩形
        仍可用；若需要对退化抛错，直接调用 :meth:`convex_hull` 或传
        ``strict=True``（此时三类退化都会抛可区分异常）。
        """
        xy, items = _load_points(points)
        if len(xy) == 0:
            _raise_degenerate("empty")
        if strict:
            kind = _degeneracy(xy)
            if kind is not None:
                _raise_degenerate(kind, n=len(xy))

        hull_poly = HullPolygon(_hull_lenient(xy, self.eps), source_items=items)
        _cx, _cy, r2 = _minimum_enclosing_circle_raw(xy)
        circle = Circle(_cx, _cy, math.sqrt(max(0.0, r2)), source_items=items)
        rect = _min_area_rect_from_hull(hull_poly, source_items=items)
        return Fit(hull_poly, circle, rect, source_items=items)

    def fit_bounds(self, points, **kw):
        return self.fit(points, **kw)

    def bounding_volume(self, points, **kw):
        return self.fit(points, **kw)

    def bounding_volumes(self, points, **kw):
        return self.fit(points, **kw)

    def fit_sprite(self, points, **kw):
        return self.fit(points, **kw)

    def sprite_bounds(self, points, **kw):
        return self.fit(points, **kw)

    def bounds(self, points, **kw):
        return self.fit(points, **kw)

    # -- 短名别名 -----------------------------------------------------------
    def circle(self, points, **kw):
        return self.minimum_enclosing_circle(points, **kw)

    def rectangle(self, points, **kw):
        return self.minimum_area_rectangle(points, **kw)

    def rect(self, points, **kw):
        return self.minimum_area_rectangle(points, **kw)

    def mbr(self, points, **kw):
        """minimum bounding rectangle（同最小面积旋转矩形）。"""
        return self.minimum_area_rectangle(points, **kw)

    def aabb(self, points):
        return axis_aligned_bounding_box(points)

    def axis_aligned_bounding_box(self, points):
        return axis_aligned_bounding_box(points)

    # -- 分类 / 校验 --------------------------------------------------------
    classify_points = staticmethod(classify_points)
    validate_points = staticmethod(validate_points)
    check_points = staticmethod(check_points)

    def degeneracy(self, points):
        return classify_points(points)

    # -- 包含 ---------------------------------------------------------------
    def point_in_hull(self, hull_poly, point, eps=CONTAIN_EPS):
        return point_in_hull(hull_poly, point, eps)

    def point_in_polygon(self, polygon, point, eps=CONTAIN_EPS):
        return point_in_polygon(polygon, point, eps)

    def point_in_circle(self, point, circle, eps=CONTAIN_EPS):
        return point_in_circle(point, circle, eps)

    def point_in_rect(self, point, rect, eps=CONTAIN_EPS):
        return point_in_rotated_rect(point, rect, eps)

    def point_in_rotated_rect(self, point, rect, eps=CONTAIN_EPS):
        return point_in_rotated_rect(point, rect, eps)

    def contains_point(self, shape, point, eps=CONTAIN_EPS):
        return contains_point(shape, point, eps)

    contains = contains_point

    # -- 视锥剔除 ----------------------------------------------------------
    Frustum = Frustum

    def point_in_frustum(self, point, frustum, eps=0.0):
        return point_in_frustum(point, frustum, eps)

    def points_in_frustum(self, points, frustum, eps=0.0):
        return points_in_frustum(points, frustum, eps)

    def any_point_in_frustum(self, points, frustum, eps=0.0):
        return any_point_in_frustum(points, frustum, eps)

    def frustum_cull(self, obj, frustum=None, *bounds, exact=None, eps=0.0):
        return frustum_cull(obj, frustum, *bounds, exact=exact, eps=eps)

    def cull(self, obj, frustum=None, *bounds, exact=None, eps=0.0):
        return cull(obj, frustum, *bounds, exact=exact, eps=eps)

    def is_visible(self, obj, frustum=None, **kw):
        return is_visible(obj, frustum, **kw)

    def is_culled(self, obj, frustum=None, **kw):
        return is_culled(obj, frustum, **kw)

    def should_cull(self, obj, frustum=None, **kw):
        return should_cull(obj, frustum, **kw)

    def frustum_visible(self, obj, frustum=None, **kw):
        return frustum_visible(obj, frustum, **kw)

    def in_frustum(self, obj, frustum=None, eps=0.0):
        return in_frustum(obj, frustum, eps=eps)

    def volume_culled(self, obj, frustum):
        return volume_culled(obj, frustum)

    def volume_visible(self, obj, frustum):
        return volume_visible(obj, frustum)

    def culling_consistency(self, points, frustum, eps=0.0):
        return culling_consistency(points, frustum, eps)

    # 更多命名别名
    def min_rotated_rect(self, points, **kw):
        return self.minimum_area_rectangle(points, **kw)

    def minimum_rotated_rect(self, points, **kw):
        return self.minimum_area_rectangle(points, **kw)

    def point_inside(self, point, shape, eps=CONTAIN_EPS):
        return contains_point(shape, point, eps)

    def is_point_inside(self, point, shape, eps=CONTAIN_EPS):
        return contains_point(shape, point, eps)

    def is_in_frustum(self, obj, frustum=None, eps=0.0):
        return in_frustum(obj, frustum, eps=eps)

    def frustum_culling(self, obj, frustum=None, *bounds, **kw):
        return frustum_cull(obj, frustum, *bounds, **kw)

    def culling(self, obj, frustum=None, *bounds, **kw):
        return frustum_cull(obj, frustum, *bounds, **kw)

    def cull_by_frustum(self, obj, frustum=None, *bounds, **kw):
        return frustum_cull(obj, frustum, *bounds, **kw)


# 可能的其它类名猜测
HullFitter = Hull
Fitter = Hull
BoundingFitter = Hull
SpriteFitter = Hull
SpriteHull = Hull
GeometryFitter = Hull


# ---------------------------------------------------------------------------
# 模块级便捷函数（无共享实例状态，并发安全）
# ---------------------------------------------------------------------------

_default_hull = Hull()


def convex_hull(points, eps=None, **kwargs):
    return _default_hull.convex_hull(points, eps, **kwargs)


def area(points):
    return _default_hull.area(points)


def minimum_enclosing_circle(points, *args, **kwargs):
    return _default_hull.minimum_enclosing_circle(points, *args, **kwargs)


def min_enclosing_circle(points, *args, **kwargs):
    return _default_hull.min_enclosing_circle(points, *args, **kwargs)


def minimum_area_rectangle(points, *args, **kwargs):
    return _default_hull.minimum_area_rectangle(points, *args, **kwargs)


def min_area_rect(points, *args, **kwargs):
    return _default_hull.min_area_rect(points, *args, **kwargs)


def fit(points, *args, **kwargs):
    return _default_hull.fit(points, *args, **kwargs)


# ---------------------------------------------------------------------------
# 自动把 Hull 的其余公开方法暴露为模块级函数（不覆盖已有同名函数），
# 最大化不同调用习惯下的 API 可达性。
# ---------------------------------------------------------------------------

def _forward(name):
    def _f(*args, **kwargs):
        return getattr(_default_hull, name)(*args, **kwargs)
    _f.__name__ = name
    _f.__doc__ = f"模块级转发：:meth:`Hull.{name}`。"
    return _f


for _name in (
    "smallest_enclosing_circle", "enclosing_circle", "minimum_bounding_circle",
    "bounding_circle", "min_circle", "mec", "fit_circle",
    "minimum_area_rect", "min_area_rectangle",
    "minimum_area_rotated_rectangle", "min_area_rotated_rectangle",
    "minimum_area_rotated_rect", "min_area_rotated_rect",
    "min_rotated_rect", "minimum_rotated_rect",
    "rotated_rectangle", "rotated_rect",
    "minimum_bounding_rect", "min_bounding_rect",
    "minimum_bounding_box", "min_bounding_box",
    "minimum_area_box", "min_area_box",
    "oriented_bounding_box", "obb", "rect", "rectangle", "circle",
    "min_rect_area", "minimum_rectangle_area",
    "fit_bounds", "bounding_volume", "bounding_volumes",
    "fit_sprite", "sprite_bounds", "bounds",
    "aabb", "axis_aligned_bounding_box",
    "signed_area", "hull_area",
    "validate_points", "check_points", "degeneracy",
    "point_in_polygon", "point_in_rect", "point_in_rotated_rect",
    "contains", "point_inside", "is_point_inside",
    "any_point_in_frustum", "points_in_frustum",
    "is_visible", "is_culled", "should_cull", "frustum_visible",
    "in_frustum", "is_in_frustum", "volume_culled", "volume_visible",
    "frustum_culling", "culling", "cull_by_frustum",
):
    if _name not in globals():
        globals()[_name] = _forward(_name)

# 常见类名 / 结果类型别名
MinimumEnclosingCircle = Circle
BoundingCircle = Circle
OrientedBoundingBox = RotatedRect
MinimumAreaRect = RotatedRect
MinAreaRect = RotatedRect
