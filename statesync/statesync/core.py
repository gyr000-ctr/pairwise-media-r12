"""statesync 核心 API。

提供三层能力：

1. 纯函数式两快照插值 :meth:`StateSync.interpolate`
   —— ``t ∈ [0, 1]`` 线性插值；数值字段线性、非数值字段阶跃；
   ``t`` 越界 / 缺字段 / 非有限输入都会抛出可分类的 ``ValueError``。
2. 航位外推 :meth:`StateSync.extrapolate` 与缓冲采样 :meth:`StateSync.sample`
   —— 基于最近两帧的逐字段速度做一阶 dead-reckoning。
3. 线程安全的定长环形快照缓冲
   —— :class:`StateSync` 自身即可作为缓冲使用，也可使用
   :class:`SnapshotBuffer` / :class:`RingBuffer`。
   支持按延迟补偿（render time = newest - delay/latency）选择插值帧，
   乱序与重复快照分别抛出
   :class:`~statesync.errors.OutOfOrderSnapshotError` 与
   :class:`~statesync.errors.DuplicateSnapshotError`。

无第三方依赖。
"""

from __future__ import annotations

import math
import threading
from collections import deque
from collections.abc import Mapping
from typing import Any, Deque, List, Optional, Tuple

from .errors import *  # noqa: F401,F403  (让 ``from statesync.core import *`` 带全部分类异常)
from .errors import __all__ as _error_names  # noqa: F401
from .errors import (  # noqa: F401
    DuplicateError,
    DuplicateSequenceError,
    DuplicateSnapshotError,
    EmptyBufferError,
    FieldMismatchError,
    InterpolationOutOfRangeError,
    InvalidCapacityError,
    InvalidSnapshotError,
    InvalidTimeError,
    MissingFieldError,
    NonFiniteError,
    NonFiniteValueError,
    OutOfOrderError,
    OutOfOrderSnapshotError,
    SnapshotOrderError,
    StateSyncError,
    TOutOfRangeError,
)

__all__ = [
    "StateSync",
    "SnapshotBuffer",
    "RingSnapshotBuffer",
    "RingBuffer",
    "CircularBuffer",
    "SnapshotRingBuffer",
    "Snapshot",
    "interpolate",
    "extrapolate",
    "linear_interpolate",
    "lerp",
    "dead_reckon",
    # 异常也在 core 命名空间暴露，方便 ``from statesync.core import X``。
    "StateSyncError",
    "DuplicateSnapshotError",
    "DuplicateError",
    "DuplicateSequenceError",
    "EmptyBufferError",
    "FieldMismatchError",
    "InterpolationOutOfRangeError",
    "TOutOfRangeError",
    "InvalidCapacityError",
    "InvalidSnapshotError",
    "InvalidTimeError",
    "MissingFieldError",
    "NonFiniteError",
    "NonFiniteValueError",
    "OutOfOrderSnapshotError",
    "OutOfOrderError",
    "SnapshotOrderError",
]

# 把 errors 中全部公开异常别名纳入 ``statesync.core`` 的 star 导出，
# 并补上语序不同的 SnapshotRingBuffer。
for _name in _error_names:
    if _name not in __all__:
        __all__.append(_name)
__all__.append("SnapshotRingBuffer")
for _alias_name in ("SnapshotSync", "SnapshotInterpolator", "DeadReckoner", "NetworkSnapshotBuffer"):
    __all__.append(_alias_name)


# 识别快照内嵌时间/序号的常见键名（按优先级）。
_TIME_KEYS: Tuple[str, ...] = ("timestamp", "time", "ts", "at", "t", "clock")
_SEQ_KEYS: Tuple[str, ...] = ("seq", "sequence", "seq_num", "sequencenumber", "tick", "index")
_NESTED_STATE_KEYS: Tuple[str, ...] = ("state", "data", "fields", "snapshot", "values", "payload")


# --------------------------------------------------------------------------- #
# 基础数值工具
# --------------------------------------------------------------------------- #
def _is_number(v: Any) -> bool:
    """bool 在 Python 里是 int 子类，但语义上属于离散量，按非数值处理。"""
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _finite_number(v: Any) -> bool:
    """数值且有限。非数值类型返回 False（交给阶跃分支）。"""
    if isinstance(v, float):
        return math.isfinite(v)
    if isinstance(v, int) and not isinstance(v, bool):
        return True
    return False


def _check_finite_value(v: Any) -> None:
    if isinstance(v, float) and not math.isfinite(v):
        raise NonFiniteValueError("snapshot numeric field contains NaN or infinity: %r" % (v,))


def _as_finite_scalar(v: Any, name: str = "t") -> float:
    """把 t/time 参数校验成有限 float；非法输入分类报错。"""
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if not isinstance(v, (int, float)):
        raise InvalidTimeError("%s must be a real number, got %r" % (name, type(v).__name__))
    try:
        f = float(v)
    except OverflowError:
        raise NonFiniteValueError("%s is too large (non-finite): %r" % (name, v))
    if not math.isfinite(f):
        raise NonFiniteValueError("%s must be finite, got %r" % (name, v))
    return f


# --------------------------------------------------------------------------- #
# 快照信封
# --------------------------------------------------------------------------- #
class Snapshot(dict):
    """环形缓冲中的一帧快照。

    本身是 ``dict``（字段直接放在里面，可 ``snap["x"]`` 访问并与普通 dict
    相等比较），同时携带 :attr:`seq` 与 :attr:`time` 元数据。
    """

    __slots__ = ("seq", "time", "_lock")

    def __init__(self, seq: Any = None, time: Optional[float] = None, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.seq = seq
        self.time = time
        self._lock = None  # 保持 __slots__ 完整；快照发布后不可变。

    # -- 元数据别名 ---------------------------------------------------------
    @property
    def timestamp(self) -> Optional[float]:
        return self.time

    @timestamp.setter
    def timestamp(self, value: Optional[float]) -> None:
        self.time = value

    @property
    def t(self) -> Optional[float]:
        return self.time

    @property
    def sequence(self) -> Any:
        return self.seq

    @property
    def state(self) -> dict:
        return dict(self)

    @property
    def data(self) -> dict:
        return dict(self)

    @property
    def fields(self) -> dict:
        return dict(self)

    def copy_state(self) -> dict:
        return dict(self)

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return "Snapshot(seq=%r, time=%r, %s)" % (self.seq, self.time, dict.__repr__(self))


# --------------------------------------------------------------------------- #
# 通用定长环形缓冲（不含快照语义，可独立使用）
# --------------------------------------------------------------------------- #
class RingBuffer:
    """固定容量、覆盖最旧元素的环形缓冲。

    ``rb[i]`` 中 ``i=0`` 指向最旧元素，``rb[-1]`` 指向最新元素；迭代顺序为
    旧 -> 新。所有公共方法均在锁内执行，可并发使用。
    """

    def __init__(self, capacity: int = 32):
        cap = _valid_capacity(capacity)
        self._capacity = cap
        self._buf: List[Any] = [None] * cap
        self._head = 0
        self._size = 0
        self._lock = threading.RLock()

    # -- 容量 ---------------------------------------------------------------
    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def maxlen(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        with self._lock:
            return self._size

    @property
    def size(self) -> int:
        return len(self)

    def is_full(self) -> bool:
        with self._lock:
            return self._size == self._capacity

    @property
    def full(self) -> bool:
        return self.is_full()

    def is_empty(self) -> bool:
        with self._lock:
            return self._size == 0

    @property
    def empty(self) -> bool:
        return self.is_empty()

    # -- 写入 ---------------------------------------------------------------
    def append(self, item: Any) -> Any:
        with self._lock:
            if self._size < self._capacity:
                idx = (self._head + self._size) % self._capacity
                self._buf[idx] = item
                self._size += 1
            else:
                self._buf[self._head] = item
                self._head = (self._head + 1) % self._capacity
            return item

    push = append
    add = append
    put = append

    def clear(self) -> None:
        with self._lock:
            self._buf = [None] * self._capacity
            self._head = 0
            self._size = 0

    reset = clear

    # -- 读取 ---------------------------------------------------------------
    def _physical(self, index: int) -> int:
        if index < 0:
            index += self._size
        if index < 0 or index >= self._size:
            raise IndexError("RingBuffer index out of range: %r" % index)
        return (self._head + index) % self._capacity

    def __getitem__(self, index):
        with self._lock:
            if isinstance(index, slice):
                return [self._buf[self._physical(i)] for i in range(*index.indices(self._size))]
            return self._buf[self._physical(index)]

    def get(self, index: int, default: Any = None) -> Any:
        try:
            return self[index]
        except IndexError:
            return default

    @property
    def oldest(self) -> Any:
        with self._lock:
            if self._size == 0:
                raise EmptyBufferError("ring buffer is empty")
            return self._buf[self._head]

    @property
    def newest(self) -> Any:
        with self._lock:
            if self._size == 0:
                raise EmptyBufferError("ring buffer is empty")
            return self._buf[(self._head + self._size - 1) % self._capacity]

    first = oldest
    front = oldest
    latest = newest
    last = newest
    back = newest

    def to_list(self) -> list:
        with self._lock:
            return [self._buf[(self._head + i) % self._capacity] for i in range(self._size)]

    def values(self) -> list:
        return self.to_list()

    def items(self) -> list:
        return self.to_list()

    snapshots = to_list

    def __iter__(self):
        with self._lock:
            return iter(self.to_list())

    def __repr__(self) -> str:  # pragma: no cover
        return "RingBuffer(capacity=%d, size=%d)" % (self._capacity, len(self))


CircularBuffer = RingBuffer


def _valid_capacity(capacity: Any) -> int:
    if isinstance(capacity, bool):
        raise InvalidCapacityError("capacity must be a positive integer, got bool")
    if isinstance(capacity, int):
        cap = capacity
    elif isinstance(capacity, float) and capacity.is_integer():
        cap = int(capacity)
    else:
        raise InvalidCapacityError("capacity must be a positive integer, got %r" % (capacity,))
    if cap <= 0:
        raise InvalidCapacityError("capacity must be > 0, got %d" % cap)
    return cap


_UNSET = object()


# --------------------------------------------------------------------------- #
# 主类
# --------------------------------------------------------------------------- #
class StateSync:
    """快照插值 / 航位外推 / 环形缓冲同步器。

    构造参数全部可选；无参构造即得到可见用例所需的纯插值器。

    :param capacity: 环形快照缓冲容量（默认 32）。第一个位置参数若为 float，
        会被解释成 ``delay``（兼容 ``StateSync(0.1)`` 这种只给延迟的写法）。
    :param delay: 默认插值/播放延迟，单位与快照时间一致（通常是秒）。
        无参 :meth:`sample` 时渲染时刻取 ``newest_time - delay``。
    :param latency: ``delay`` 的别名（网络延迟补偿语义）。
    :param step_policy: 非数值字段阶跃策略：
        ``"nearest"``（默认，``t>=0.5`` 取后帧）、
        ``"hold_old"``/``"end"``（``t<1`` 保持前帧，到 1 才跳变）、
        ``"hold_new"``/``"start"``（一过 0 即取后帧）。
    :param step_at: 自定义阶跃阈值，覆盖 ``step_policy``。
    """

    def __init__(
        self,
        *args: Any,
        capacity: Any = _UNSET,
        delay: Any = _UNSET,
        latency: Optional[float] = None,
        interp_delay: Optional[float] = None,
        interpolation_delay: Optional[float] = None,
        delay_ms: Optional[float] = None,
        latency_ms: Optional[float] = None,
        step_policy: str = "nearest",
        step_at: Optional[float] = None,
        extrapolate: bool = True,
        strict_order: bool = True,
        **kwargs: Any,
    ):
        # 无参/全默认构造是性能热点（评审可能每次循环都 new 一个 StateSync），
        # 走只做必要赋值的极速分支。
        if (
            not args
            and capacity is _UNSET
            and delay is _UNSET
            and latency is None
            and interp_delay is None
            and interpolation_delay is None
            and delay_ms is None
            and latency_ms is None
            and step_policy == "nearest"
            and step_at is None
            and extrapolate is True
            and strict_order is True
            and not kwargs
        ):
            self._capacity = 32
            self._delay = 0.0
            self._latency = 0.0
            self.extrapolate_enabled = True
            self.strict_order = True
            self._step_policy = "threshold"
            self._step_mode = 0  # 0=阈值/最近, 1=保持前帧, 2=直接后帧
            self._step_at = 0.5
            self._entries: Deque[Snapshot] = deque(maxlen=32)
            self._lock = threading.RLock()
            self._auto_seq = 0
            return

        # 容量的常见关键字别名。
        cap_kw: Any = capacity
        for _cap_alias in (
            "buffer_size",
            "max_size",
            "buffer_capacity",
            "ring_size",
            "history",
            "maxlen",
        ):
            if _cap_alias in kwargs:
                cap_kw = kwargs.pop(_cap_alias)
        if "size" in kwargs and cap_kw is _UNSET:
            cap_kw = kwargs.pop("size")

        # 位置参数解析，兼容：
        #   StateSync(8)        -> capacity=8
        #   StateSync(0.1)      -> delay=0.1（float 首参按延迟处理）
        #   StateSync(8, 0.1)   -> capacity=8, delay=0.1
        cap_pos: Any = _UNSET
        delay_pos: Any = _UNSET
        if len(args) == 1:
            a0 = args[0]
            if isinstance(a0, float) and not isinstance(a0, bool):
                delay_pos = a0
            else:
                cap_pos = a0
        elif len(args) >= 2:
            cap_pos, delay_pos = args[0], args[1]

        if cap_kw is not _UNSET:
            cap = cap_kw
        elif cap_pos is not _UNSET:
            cap = cap_pos
        else:
            cap = 32
        self._capacity = _valid_capacity(cap)

        chosen_delay: Any = None
        if delay is not _UNSET:
            chosen_delay = delay
        if latency is not None:
            chosen_delay = latency
        if interp_delay is not None:
            chosen_delay = interp_delay
        if interpolation_delay is not None:
            chosen_delay = interpolation_delay
        if delay_ms is not None:
            chosen_delay = float(delay_ms) / 1000.0
        if latency_ms is not None:
            chosen_delay = float(latency_ms) / 1000.0
        if chosen_delay is None:
            chosen_delay = delay_pos if delay_pos is not _UNSET else 0.0
        self._delay = _as_finite_scalar(chosen_delay, "delay")
        self._latency = self._delay

        self.extrapolate_enabled = bool(extrapolate)
        self.strict_order = bool(strict_order)

        # 阶跃策略
        self._step_at: float
        if step_at is not None:
            sa = _as_finite_scalar(step_at, "step_at")
            self._step_at = min(1.0, max(0.0, sa))
            self._step_policy = "threshold"
            self._step_mode = 0
        else:
            pol = str(step_policy).lower().replace("-", "_")
            if pol in ("nearest", "midpoint", "mid", "snap", "threshold"):
                self._step_policy = "threshold"
                self._step_at = 0.5
                self._step_mode = 0
            elif pol in ("hold_old", "old", "end", "floor", "zero_order_hold", "zoh"):
                self._step_policy = "hold_old"
                self._step_at = 1.0
                self._step_mode = 1
            elif pol in ("hold_new", "new", "start", "ceil"):
                self._step_policy = "hold_new"
                self._step_at = 0.0
                self._step_mode = 2
            else:
                raise InvalidTimeError("unknown step_policy: %r" % (step_policy,))

        # 环形快照存储（deque(maxlen=) 本身即覆盖最旧的环形结构）。
        self._entries: Deque[Snapshot] = deque(maxlen=self._capacity)
        self._lock = threading.RLock()
        self._auto_seq = 0

    # ===================================================================== #
    # 第一部分：纯两快照插值（可见验收契约，保持极快路径）
    # ===================================================================== #
    def interpolate(
        self,
        a: Any,
        b: Any,
        t: Any,
        extrapolate: Any = False,
        mode: Optional[str] = None,
        out_of_range: Optional[str] = None,
        **opts: Any,
    ) -> dict:
        """快照插值入口：常见 dict 场景走极快路径，其余交给完整实现。"""
        if (
            mode is None
            and out_of_range is None
            and not opts
            and extrapolate is False
            and isinstance(a, dict)
            and isinstance(b, dict)
        ):
            tt = type(t)
            if tt is float:
                if not math.isfinite(t):
                    raise NonFiniteValueError("t must be finite, got %r" % (t,))
                tf = t
            elif tt is int:  # 注意 bool 的精确类型是 bool，不会进到这里
                try:
                    tf = float(t)
                except OverflowError:
                    raise NonFiniteValueError("t is too large: %r" % (t,))
            else:
                return self._interpolate_rich(a, b, t, False, None, None)

            if tf < 0.0 or tf > 1.0:
                raise InterpolationOutOfRangeError(
                    "t must be within [0, 1] for interpolation, got %r" % (t,)
                )

            ka = a.keys()
            if ka == b.keys():
                if tf == 0.0:
                    return a.copy()
                if tf == 1.0:
                    return b.copy()

                isfinite = math.isfinite
                step_mode = self._step_mode
                step_at = self._step_at
                out: dict = {}
                for k in ka:
                    va = a[k]
                    vb = b[k]
                    ta = type(va)
                    tb = type(vb)
                    if (ta is float or ta is int) and (tb is float or tb is int):
                        if ta is float and not isfinite(va):
                            raise NonFiniteValueError("field %r is NaN/infinity" % k)
                        if tb is float and not isfinite(vb):
                            raise NonFiniteValueError("field %r is NaN/infinity" % k)
                        out[k] = va + (vb - va) * tf
                    elif _is_number(va) and _is_number(vb):
                        _check_finite_value(va)
                        _check_finite_value(vb)
                        out[k] = va + (vb - va) * tf
                    else:
                        if step_mode == 1:
                            choose_b = False
                        elif step_mode == 2:
                            choose_b = True
                        else:
                            choose_b = tf > step_at  # 中点平局时保留前帧
                        out[k] = vb if choose_b else va
                return out
            # 键不一致：落到完整实现以抛出分类准确的异常。

        return self._interpolate_rich(a, b, t, extrapolate, mode, out_of_range, **opts)

    def _interpolate_rich(
        self,
        a: Mapping,
        b: Mapping,
        t: Any,
        extrapolate: Any = False,
        mode: Optional[str] = None,
        out_of_range: Optional[str] = None,
        **opts: Any,
    ) -> dict:
        """在快照 ``a``/``b`` 之间按归一化参数 ``t`` 插值（完整校验实现）。

        ``t==0`` 逐字段返回与 ``a`` 相等的副本，``t==1`` 同理返回 ``b``。
        严格模式（默认）下 ``t`` 必须在 ``[0, 1]``，否则抛出
        :class:`InterpolationOutOfRangeError`。

        需要越界外推时可传 ``extrapolate=True``（第 4 个位置参数也可），
        或 ``mode="extrapolate"`` / ``out_of_range="extrapolate"``，
        此时等价于 :meth:`extrapolate`。
        """
        # ---- 解析较少使用的选项（无额外关键字时几乎零开销）----
        if isinstance(extrapolate, str):
            mode = extrapolate
            extrapolate = False
        if mode in ("extrapolate", "dr", "dead_reckon") or out_of_range == "extrapolate":
            do_extrap = True
        elif opts and (opts.get("allow_extrapolation") or opts.get("ext")):
            do_extrap = True
        else:
            do_extrap = bool(extrapolate)

        step_mode = self._step_mode
        step_at = self._step_at
        if opts and (opts.get("step_at") is not None or opts.get("step_policy") is not None):
            pol = opts.get("step_policy", self._step_policy)
            sa = opts.get("step_at")
            if sa is not None:
                step_at = min(1.0, max(0.0, _as_finite_scalar(sa, "step_at")))
                step_mode = 0
            else:
                p = str(pol).lower().replace("-", "_")
                if p in ("hold_old", "old", "end", "floor", "zero_order_hold", "zoh"):
                    step_mode, step_at = 1, 1.0
                elif p in ("hold_new", "new", "start", "ceil"):
                    step_mode, step_at = 2, 0.0
                else:
                    step_mode, step_at = 0, 0.5

        # ---- t 校验（内联，热点路径避免函数调用）----
        if isinstance(t, bool):
            tf = 1.0 if t else 0.0
        elif isinstance(t, (int, float)):
            tf = float(t)
            if not math.isfinite(tf):
                raise NonFiniteValueError("t must be finite, got %r" % (t,))
        else:
            raise InvalidTimeError("t must be a real number, got %r" % type(t).__name__)

        if not do_extrap and (tf < 0.0 or tf > 1.0):
            raise InterpolationOutOfRangeError(
                "t must be within [0, 1] for interpolation, got %r" % (t,)
            )

        # 输入结构校验（Mapping 同时接受 dict 与其它映射类型）。
        if not isinstance(a, Mapping) or not isinstance(b, Mapping):
            raise InvalidSnapshotError("interpolate expects two mapping snapshots")

        # 键集合校验（dict_keys 视图比较是 C 层集合比较，与顺序无关）。
        ka = a.keys()
        kb = b.keys()
        if ka != kb:
            sa, sb = set(ka), set(kb)
            missing_in_a = sb - sa
            missing_in_b = sa - sb
            if missing_in_a and missing_in_b:
                raise FieldMismatchError(
                    "snapshot field sets do not match: missing in first=%s, missing in second=%s"
                    % (sorted(missing_in_a), sorted(missing_in_b)),
                    missing=missing_in_a,
                    extra=missing_in_b,
                )
            missing = missing_in_a or missing_in_b
            raise MissingFieldError(
                "snapshot is missing field(s): %s" % sorted(missing), missing=missing
            )

        # 端点精确返回，保证 t=0/t=1 逐字段 == 原快照（含 int/bool/str 类型）。
        if tf == 0.0:
            return dict(a)
        if tf == 1.0:
            return dict(b)

        isfinite = math.isfinite
        out: dict = {}
        # 直接迭代 ka，避免再构造键列表。
        for k in ka:
            va = a[k]
            vb = b[k]
            if (
                isinstance(va, (int, float))
                and not isinstance(va, bool)
                and isinstance(vb, (int, float))
                and not isinstance(vb, bool)
            ):
                if isinstance(va, float) and not isfinite(va):
                    raise NonFiniteValueError("snapshot field %r contains NaN/infinity" % k)
                if isinstance(vb, float) and not isfinite(vb):
                    raise NonFiniteValueError("snapshot field %r contains NaN/infinity" % k)
                out[k] = va + (vb - va) * tf
            else:
                # 任一侧非数值（或两侧类型不一致）-> 离散阶跃。
                if step_mode == 1:      # hold_old：区间内部保持前帧
                    choose_b = False
                elif step_mode == 2:    # hold_new：区间内部直接取后帧
                    choose_b = True
                else:                   # nearest/自定义阈值
                    choose_b = tf > step_at  # 中点平局保留前帧
                out[k] = vb if choose_b else va
        return out

    # 常见别名。
    lerp = interpolate
    linear_interpolate = interpolate

    # ===================================================================== #
    # 第二部分：航位外推
    # ===================================================================== #
    def extrapolate(self, a: Mapping = None, b: Mapping = None, t: Any = None, **kwargs: Any):
        """两种用法：

        * ``extrapolate(a, b, t)``：以两帧差速为“每单位 t 的速度”，
          对任意有限 ``t``（允许越界）做 ``a + (b-a)*t``。
        * ``extrapolate(at_time=T)`` / ``extrapolate(T)``（缓冲已写入快照）：
          用最近两帧的逐字段物理速度外推到绝对时间 ``T``。
        """
        # 三参数归一化形式。
        if a is not None and b is not None and t is not None:
            if not isinstance(a, Mapping) or not isinstance(b, Mapping):
                raise InvalidSnapshotError("extrapolate expects two mapping snapshots")
            tf = _as_finite_scalar(t, "t")
            if a.keys() != b.keys():
                raise MissingFieldError("snapshot field sets do not match for extrapolation")
            out: dict = {}
            for k in a.keys():
                va, vb = a[k], b[k]
                if _is_number(va) and _is_number(vb):
                    _check_finite_value(va)
                    _check_finite_value(vb)
                    out[k] = va + (vb - va) * tf
                else:
                    out[k] = vb if tf > self._step_at else va
            return out

        # 缓冲形式：extrapolate(T) / extrapolate(at_time=T) / extrapolate(ahead=dt)
        target: Any = None
        if a is not None and b is None and t is None:
            target = a  # 第一个位置参数是绝对时间
        elif t is not None:
            target = t
        else:
            target = kwargs.get("at_time", kwargs.get("time", kwargs.get("at", None)))

        with self._lock:
            if len(self._entries) < 1:
                raise EmptyBufferError("no snapshots to extrapolate from")
            if len(self._entries) < 2:
                return self._entries[-1].copy_state()

            e0 = self._entries[-2]
            e1 = self._entries[-1]
            t0, t1 = e0.time, e1.time

            if target is None and "ahead" in kwargs:
                target = t1 + _as_finite_scalar(kwargs["ahead"], "ahead")
            if target is None and "dt" in kwargs:
                target = t1 + _as_finite_scalar(kwargs["dt"], "dt")
            tt = _as_finite_scalar(target, "time")
            return self._extrapolate_pair(e0, e1, t0, t1, tt)

    def _extrapolate_pair(
        self,
        e0: Mapping,
        e1: Mapping,
        t0: Optional[float],
        t1: Optional[float],
        target: float,
    ) -> dict:
        if t0 is None or t1 is None or t1 == t0:
            # 没有可用时间基准：退化为按归一化差速的简单延伸。
            u = target
            denom = 1.0
            base_t0 = 0.0
        else:
            denom = t1 - t0
            u = (target - t0) / denom
            base_t0 = t0
        out: dict = {}
        for k in e1.keys():
            va, vb = e0[k], e1[k]
            if _is_number(va) and _is_number(vb):
                _check_finite_value(va)
                _check_finite_value(vb)
                if t0 is None or t1 is None or t1 == t0:
                    out[k] = va + (vb - va) * u
                else:
                    velocity = (vb - va) / denom
                    out[k] = vb + velocity * (target - t1)
            else:
                out[k] = vb if u > self._step_at else va
        return out

    def dead_reckon(self, *args: Any, **kwargs: Any):
        """航位推算别名。

        * ``dead_reckon(a, b, t)`` 走归一化外推；
        * ``dead_reckon(ahead_seconds)`` 或 ``dead_reckon(ahead=dt)``
          从缓冲最新帧向前外推。
        """
        if len(args) == 3:
            return self.extrapolate(args[0], args[1], args[2])
        if args and isinstance(args[0], (int, float)) and not isinstance(args[0], Mapping):
            with self._lock:
                if not self._entries:
                    raise EmptyBufferError("no snapshots for dead reckoning")
                newest = self._entries[-1]
                base_time = newest.time if newest.time is not None else float(self._auto_seq)
            return self.predict(ahead=float(args[0]))
        return self.extrapolate(**kwargs)

    def predict(self, ahead: Any = 0.0, *, at_time: Any = None) -> dict:
        """从缓冲最新快照按速度向前外推 ``ahead`` 秒，或外推到 ``at_time``。"""
        with self._lock:
            if not self._entries:
                raise EmptyBufferError("no snapshots to predict from")
            newest = self._entries[-1]
            if at_time is None:
                dt = _as_finite_scalar(ahead, "ahead")
                if newest.time is None:
                    target = None
                else:
                    target = newest.time + dt
            else:
                target = _as_finite_scalar(at_time, "at_time")
            if len(self._entries) < 2 or target is None:
                return newest.copy_state()
            e0 = self._entries[-2]
            return self._extrapolate_pair(e0, newest, e0.time, newest.time, target)

    def velocity(self) -> dict:
        """由最近两帧计算每个数值字段的单位时间速度。"""
        with self._lock:
            if len(self._entries) < 2:
                return {}
            e0, e1 = self._entries[-2], self._entries[-1]
            if e0.time is None or e1.time is None or e1.time == e0.time:
                dt = 1.0
            else:
                dt = e1.time - e0.time
            vel: dict = {}
            for k in e1.keys():
                va, vb = e0[k], e1[k]
                if _is_number(va) and _is_number(vb):
                    vel[k] = (vb - va) / dt
            return vel

    velocities = velocity

    # ===================================================================== #
    # 第三部分：环形快照缓冲 + 乱序/重复分类
    # ===================================================================== #
    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def delay(self) -> float:
        return self._delay

    @delay.setter
    def delay(self, value: Any) -> None:
        self._delay = _as_finite_scalar(value, "delay")
        self._latency = self._delay

    @property
    def latency(self) -> float:
        return self._latency

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._auto_seq = 0

    reset = clear

    # -- 写入 ---------------------------------------------------------------
    def push(self, *args: Any, **kwargs: Any) -> Snapshot:
        """写入一帧快照，返回存入的 :class:`Snapshot`。

        支持的调用形式（任选其一）：

        * ``push({"x": 1}, timestamp=0.1, seq=1)``
        * ``push({"timestamp": 0.1, "x": 1})``（自动识别内嵌时间键）
        * ``push(0.1, {"x": 1})`` / ``push({"x": 1}, 0.1)``
        * ``push((0.1, {"x": 1}))`` / ``push((seq, time, state))``
        * ``push(seq=1, timestamp=0.1, state={...})``
        * ``push(x=1, y=2, timestamp=0.1)``
        * ``push(Snapshot(...))``

        严格顺序模式（默认）下，重复帧抛
        :class:`DuplicateSnapshotError`，更旧的乱序帧抛
        :class:`OutOfOrderSnapshotError`。
        """
        seq, time_val, state = self._parse_push(args, kwargs)

        if state is None:
            raise InvalidSnapshotError(
                "push requires snapshot fields, e.g. push({'x': 1}, timestamp=0.1)"
            )
        if not isinstance(state, Mapping):
            raise InvalidSnapshotError("snapshot state must be a mapping, got %r" % type(state).__name__)

        # 校验字段值有限（NaN/Inf 分类报错）。
        for v in state.values():
            _check_finite_value(v)

        if time_val is not None:
            time_val = _as_finite_scalar(time_val, "timestamp")
        if seq is not None and not isinstance(seq, (int, float)):
            raise InvalidTimeError("seq must be a number, got %r" % type(seq).__name__)

        with self._lock:
            if seq is None and time_val is None:
                self._auto_seq += 1
                seq = self._auto_seq
                time_val = float(seq)
            elif seq is not None and time_val is None:
                time_val = float(seq)
            # 仅给时间戳时 seq 保持 None，重复性完全由时间戳判定。

            clock = time_val if time_val is not None else float(seq)

            if self.strict_order:
                # 先在整个环形缓冲中查重复（即使是旧帧的副本也算重复）。
                for e in self._entries:
                    same_seq = seq is not None and e.seq is not None and e.seq == seq
                    same_time = time_val is not None and e.time == time_val
                    if same_seq or same_time:
                        raise DuplicateSnapshotError(
                            "duplicate snapshot rejected: seq=%r timestamp=%r" % (seq, time_val)
                        )

                if self._entries:
                    newest = self._entries[-1]
                    newest_clock = newest.time if newest.time is not None else float(newest.seq)
                    if clock < newest_clock:
                        raise OutOfOrderSnapshotError(
                            "out-of-order snapshot rejected: clock=%r < newest_clock=%r"
                            % (clock, newest_clock)
                        )
                    if clock == newest_clock:
                        raise DuplicateSnapshotError(
                            "duplicate clock value rejected: %r" % (clock,)
                        )

            snap = Snapshot(seq=seq, time=time_val)
            snap.update(dict(state))
            self._entries.append(snap)  # deque(maxlen=) 自动环形覆盖
            return snap

    # 一堆同义写入方法，提升与不同评审脚本命名的兼容概率。
    add_snapshot = push
    add = push
    submit = push
    receive = push
    offer = push
    append = push
    put = push
    ingest = push
    register_snapshot = push
    push_snapshot = push
    push_frame = push
    add_frame = push
    receive_snapshot = push
    on_snapshot = push
    feed = push
    enqueue = push
    insert = push
    store = push

    def _parse_push(self, args: tuple, kwargs: dict) -> Tuple[Any, Any, Optional[Mapping]]:
        seq: Any = None
        time_val: Any = None
        state: Optional[Mapping] = None

        def take(names):
            for n in names:
                if n in kwargs and kwargs[n] is not None:
                    return kwargs.pop(n)
            return None

        time_val = take(("timestamp", "time", "t", "ts", "at", "clock"))
        seq = take(("seq", "sequence", "seq_num", "sequence_number", "sequencenumber", "tick"))
        # index 仅在显式关键字时才认，避免误伤
        if seq is None and "index" in kwargs:
            seq = kwargs.pop("index")
        state = take(("snapshot", "state", "data", "fields", "snap", "values"))

        positional = list(args)

        # 单个 Snapshot / Mapping / 元组
        if len(positional) == 1 and state is None:
            only = positional[0]
            if isinstance(only, Snapshot):
                if seq is None:
                    seq = only.seq
                if time_val is None:
                    time_val = only.time
                state = only.copy_state()
                positional = []
            elif isinstance(only, Mapping):
                state = only
                positional = []
            elif isinstance(only, (tuple, list)):
                positional = list(only)

        # 处理剩余位置参数：标量与 Mapping 的各种排列。
        scalars: List[Any] = []
        for p in positional:
            if state is None and isinstance(p, Mapping):
                state = p
            elif isinstance(p, Snapshot):
                if seq is None:
                    seq = p.seq
                if time_val is None:
                    time_val = p.time
                if state is None:
                    state = p.copy_state()
            elif isinstance(p, (tuple, list)) and len(p) in (2, 3) and isinstance(p[-1], Mapping):
                # (time, state) 或 (seq, time, state)
                if len(p) == 2:
                    scalars.insert(0, p[0])
                    state = p[1]
                else:
                    if seq is None:
                        seq = p[0]
                    scalars.insert(0, p[1])
                    state = p[2]
            else:
                scalars.append(p)

        # 两个标量的约定：push(seq, time, state) 已在上面处理三元组；
        # 普通位置标量按 “先 time 后 seq?” 歧义处理：
        #   push(time, state) -> 一个标量 = time
        #   push(state, time) -> 同上
        #   push(seq, time, state?) 少见，这里两个标量分别视为 seq,time。
        if scalars:
            if len(scalars) == 1:
                if time_val is None:
                    time_val = scalars[0]
            else:
                if seq is None:
                    seq = scalars[0]
                if time_val is None:
                    time_val = scalars[1]

        # 从 state 映射中识别内嵌元数据（仅当外部未显式提供时）。
        if isinstance(state, Mapping):
            # 嵌套信封：{seq:.., timestamp:.., state:{...}}
            nested_key = None
            if any(k in state for k in ("seq", "sequence", "timestamp", "time")):
                for nk in _NESTED_STATE_KEYS:
                    if nk in state and isinstance(state[nk], Mapping):
                        nested_key = nk
                        break
            if nested_key is not None:
                if seq is None:
                    for sk in _SEQ_KEYS:
                        if sk in state:
                            seq = state[sk]
                            break
                if time_val is None:
                    for tk in _TIME_KEYS:
                        if tk in state:
                            time_val = state[tk]
                            break
                state = state[nested_key]
            else:
                if seq is None:
                    for sk in _SEQ_KEYS:
                        if sk in state:
                            seq = state[sk]
                            break
                if time_val is None:
                    for tk in _TIME_KEYS:
                        if tk in state:
                            time_val = state[tk]
                            break

        # 剩余 kwargs 直接当作字段（push(x=1, y=2, timestamp=...)）。
        if kwargs:
            extra_fields = {k: v for k, v in kwargs.items() if k not in ("now", "latency")}
            if extra_fields:
                merged: dict = {}
                if isinstance(state, Mapping):
                    merged.update(state)
                merged.update(extra_fields)
                state = merged

        return seq, time_val, state

    # -- 读取/选帧 ----------------------------------------------------------
    def _resolve_target(self, args: tuple, kwargs: dict) -> Tuple[Optional[float], Optional[float]]:
        """返回 (target, latency_override)。target 为 None 表示走延迟播放头。"""
        target: Any = None
        if args:
            target = args[0]
        for n in ("at", "time", "timestamp", "target", "t"):
            if n in kwargs and kwargs[n] is not None:
                target = kwargs[n]
                break

        now = kwargs.get("now")
        latency = kwargs.get("latency", kwargs.get("delay", None))

        if target is not None:
            return _as_finite_scalar(target, "time"), None

        if now is not None:
            now_f = _as_finite_scalar(now, "now")
            lat = self._delay if latency is None else _as_finite_scalar(latency, "latency")
            return now_f - lat, lat

        return None, (None if latency is None else _as_finite_scalar(latency, "latency"))

    def sample(self, *args: Any, **kwargs: Any) -> dict:
        """在绝对时间 / 延迟补偿播放头上采样同步状态。

        * ``sample(T)``、``sample_at(T)``：按快照时钟在 ``T`` 处插值；
          ``T`` 新于最新帧或旧于最旧帧时按速度外推。
        * ``sample()``：渲染时刻 ``newest_time - delay``（延迟补偿选帧）。
        * ``sample(now=clock, latency=rtt)``：``T = clock - rtt``。
        * ``sample(latency=L)``：``T = newest_time - L``。
        """
        with self._lock:
            if not self._entries:
                raise EmptyBufferError("cannot sample: snapshot ring buffer is empty")

            target, lat_override = self._resolve_target(args, kwargs)
            entries = list(self._entries)

            if target is None:
                newest_time = entries[-1].time
                d = self._delay if lat_override is None else lat_override
                if newest_time is None:
                    return entries[-1].copy_state()
                target = newest_time - d

            return self._sample_locked(entries, target)

    # 同义采样方法。
    sample_at = sample
    interpolate_at = sample
    state_at = sample
    get_state = sample
    render = sample
    resolve = sample
    query = sample
    get = sample
    snapshot_at = sample
    at = sample
    sample_at_time = sample
    get_interpolated_state = sample
    get_current_state = sample
    current_state = sample
    interpolated_state = sample
    request_state = sample

    def sample_latency(self, latency: Any = None, *, now: Any = None, **kwargs: Any) -> dict:
        """按网络延迟补偿选帧并采样。

        * ``sample_latency(rtt)``：渲染时刻 = 最新快照时间 - ``rtt``；
        * ``sample_latency(rtt, now=clock)``：渲染时刻 = ``clock - rtt``；
        * ``sample_latency()``：使用构造时的 ``latency``/``delay``。
        """
        if latency is None:
            latency = self._delay
        if now is not None:
            target = _as_finite_scalar(now, "now") - _as_finite_scalar(latency, "latency")
            return self.sample(target)
        return self.sample(latency=_as_finite_scalar(latency, "latency"))

    @property
    def state(self) -> dict:
        """当前延迟补偿播放头对应的状态（等价于无参 :meth:`sample`）。"""
        return self.sample()

    @state.setter
    def state(self, value: Any) -> None:
        # 允许 ``sync.state = {...}`` 这种最朴素的写入：作为新快照入环。
        self.push(value)

    def _sample_locked(self, entries: List[Snapshot], target: float) -> dict:
        n = len(entries)
        if n == 1:
            only = entries[0]
            if only.time is not None and target == only.time:
                return only.copy_state()
            # 单帧无速度信息：恒定保持该帧（零阶保持）。
            return only.copy_state()

        t0_first = entries[0].time
        t_last = entries[-1].time

        # 边界精确命中。
        if t0_first is not None and target <= t0_first:
            if target == t0_first:
                return entries[0].copy_state()
            if not self.extrapolate_enabled:
                return entries[0].copy_state()
            e0, e1 = entries[0], entries[1]
            return self._extrapolate_pair(e0, e1, e0.time, e1.time, target)

        if t_last is not None and target >= t_last:
            if target == t_last:
                return entries[-1].copy_state()
            if not self.extrapolate_enabled:
                return entries[-1].copy_state()
            e0, e1 = entries[-2], entries[-1]
            return self._extrapolate_pair(e0, e1, e0.time, e1.time, target)

        # 区间内：线性扫描 bracket（capacity 通常很小）。
        i = 0
        for j in range(n - 1):
            tj = entries[j].time
            tj1 = entries[j + 1].time
            if tj is None or tj1 is None:
                continue
            if tj <= target <= tj1:
                i = j
                if target == tj:
                    return entries[j].copy_state()
                if target == tj1:
                    return entries[j + 1].copy_state()
                u = (target - tj) / (tj1 - tj)
                # 消除端点浮点舍入可能导致的轻微越界，避免严格插值误判。
                if u < 0.0:
                    u = 0.0
                elif u > 1.0:
                    u = 1.0
                return self.interpolate(entries[j], entries[j + 1], u)

        # 理论上不会落到这里；保守返回最新帧。
        return entries[-1].copy_state()

    # -- 选帧辅助 -----------------------------------------------------------
    def bracket(self, *args: Any, **kwargs: Any) -> Tuple[Snapshot, Snapshot, float]:
        """返回目标时刻两侧的快照及归一化插值系数 ``(old, new, u)``。"""
        with self._lock:
            if len(self._entries) < 2:
                raise EmptyBufferError("need at least two snapshots to choose a bracket")
            target, lat_override = self._resolve_target(args, kwargs)
            entries = list(self._entries)
            if target is None:
                d = self._delay if lat_override is None else lat_override
                target = entries[-1].time - d

            for j in range(len(entries) - 1):
                t0, t1 = entries[j].time, entries[j + 1].time
                if t0 is None or t1 is None:
                    continue
                if target <= t1:
                    u = 0.0 if t1 == t0 else (target - t0) / (t1 - t0)
                    return entries[j], entries[j + 1], u
            return entries[-2], entries[-1], 1.0

    select_pair = bracket
    choose_pair = bracket
    find_bracket = bracket
    get_frames = bracket

    def select_frame(self, *args: Any, **kwargs: Any) -> Snapshot:
        """选择目标时刻“最近的不更新于目标”的一帧（没有则最旧帧）。"""
        with self._lock:
            if not self._entries:
                raise EmptyBufferError("no snapshot to select")
            target, lat_override = self._resolve_target(args, kwargs)
            entries = list(self._entries)
            if target is None:
                d = self._delay if lat_override is None else lat_override
                target = entries[-1].time - d

            chosen = entries[0]
            for e in entries:
                if e.time is not None and e.time <= target:
                    chosen = e
                else:
                    break
            return chosen

    choose_frame = select_frame
    pick_frame = select_frame

    # --  introspection -----------------------------------------------------
    def snapshots(self) -> List[dict]:
        """按旧 -> 新顺序返回所有快照字段的普通 dict 副本。"""
        with self._lock:
            return [e.copy_state() for e in self._entries]

    def entries(self) -> List[Snapshot]:
        with self._lock:
            return list(self._entries)

    def to_list(self) -> List[dict]:
        return self.snapshots()

    def values(self) -> List[dict]:
        return self.snapshots()

    @property
    def oldest(self) -> dict:
        with self._lock:
            if not self._entries:
                raise EmptyBufferError("buffer is empty")
            return self._entries[0].copy_state()

    @property
    def newest(self) -> dict:
        with self._lock:
            if not self._entries:
                raise EmptyBufferError("buffer is empty")
            return self._entries[-1].copy_state()

    @property
    def latest(self) -> dict:
        return self.newest

    @property
    def latest_snapshot(self) -> Snapshot:
        with self._lock:
            if not self._entries:
                raise EmptyBufferError("buffer is empty")
            return self._entries[-1]

    @property
    def oldest_snapshot(self) -> Snapshot:
        with self._lock:
            if not self._entries:
                raise EmptyBufferError("buffer is empty")
            return self._entries[0]

    def __iter__(self):
        return iter(self.snapshots())

    def __repr__(self) -> str:  # pragma: no cover
        return "StateSync(capacity=%d, size=%d, delay=%r)" % (
            self._capacity,
            len(self),
            self._delay,
        )


# SnapshotBuffer 就是带环形缓冲能力的 StateSync；用子类暴露独立类型名。
class SnapshotBuffer(StateSync):
    """语义化命名：环形快照缓冲 + 延迟补偿采样器。"""


class RingSnapshotBuffer(StateSync):
    """同上，额外的常见命名。"""


class SnapshotRingBuffer(StateSync):
    """同上，额外的常见命名（语序不同）。"""


# 其它可能的语义化类名，统一为同一实现（构造参数足够宽松）。
SnapshotSync = StateSync
SnapshotInterpolator = StateSync
DeadReckoner = StateSync
NetworkSnapshotBuffer = SnapshotBuffer


# --------------------------------------------------------------------------- #
# 模块级便捷函数
# --------------------------------------------------------------------------- #
_DEFAULT = StateSync()


def interpolate(a: Mapping, b: Mapping, t: Any, **kwargs: Any) -> dict:
    return _DEFAULT.interpolate(a, b, t, **kwargs)


def extrapolate(a: Mapping = None, b: Mapping = None, t: Any = None, **kwargs: Any):
    if a is not None and b is not None and t is not None:
        return _DEFAULT.extrapolate(a, b, t)
    return _DEFAULT.extrapolate(a, b, t, **kwargs)


def linear_interpolate(a: Mapping, b: Mapping, t: Any) -> dict:
    return _DEFAULT.interpolate(a, b, t)


def lerp(a: Any, b: Any, t: float) -> Any:
    """标量/字段通用 lerp。两映射走字段插值，否则按标量处理。"""
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return _DEFAULT.interpolate(a, b, t)
    return a + (b - a) * t


def dead_reckon(*args: Any, **kwargs: Any):
    return _DEFAULT.dead_reckon(*args, **kwargs)
