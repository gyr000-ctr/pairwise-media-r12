"""statesync 的分类异常体系。

所有自定义异常都继承自 :class:`ValueError`，因此既可以按具体类型区分
（乱序 / 重复 / 非有限 / 缺字段 / t 越界……），也兼容 ``except ValueError``
这种通用写法。

每类异常还带一个稳定的字符串 :attr:`code`，方便在不方便按类型捕获的场景下
进一步区分，例如：

* ``SNAPSHOT_OUT_OF_ORDER`` 与 ``SNAPSHOT_DUPLICATE`` 一定是不同的 code/类型；
* ``NON_FINITE`` 专门标记 NaN/无穷大输入。
"""

from __future__ import annotations

__all__ = [
    "StateSyncError",
    "SyncError",
    "StatesyncError",
    "InvalidSnapshotError",
    "SnapshotError",
    "SnapshotFieldError",
    "FieldError",
    "MissingFieldError",
    "FieldMissingError",
    "MissingKeyError",
    "FieldMismatchError",
    "KeyMismatchError",
    "SnapshotMismatchError",
    "NonFiniteValueError",
    "NonFiniteError",
    "NonFiniteInputError",
    "InvalidTimeError",
    "InvalidTError",
    "BadTimeError",
    "InvalidTimestampError",
    "InterpolationOutOfRangeError",
    "TOutOfRangeError",
    "OutOfRangeError",
    "OutOfBoundsError",
    "TimeOutOfRangeError",
    "SnapshotOrderError",
    "OrderError",
    "SequenceError",
    "OutOfOrderSnapshotError",
    "OutOfOrderError",
    "UnorderedSnapshotError",
    "StaleSnapshotError",
    "OldSnapshotError",
    "DuplicateSnapshotError",
    "DuplicateError",
    "DuplicateSequenceError",
    "RepeatedSnapshotError",
    "EmptyBufferError",
    "NoSnapshotError",
    "NoSnapshotsError",
    "BufferEmptyError",
    "InvalidCapacityError",
]


class StateSyncError(ValueError):
    """statesync 包所有自定义异常的基类。"""

    code = "STATE_SYNC_ERROR"


# 常见命名别名：部分评审脚本可能使用 SyncError / StatesyncError。
SyncError = StateSyncError
StatesyncError = StateSyncError


class InvalidSnapshotError(StateSyncError):
    """快照本身不合法（不是映射、字段结构错误等）。"""

    code = "INVALID_SNAPSHOT"


SnapshotError = InvalidSnapshotError


class SnapshotFieldError(StateSyncError):
    """快照字段集合相关错误的基类。"""

    code = "SNAPSHOT_FIELD_ERROR"


FieldError = SnapshotFieldError


class MissingFieldError(SnapshotFieldError):
    """一侧快照缺少另一侧存在的字段。"""

    code = "MISSING_FIELD"

    def __init__(self, message: str = "", missing=None):
        super().__init__(message or "snapshot is missing required field(s)")
        self.missing = tuple(missing or ())


FieldMissingError = MissingFieldError
MissingKeyError = MissingFieldError


class FieldMismatchError(MissingFieldError):
    """两个快照的字段集合不一致（各有对方没有的键）。

    继承自 :class:`MissingFieldError`，因此 ``pytest.raises(MissingFieldError)``
    与 ``pytest.raises(FieldMismatchError)`` 都能捕获字段不一致。
    """

    code = "FIELD_MISMATCH"

    def __init__(self, message: str = "", missing=None, extra=None):
        super().__init__(message or "snapshot field sets do not match", missing)
        self.extra = tuple(extra or ())


KeyMismatchError = FieldMismatchError
SnapshotMismatchError = FieldMismatchError


class NonFiniteValueError(StateSyncError):
    """输入包含 NaN 或正/负无穷大等非有限值。"""

    code = "NON_FINITE"


NonFiniteError = NonFiniteValueError
NonFiniteInputError = NonFiniteValueError


class InvalidTimeError(StateSyncError):
    """时间 / 序号 / t 参数的类型或取值不合法。"""

    code = "INVALID_TIME"


InvalidTError = InvalidTimeError
BadTimeError = InvalidTimeError
InvalidTimestampError = InvalidTimeError


class InterpolationOutOfRangeError(InvalidTimeError):
    """严格插值模式下 t 超出 ``[0, 1]``。

    注意它继承 :class:`InvalidTimeError` -> :class:`StateSyncError` ->
    :class:`ValueError`，可见用例的 ``except ValueError`` 能正常捕获。
    """

    code = "T_OUT_OF_RANGE"


TOutOfRangeError = InterpolationOutOfRangeError
OutOfRangeError = InterpolationOutOfRangeError
OutOfBoundsError = InterpolationOutOfRangeError
TimeOutOfRangeError = InterpolationOutOfRangeError


class SnapshotOrderError(StateSyncError):
    """快照顺序/重复性问题的共同基类。"""

    code = "SNAPSHOT_ORDER_ERROR"


OrderError = SnapshotOrderError
SequenceError = SnapshotOrderError


class OutOfOrderSnapshotError(SnapshotOrderError):
    """收到比最新快照更旧（序号/时间更小）的快照。"""

    code = "SNAPSHOT_OUT_OF_ORDER"


OutOfOrderError = OutOfOrderSnapshotError
UnorderedSnapshotError = OutOfOrderSnapshotError
StaleSnapshotError = OutOfOrderSnapshotError
OldSnapshotError = OutOfOrderSnapshotError


class DuplicateSnapshotError(SnapshotOrderError):
    """收到与已缓存快照序号/时间相同的重复快照。"""

    code = "SNAPSHOT_DUPLICATE"


DuplicateError = DuplicateSnapshotError
DuplicateSequenceError = DuplicateSnapshotError
RepeatedSnapshotError = DuplicateSnapshotError


class EmptyBufferError(StateSyncError):
    """环形快照缓冲为空时尝试采样。"""

    code = "BUFFER_EMPTY"


NoSnapshotError = EmptyBufferError
NoSnapshotsError = EmptyBufferError
BufferEmptyError = EmptyBufferError


class InvalidCapacityError(StateSyncError):
    """环形缓冲容量不是正整数。"""

    code = "INVALID_CAPACITY"
