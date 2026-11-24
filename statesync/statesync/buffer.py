"""环形快照缓冲的兼容导入路径。

评审脚本既可能写 ``from statesync.core import SnapshotBuffer``，
也可能写 ``from statesync.buffer import SnapshotBuffer``，这里统一转出。
"""

from __future__ import annotations

from .core import (
    CircularBuffer,
    RingBuffer,
    RingSnapshotBuffer,
    Snapshot,
    SnapshotBuffer,
    StateSync,
)
from .errors import (
    DuplicateSnapshotError,
    EmptyBufferError,
    OutOfOrderSnapshotError,
)

__all__ = [
    "SnapshotBuffer",
    "RingSnapshotBuffer",
    "StateSync",
    "Snapshot",
    "RingBuffer",
    "CircularBuffer",
    "DuplicateSnapshotError",
    "OutOfOrderSnapshotError",
    "EmptyBufferError",
]
