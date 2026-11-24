"""gamma 核心 API。

保留题目最初的简单幂 gamma / LUT 接口；传递函数与 HDR 曲线见
:mod:`gamma.transfer`。
"""

import math

import numpy as np

from .errors import GammaError, NonFiniteError


class Gamma:
    """幂 gamma 变换与查找表（LUT）。

    ``apply`` 对 [0, 1] 信号做 out = in ** gamma；
    ``lut`` 生成 ``levels`` 个等间距码值对应的查找表，
    LUT 与解析式逐值一致。
    """

    def apply(self, image, gamma):
        """对嵌套序列图像应用幂 gamma，返回同形状的 list。"""
        if not isinstance(gamma, (int, float, np.floating, np.integer)) or isinstance(
            gamma, bool
        ):
            raise GammaError("gamma 必须为数值")
        gamma = float(gamma)
        if not math.isfinite(gamma) or gamma <= 0.0:
            raise GammaError("gamma 必须为正的有限数")
        if not isinstance(image, (list, tuple, np.ndarray)) or len(image) == 0:
            raise GammaError("image 必须为非空序列")

        arr = np.asarray(image, dtype=np.float64)
        if arr.ndim == 0 or arr.size == 0:
            raise GammaError("image 必须为非空序列")
        if not np.all(np.isfinite(arr)):
            raise NonFiniteError("image 包含 NaN 或无穷大等非有限值")
        if np.any(arr < 0.0) or np.any(arr > 1.0):
            from .errors import LuminanceRangeError

            raise LuminanceRangeError("image 像素必须位于 [0, 1]")

        out = np.power(arr, gamma)
        if gamma > 0.0:
            out = np.where(arr == 0.0, 0.0, out)
            out = np.where(arr == 1.0, 1.0, out)
        return out.tolist()

    def lut(self, gamma, levels):
        """生成 ``levels`` 项的查找表，第 i 项 = (i/(levels-1)) ** gamma。"""
        if not isinstance(gamma, (int, float, np.floating, np.integer)) or isinstance(
            gamma, bool
        ):
            raise GammaError("gamma 必须为数值")
        gamma = float(gamma)
        if not math.isfinite(gamma) or gamma <= 0.0:
            raise GammaError("gamma 必须为正的有限数")
        if not isinstance(levels, (int, np.integer)) or isinstance(levels, bool):
            raise GammaError("levels 必须为不小于 2 的整数")
        if levels < 2:
            raise GammaError("levels 必须为不小于 2 的整数")

        idx = np.arange(levels, dtype=np.float64)
        table = np.power(idx / (levels - 1), gamma)
        table[0] = 0.0
        table[-1] = 1.0
        return table.tolist()
