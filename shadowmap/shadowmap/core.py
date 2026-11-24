"""深度偏移与 PCF 阴影核心 API。

判据（偏移消除自阴影）：
    对每个 PCF 采样纹素，当 ``shadow_depth + bias >= scene_depth`` 时判为受光。
    因此同一深度处（shadow_depth == scene_depth）恒受光，不会产生自阴影。

PCF：
    采用 Halton(2, 3) 低差异序列映射到单位圆盘（r = sqrt(h2), θ = 2π·h3）。
    采样点列与采样总数无关——任意 N 都是同一条无穷点列的前缀——因此序列是
    嵌套的，估计随采样数增加按 QMC 方式趋于收敛，且结果完全确定、可复现，
    不依赖随机数生成器状态（并发安全）。

可见性取受光采样数的整数平均，故全遮挡恰为 ``0.0``、全受光恰为 ``1.0``。

本类实例构造后不可变，所有方法均为纯函数式计算，可被多线程并发调用。
"""

import math

__all__ = ["ShadowMap"]


def _halton(index, base):
    """Halton 序列第 index 项（index 从 1 起），结果落在 (0, 1)。"""
    result = 0.0
    scale = 1.0
    i = index
    while i > 0:
        scale /= base
        result += scale * (i % base)
        i //= base
    return result


def _is_real_number(value):
    """布尔值不视为合法数值参数；其余 int/float（含非有限值）交由 isfinite 判。"""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class ShadowMap:
    """带深度偏移的 PCF 软阴影查询器。

    Parameters
    ----------
    bias:
        深度偏移量，必须为非负有限数。场景深度不超过
        ``阴影深度 + bias`` 即判受光。
    samples:
        PCF 采样数，必须为正整数，默认 8。
    """

    def __init__(self, bias, samples=8):
        if not _is_real_number(bias) or not math.isfinite(bias):
            raise ValueError("bias 必须为有限数值")
        if bias < 0.0:
            raise ValueError("bias 必须非负，得到 %r" % (bias,))
        if isinstance(samples, bool) or not isinstance(samples, int):
            raise ValueError("samples 必须为正整数，得到 %r" % (samples,))
        if samples <= 0:
            raise ValueError("samples 必须为正整数，得到 %r" % (samples,))

        self.bias = float(bias)
        self.samples = samples

        # Halton(2, 3) 低差异盘序列：与 N 无关的无穷点列，取前 samples 项，
        # 故不同采样数实例的偏移天然嵌套，采样核半径 = 1 纹素，构造后只读复用。
        offsets = []
        for i in range(1, samples + 1):
            h2 = _halton(i, 2)
            h3 = _halton(i, 3)
            radius = math.sqrt(h2)
            theta = 2.0 * math.pi * h3
            offsets.append((radius * math.cos(theta), radius * math.sin(theta)))
        self._offsets = tuple(offsets)

    def visibility(self, depth_map, uv, depth):
        """返回 ``uv`` 处的可见性（受光比例），取值区间 [0.0, 1.0]。

        Parameters
        ----------
        depth_map:
            二维深度图，按行排列的序列（list/tuple 均可），各行等长且非空，
            元素为有限数值。
        uv:
            归一化坐标 ``(u, v)``，两分量均为有限数且落在 ``[0.0, 1.0)``。
        depth:
            待查询点的场景深度，必须为有限数值。

        Returns
        -------
        float
            受光采样数 / 总采样数；全遮挡为 ``0.0``，全受光为 ``1.0``。

        Raises
        ------
        ValueError
            任一入参非法（空/异形深度图、坐标越界、非有限数值等）。
        """
        try:
            height = len(depth_map)
        except TypeError:
            raise ValueError("depth_map 必须为二维序列") from None
        if height == 0:
            raise ValueError("depth_map 不能为空")

        width = None
        for row in depth_map:
            try:
                row_len = len(row)
            except TypeError:
                raise ValueError("depth_map 必须为二维序列") from None
            if row_len == 0:
                raise ValueError("depth_map 不能含空行")
            if width is None:
                width = row_len
            elif row_len != width:
                raise ValueError("depth_map 各行必须等长")
            head = row[0]
            if not _is_real_number(head) or not math.isfinite(head):
                raise ValueError("depth_map 元素必须为有限数值")

        try:
            u, v = uv
        except (TypeError, ValueError):
            raise ValueError("uv 必须为含两个分量的序列") from None
        if not (_is_real_number(u) and _is_real_number(v)) or not (
            math.isfinite(u) and math.isfinite(v)
        ):
            raise ValueError("uv 分量必须为有限数值")
        if not (0.0 <= u < 1.0 and 0.0 <= v < 1.0):
            raise ValueError("uv 必须落在 [0.0, 1.0)，得到 %r" % ((u, v),))

        if not _is_real_number(depth) or not math.isfinite(depth):
            raise ValueError("depth 必须为有限数值")

        # 纹素中心坐标：uv 0 -> 中心 0，uv 1 -> 中心 W（故减 0.5）。
        center_x = u * width - 0.5
        center_y = v * height - 0.5
        max_x = width - 1
        max_y = height - 1
        bias = self.bias

        lit = 0
        # 热路径：全部使用局部变量、无函数调用式索引，保证纯 Python 下的性能。
        for ox, oy in self._offsets:
            ix = int(center_x + ox)
            iy = int(center_y + oy)
            if ix < 0:
                ix = 0
            elif ix > max_x:
                ix = max_x
            if iy < 0:
                iy = 0
            elif iy > max_y:
                iy = max_y

            shadow_depth = depth_map[iy][ix]
            if not _is_real_number(shadow_depth) or not math.isfinite(shadow_depth):
                raise ValueError("depth_map 元素必须为有限数值")
            # 偏移判据：同深度处 +bias 后恒满足，自阴影被消除。
            if shadow_depth + bias >= depth:
                lit += 1

        return lit / self.samples
