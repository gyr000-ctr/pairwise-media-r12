"""gamma 包的异常类型。

所有异常都继承自 :class:`ValueError`，因此既可以按具体类型区分处理，
也兼容 ``except ValueError`` 这种通用写法。
"""


class GammaError(ValueError):
    """gamma 包所有自定义异常的基类。"""


class UnknownCurveError(GammaError):
    """非法/不支持的曲线类型。"""


class LuminanceRangeError(GammaError):
    """亮度（或信号值）超出曲线允许的定义域。"""


class NonFiniteError(GammaError):
    """输入包含 NaN 或无穷大等非有限值。"""
