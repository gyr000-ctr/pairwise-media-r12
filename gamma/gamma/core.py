"""gamma 核心 API（初始未实现）。"""


class Gamma:
    def apply(self, image, gamma):
        """应用"""
        raise NotImplementedError

    def lut(self, gamma, levels):
        """查找表"""
        raise NotImplementedError
