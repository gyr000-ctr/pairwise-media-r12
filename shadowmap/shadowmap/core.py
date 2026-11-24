"""shadowmap 核心 API（初始未实现）。"""


class ShadowMap:
    def __init__(self, bias, samples=8):
        """构造"""
        raise NotImplementedError

    def visibility(self, depth_map, uv, depth):
        """可见性"""
        raise NotImplementedError
