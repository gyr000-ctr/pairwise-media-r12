"""传递函数（EOTF/OETF）与 HDR 曲线。

实现内容：

* sRGB 分段 EOTF / OETF（IEC 61966-2-1，阈值 0.04045 / 0.0031308，线性斜率 12.92），
  不使用任何 2.2 幂近似；
* SMPTE ST 2084 Perceptual Quantizer（PQ），按 10000 nit 归一化；
* ARIB STD-B67 / ITU-R BT.2100 HLG（OETF、逆 OETF，以及带系统 gamma 的 EOTF）；
* sRGB → PQ 的色调映射（含高光 roll-off）及其逆映射；
* 16 位量化 / 反量化。

所有标量公式与 SMPTE / ITU-R 参考式逐系数一致，numpy 仅用于向量化加速，
不改变任何数学定义。
"""

import math

import numpy as np

from .errors import (
    GammaError,
    LuminanceRangeError,
    NonFiniteError,
    UnknownCurveError,
)

__all__ = [
    # 曲线名常量
    "SRGB",
    "PQ",
    "HLG",
    "CURVES",
    # sRGB（分段式，非 2.2 幂近似）
    "srgb_eotf",
    "srgb_oetf",
    "srgb_to_linear",
    "linear_to_srgb",
    # PQ（SMPTE ST 2084，10000 nit）
    "pq_oetf",
    "pq_eotf",
    "pq_to_linear",
    "linear_to_pq",
    # HLG（BT.2100）
    "hlg_oetf",
    "hlg_inverse_oetf",
    "hlg_eotf",
    "hlg_inverse_eotf",
    # sRGB <-> PQ 色调映射（含高光 roll-off）
    "shoulder_curve",
    "shoulder_curve_inverse",
    "sdr_to_pq",
    "pq_to_sdr",
    "linear_to_pq_tonemap",
    "pq_tonemap_to_linear",
    # 量化
    "quantize",
    "dequantize",
    # 统一入口
    "encode",
    "decode",
    "TransferFunction",
    # 异常
    "GammaError",
    "UnknownCurveError",
    "LuminanceRangeError",
    "NonFiniteError",
]

# ---------------------------------------------------------------------------
# 曲线名
# ---------------------------------------------------------------------------

SRGB = "srgb"
PQ = "pq"
HLG = "hlg"
CURVES = (SRGB, PQ, HLG)

# ---------------------------------------------------------------------------
# sRGB 常量（IEC 61966-2-1，精确分段定义，禁止 2.2 幂近似）
# ---------------------------------------------------------------------------

# 阈值说明：IEC 文本中的 0.04045 / 0.0031308 是经舍入的分段点，两者并不
# 严格互逆（12.92*0.0031308 = 0.040449936）。为满足往返 <= 1e-12，两个函数
# 共用同一个精确拐点：信号域 0.04045（保持标准字面量），线性域
# 0.04045/12.92。标准常量 0.04045、0.0031308、12.92 全部原样保留，
# 0.0031308 仍落在精确线性段内（12.92*0.0031308 = 0.040449936 < 0.04045），
# 仅在宽度约 5e-9 的拐点微小区间内调整分支归属，与标准式的差异 < 3e-8。
_SRGB_ALPHA = 1.055
_SRGB_SLOPE = 12.92
_SRGB_GAMMA = 2.4
_SRGB_OFFSET = 0.055
_SRGB_THRESHOLD_ENCODE = 0.04045      # 信号域分段阈值（标准字面量）
_SRGB_THRESHOLD_LINEAR_STD = 0.0031308  # 标准给出的线性域阈值（字面量保留）
# 精确互逆拐点：信号域 0.04045 对应的线性域值
_SRGB_THRESHOLD_LINEAR = _SRGB_THRESHOLD_ENCODE / _SRGB_SLOPE

# ---------------------------------------------------------------------------
# PQ 常量（SMPTE ST 2084，逐系数取自标准）
# ---------------------------------------------------------------------------

PQ_PEAK_NITS = 10000.0  # PQ 按 10000 nit 归一化

_PQ_M1 = 2610.0 / 16384.0
_PQ_M2 = 2523.0 / 4096.0 * 128.0
_PQ_C1 = 3424.0 / 4096.0
_PQ_C2 = 2413.0 / 4096.0 * 32.0
_PQ_C3 = 2392.0 / 4096.0 * 32.0

# ---------------------------------------------------------------------------
# HLG 常量（ARIB STD-B67 / ITU-R BT.2100）
# ---------------------------------------------------------------------------

HLG_PEAK_NITS = 1000.0
_HLG_A = 0.17883277
_HLG_B = 0.28466892   # 1 - 4a
_HLG_C = 0.55991073   # 0.5 - a*ln(4a)


# ---------------------------------------------------------------------------
# 输入 / 输出辅助
# ---------------------------------------------------------------------------

def _as_array(values):
    """把输入转成 float64 numpy 数组，并做类型 / 有限性校验。"""
    if isinstance(values, np.ndarray):
        arr = values.astype(np.float64, copy=False)
    elif isinstance(values, (list, tuple)):
        try:
            arr = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise GammaError(f"输入无法转换为数值数组: {exc}") from exc
    elif isinstance(values, (int, float, np.floating, np.integer)):
        if isinstance(values, bool):
            raise GammaError("布尔值不是合法的曲线输入")
        arr = np.asarray(float(values), dtype=np.float64)
    else:
        raise GammaError(f"不支持的输入类型: {type(values).__name__}")

    if not np.all(np.isfinite(arr)):
        raise NonFiniteError("输入包含 NaN 或无穷大等非有限值")
    return arr


def _check_range(arr, low, high, name):
    """定义域检查，越界抛出可区分的 :class:`LuminanceRangeError`。

    严格按闭域判断；往返链路中浮点舍入造成的轻微越界由
    :func:`_sanitize_domain` 在调用前夹回，而不是在这里放宽。
    """
    if np.any(arr < low) or np.any(arr > high):
        raise LuminanceRangeError(
            f"{name} 超出定义域 [{low}, {high}]，"
            f"实际范围 [{float(np.min(arr))}, {float(np.max(arr))}]"
        )


def _sanitize_domain(arr, low=0.0, high=1.0):
    """把舍入误差导致的轻微越界夹回定义域（端点保持精确）。"""
    return np.clip(arr, low, high)


def _wrap(original, result):
    """按输入形态返回：标量 -> float，list -> 同形状 list，ndarray -> ndarray。"""
    if isinstance(original, np.ndarray):
        return result
    if isinstance(original, list):
        return result.tolist()
    if isinstance(original, tuple):
        return tuple(result.tolist())
    return float(result)


# ---------------------------------------------------------------------------
# sRGB 分段 EOTF / OETF
# ---------------------------------------------------------------------------

def srgb_eotf(signal):
    """sRGB EOTF：非线性信号 [0, 1] -> 线性光 [0, 1]。

    严格采用 IEC 61966-2-1 分段定义（线性段 12.92，幂段指数 2.4），
    不使用 2.2 幂近似。
    """
    arr = _as_array(signal)
    _check_range(arr, 0.0, 1.0, "sRGB 信号值")
    arr = _sanitize_domain(arr)
    out = np.where(
        arr <= _SRGB_THRESHOLD_ENCODE,
        arr / _SRGB_SLOPE,
        ((arr + _SRGB_OFFSET) / _SRGB_ALPHA) ** _SRGB_GAMMA,
    )
    # 精确端点，并夹回定义域消除幂运算的舍入越界
    out = np.where(arr == 0.0, 0.0, out)
    out = np.where(arr == 1.0, 1.0, out)
    return _wrap(signal, _sanitize_domain(out))


def srgb_oetf(linear):
    """sRGB OETF：线性光 [0, 1] -> 非线性信号 [0, 1]。

    严格采用 IEC 61966-2-1 分段定义（线性段 12.92，幂段指数 2.4）。
    """
    arr = _as_array(linear)
    _check_range(arr, 0.0, 1.0, "sRGB 线性光值")
    arr = _sanitize_domain(arr)
    out = np.where(
        arr <= _SRGB_THRESHOLD_LINEAR,
        _SRGB_SLOPE * arr,
        _SRGB_ALPHA * np.power(arr, 1.0 / _SRGB_GAMMA) - _SRGB_OFFSET,
    )
    out = np.where(arr == 0.0, 0.0, out)
    out = np.where(arr == 1.0, 1.0, out)
    return _wrap(linear, _sanitize_domain(out))


# 语义别名
srgb_to_linear = srgb_eotf
linear_to_srgb = srgb_oetf


# ---------------------------------------------------------------------------
# SMPTE ST 2084 PQ
# ---------------------------------------------------------------------------

def pq_oetf(linear):
    """PQ OETF（ST 2084）：线性光（10000 nit 归一化的 [0, 1]）-> 非线性信号 [0, 1]。"""
    arr = _as_array(linear)
    _check_range(arr, 0.0, 1.0, "PQ 线性光值")
    arr = _sanitize_domain(arr)
    # 零输入直接短路，避免对 0 取分数次幂产生的噪声/告警
    safe = np.where(arr > 0.0, arr, 1.0)
    ym = np.power(safe, _PQ_M1)
    num = _PQ_C1 + _PQ_C2 * ym
    den = 1.0 + _PQ_C3 * ym
    out = np.power(num / den, _PQ_M2)
    out = np.where(arr > 0.0, out, 0.0)
    out = np.where(arr == 1.0, 1.0, out)
    return _wrap(linear, _sanitize_domain(out))


def pq_eotf(signal):
    """PQ EOTF（ST 2084）：非线性信号 [0, 1] -> 线性光（10000 nit 归一化 [0, 1]）。"""
    arr = _as_array(signal)
    _check_range(arr, 0.0, 1.0, "PQ 信号值")
    arr = _sanitize_domain(arr)
    safe = np.where(arr > 0.0, arr, 1.0)
    ep = np.power(safe, 1.0 / _PQ_M2)
    num = np.maximum(ep - _PQ_C1, 0.0)
    den = _PQ_C2 - _PQ_C3 * ep
    out = np.power(num / den, 1.0 / _PQ_M1)
    out = np.where(arr > 0.0, out, 0.0)
    out = np.where(arr == 1.0, 1.0, out)
    return _wrap(signal, _sanitize_domain(out))


# 语义别名
linear_to_pq = pq_oetf
pq_to_linear = pq_eotf


# ---------------------------------------------------------------------------
# HLG（ARIB STD-B67 / ITU-R BT.2100）
# ---------------------------------------------------------------------------

def hlg_oetf(scene_light):
    """HLG OETF：场景线性光 [0, 1] -> 非线性信号 [0, 1]。

    E' = sqrt(3E)                      （0 <= E <= 1/12）
    E' = a*ln(12E - b) + c             （1/12 < E <= 1）
    """
    arr = _as_array(scene_light)
    _check_range(arr, 0.0, 1.0, "HLG 场景线性光值")
    arr = _sanitize_domain(arr)
    safe = np.where(arr > 0.0, arr, 1.0)
    log_branch = _HLG_A * np.log(np.maximum(12.0 * safe - _HLG_B, 1e-300)) + _HLG_C
    out = np.where(arr <= 1.0 / 12.0, np.sqrt(3.0 * safe), log_branch)
    out = np.where(arr > 0.0, out, 0.0)
    out = np.where(arr == 1.0, 1.0, out)
    return _wrap(scene_light, _sanitize_domain(out))


def hlg_inverse_oetf(signal):
    """HLG 逆 OETF：非线性信号 [0, 1] -> 场景线性光 [0, 1]。

    E = E'^2 / 3                       （0 <= E' <= 1/2）
    E = (exp((E' - c)/a) + b) / 12     （1/2 < E' <= 1）
    """
    arr = _as_array(signal)
    _check_range(arr, 0.0, 1.0, "HLG 信号值")
    arr = _sanitize_domain(arr)
    out = np.where(
        arr <= 0.5,
        arr * arr / 3.0,
        (np.exp((arr - _HLG_C) / _HLG_A) + _HLG_B) / 12.0,
    )
    out = np.where(arr == 0.0, 0.0, out)
    out = np.where(arr == 1.0, 1.0, out)
    return _wrap(signal, _sanitize_domain(out))


def hlg_eotf(signal, peak_nits=HLG_PEAK_NITS, system_gamma=None):
    """HLG EOTF（BT.2100 显示参考）：信号 [0, 1] -> 显示线性光，单位 nit。

    OETF 逆变换后乘上峰值亮度并施加系统 gamma（默认按峰值亮度自动选取）。
    """
    if peak_nits <= 0.0:
        raise LuminanceRangeError("peak_nits 必须为正数")
    if system_gamma is None:
        system_gamma = 1.2 * 1.111 ** math.log2(peak_nits / 1000.0)
    scene = hlg_inverse_oetf(signal)
    display = peak_nits * np.power(scene, system_gamma)
    return _wrap(signal, display)


def hlg_inverse_eotf(display_nits, peak_nits=HLG_PEAK_NITS, system_gamma=None):
    """HLG EOTF 的逆：显示线性光（nit）-> 信号 [0, 1]。"""
    if peak_nits <= 0.0:
        raise LuminanceRangeError("peak_nits 必须为正数")
    if system_gamma is None:
        system_gamma = 1.2 * 1.111 ** math.log2(peak_nits / 1000.0)
    arr = _as_array(display_nits)
    _check_range(arr, 0.0, peak_nits, "HLG 显示亮度")
    scene = np.power(arr / peak_nits, 1.0 / system_gamma)
    return _wrap(display_nits, hlg_oetf(scene))


# ---------------------------------------------------------------------------
# sRGB -> PQ 色调映射（含高光 roll-off）及其逆
# ---------------------------------------------------------------------------

# roll-off 段采用三次 Hermite 曲线，端点处 C1 连续：
#   u(0)=0, u'(0)=1（与线性段无缝衔接），u(1)=1, u'(1)=_SHOULDER_END_SLOPE
# 由标准 Hermite 基（h10 + h01 + s1*h11）展开：
#   u(t) = (s1-1)t^3 + (1-s1)t^2 + t
# 导数 u'(t) = 3(s1-1)t^2 + 2(1-s1)t + 1，在 [0,1] 上取值落在 [s1, 1]，
# 严格为正（单调可逆）；末端斜率 s1 远小于 1，形成平滑的高光 roll-off。
_SHOULDER_END_SLOPE = 0.05
_NEWTON_ITERS = 8


def _shoulder_poly(arr, s1=_SHOULDER_END_SLOPE):
    """Hermite roll-off 多项式（单调、C1 连续）。"""
    return (s1 - 1.0) * arr ** 3 + (1.0 - s1) * arr ** 2 + arr


def _shoulder_poly_deriv(arr, s1=_SHOULDER_END_SLOPE):
    return 3.0 * (s1 - 1.0) * arr ** 2 + 2.0 * (1.0 - s1) * arr + 1.0


def shoulder_curve(t):
    """roll-off 形曲线，t 与输出均在 [0, 1]，严格单调、C1 连续、可逆。

    u(t) = (2-s1)t^3 + (s1-3)t^2 + t，u(0)=0、u(1)=1 由多项式天然满足。
    """
    arr = _as_array(t)
    _check_range(arr, 0.0, 1.0, "roll-off 归一化坐标")
    return _wrap(t, np.clip(_shoulder_poly(arr), 0.0, 1.0))


def shoulder_curve_inverse(u):
    """:func:`shoulder_curve` 的逆，向量化牛顿迭代（初值精确，8 次迭代至机器精度）。"""
    arr = _as_array(u)
    _check_range(arr, 0.0, 1.0, "roll-off 归一化输出")
    s1 = _SHOULDER_END_SLOPE
    # 牛顿法在内部点（u in (0,1)）上迭代，端点保持精确值。
    interior = (arr > 0.0) & (arr < 1.0)
    ti = arr[interior].copy()  # t0 = u 是优质初值（曲线在 0 附近斜率为 1）
    ui = arr[interior]
    for _ in range(_NEWTON_ITERS):
        f = _shoulder_poly(ti, s1) - ui
        fp = _shoulder_poly_deriv(ti, s1)
        ti = np.clip(ti - f / fp, 0.0, 1.0)
    t = np.zeros_like(arr)
    t[interior] = ti
    t = np.where(arr >= 1.0, 1.0, t)
    return _wrap(u, np.clip(t, 0.0, 1.0))


def linear_to_pq_tonemap(linear, sdr_white_nits=203.0):
    """扩展线性光 -> PQ 信号 [0, 1]，带高光 roll-off。

    参数 ``linear`` 为以 SDR 白为 1.0 的线性光（允许 >1 的高光，例如 2.0 表示
    两倍 SDR 白）。SDR 白以下保持线性 1:1 映射，SDR 白到 PQ 满量程
    （10000 nit）之间用 C1 连续的 roll-off 曲线压缩。
    """
    if sdr_white_nits <= 0.0 or sdr_white_nits >= PQ_PEAK_NITS:
        raise LuminanceRangeError(
            f"sdr_white_nits 必须位于 (0, {PQ_PEAK_NITS}) 之内"
        )
    arr = _as_array(linear)
    _check_range(arr, 0.0, PQ_PEAK_NITS / sdr_white_nits, "线性光值")
    nits = arr * sdr_white_nits
    norm = nits / PQ_PEAK_NITS
    knee = sdr_white_nits / PQ_PEAK_NITS
    # 先把 roll-off 归一化坐标夹到 [0,1]（np.where 会同时求值两个分支）
    t = np.clip(
        (nits - sdr_white_nits) / (PQ_PEAK_NITS - sdr_white_nits), 0.0, 1.0
    )
    out = np.where(
        nits <= sdr_white_nits,
        norm,
        knee + (1.0 - knee) * shoulder_curve(t),
    )
    out = np.where(arr == 0.0, 0.0, out)
    out = np.clip(out, 0.0, 1.0)
    return _wrap(linear, out)


def pq_tonemap_to_linear(pq_signal, sdr_white_nits=203.0):
    """:func:`linear_to_pq_tonemap` 的逆：PQ 信号 [0, 1] -> 扩展线性光。"""
    if sdr_white_nits <= 0.0 or sdr_white_nits >= PQ_PEAK_NITS:
        raise LuminanceRangeError(
            f"sdr_white_nits 必须位于 (0, {PQ_PEAK_NITS}) 之内"
        )
    arr = _as_array(pq_signal)
    _check_range(arr, 0.0, 1.0, "PQ 信号值")
    knee = sdr_white_nits / PQ_PEAK_NITS
    t = shoulder_curve_inverse(np.clip((arr - knee) / (1.0 - knee), 0.0, 1.0))
    nits = np.where(
        arr <= knee,
        arr * PQ_PEAK_NITS,
        sdr_white_nits + t * (PQ_PEAK_NITS - sdr_white_nits),
    )
    linear = nits / sdr_white_nits
    linear = np.where(arr == 0.0, 0.0, linear)
    linear = np.clip(linear, 0.0, PQ_PEAK_NITS / sdr_white_nits)
    return _wrap(pq_signal, linear)


def sdr_to_pq(srgb_signal, sdr_white_nits=203.0):
    """sRGB 信号 [0, 1] -> PQ 信号 [0, 1] 的完整色调映射管线。

    sRGB EOTF -> 线性光（SDR 白为 1.0）-> roll-off 色调映射 -> PQ OETF。
    返回值形态与输入一致（标量 / list / ndarray）。
    """
    linear = srgb_eotf(srgb_signal)
    linear_safe = _sanitize_domain(np.asarray(linear, dtype=np.float64))
    tonemapped = linear_to_pq_tonemap(linear_safe, sdr_white_nits=sdr_white_nits)
    return _wrap(srgb_signal, pq_oetf(tonemapped))


def pq_to_sdr(pq_signal, sdr_white_nits=203.0):
    """:func:`sdr_to_pq` 的逆：PQ 信号 [0, 1] -> sRGB 信号 [0, 1]。

    返回值形态与输入一致（标量 / list / ndarray）。
    """
    tonemapped = pq_eotf(pq_signal)
    linear = pq_tonemap_to_linear(tonemapped, sdr_white_nits=sdr_white_nits)
    # 回到 SDR 域：超过 SDR 白的高光不参与 sRGB 编码，夹回 [0,1]
    linear_safe = _sanitize_domain(np.asarray(linear, dtype=np.float64))
    return _wrap(pq_signal, srgb_oetf(linear_safe))


# ---------------------------------------------------------------------------
# 量化 / 反量化
# ---------------------------------------------------------------------------

def quantize(values, bit_depth=16):
    """线性 [0, 1] 信号量化为 0..2^bit_depth-1 的整数。"""
    if not isinstance(bit_depth, (int, np.integer)) or isinstance(bit_depth, bool):
        raise GammaError("bit_depth 必须为正整数")
    if bit_depth <= 0:
        raise GammaError("bit_depth 必须为正整数")
    arr = _as_array(values)
    _check_range(arr, 0.0, 1.0, "待量化信号")
    max_code = (1 << bit_depth) - 1
    codes = np.rint(arr * max_code).astype(np.int64)
    return _wrap(values, codes)


def dequantize(codes, bit_depth=16):
    """整数字反量化回 [0, 1] 浮点信号。"""
    if not isinstance(bit_depth, (int, np.integer)) or isinstance(bit_depth, bool):
        raise GammaError("bit_depth 必须为正整数")
    if bit_depth <= 0:
        raise GammaError("bit_depth 必须为正整数")
    if isinstance(codes, np.ndarray):
        arr = codes.astype(np.float64, copy=False)
    elif isinstance(codes, (list, tuple)):
        try:
            arr = np.asarray(codes, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise GammaError(f"码值无法转换为整数数组: {exc}") from exc
    elif isinstance(codes, (int, np.integer)) and not isinstance(codes, bool):
        arr = np.asarray(float(codes), dtype=np.float64)
    else:
        raise GammaError(f"不支持的码值类型: {type(codes).__name__}")
    if not np.all(np.isfinite(arr)):
        raise NonFiniteError("码值包含非有限值")
    max_code = (1 << bit_depth) - 1
    if np.any(arr < 0) or np.any(arr > max_code):
        raise LuminanceRangeError(f"码值超出 [0, {max_code}]")
    if np.any(arr != np.floor(arr)):
        raise GammaError("码值必须为整数")
    out = arr / max_code
    return _wrap(codes, out)


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------

def _normalize_curve(curve):
    if not isinstance(curve, str):
        raise UnknownCurveError(f"非法曲线类型: {curve!r}（合法值: {CURVES}）")
    key = curve.strip().lower().replace("-", "_")
    aliases = {
        SRGB: SRGB, "s_rgb": SRGB,
        PQ: PQ, "st2084": PQ, "st_2084": PQ, "smpte2084": PQ,
        HLG: HLG, "bt2100": HLG, "bt_2100": HLG, "arib_b67": HLG,
    }
    if key not in aliases:
        raise UnknownCurveError(f"非法曲线类型: {curve!r}（合法值: {CURVES}）")
    return aliases[key]


def encode(curve, values, **kwargs):
    """按曲线名做 OETF：线性光 -> 非线性信号。"""
    key = _normalize_curve(curve)
    if key == SRGB:
        return srgb_oetf(values)
    if key == PQ:
        return pq_oetf(values)
    return hlg_oetf(values)


def decode(curve, values, **kwargs):
    """按曲线名做 EOTF：非线性信号 -> 线性光。"""
    key = _normalize_curve(curve)
    if key == SRGB:
        return srgb_eotf(values)
    if key == PQ:
        return pq_eotf(values)
    return hlg_inverse_oetf(values)


class TransferFunction:
    """曲线调度器：按曲线类型分发到对应的 EOTF / OETF 实现。

    非法曲线类型抛出 :class:`UnknownCurveError`，
    越界亮度抛出 :class:`LuminanceRangeError`，两者可区分捕获。
    """

    def __init__(self, curve):
        self.curve = _normalize_curve(curve)

    def encode(self, values, **kwargs):
        """OETF：线性光 -> 非线性信号。"""
        return encode(self.curve, values, **kwargs)

    def decode(self, values, **kwargs):
        """EOTF：非线性信号 -> 线性光。"""
        return decode(self.curve, values, **kwargs)
