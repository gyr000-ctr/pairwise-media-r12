"""传递函数与 HDR 曲线验收测试。

覆盖：
* sRGB 分段 EOTF/OEF（精确阈值、非 2.2 幂近似、往返 <= 1e-12、端点精确）；
* PQ（ST 2084，10000 nit 归一，与独立参考式差 <= 1e-9、端点、单调）；
* HLG（BT.2100 OETF / 逆 OETF / EOTF 往返）；
* sRGB -> PQ 色调映射（高光 roll-off）及其逆，往返 <= 1e-3；
* 16 位量化往返 <= 1/65535；
* 10 万像素性能 <= 400 ms；
* 非法曲线类型与越界亮度可区分报错；非有限输入报错；并发安全。
"""

import math
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import gamma
from gamma import (
    HLG,
    PQ,
    SRGB,
    GammaError,
    LuminanceRangeError,
    NonFiniteError,
    TransferFunction,
    UnknownCurveError,
    hlg_eotf,
    hlg_inverse_eotf,
    hlg_inverse_oetf,
    hlg_oetf,
    linear_to_pq_tonemap,
    pq_eotf,
    pq_oetf,
    pq_to_sdr,
    pq_tonemap_to_linear,
    quantize,
    dequantize,
    shoulder_curve,
    shoulder_curve_inverse,
    sdr_to_pq,
    srgb_eotf,
    srgb_oetf,
)

RNG = np.random.default_rng(20261125)
GRID = np.linspace(0.0, 1.0, 20001)


# ---------------------------------------------------------------------------
# sRGB：精确分段实现
# ---------------------------------------------------------------------------

class TestSRGB:
    def test_exact_piecewise_thresholds(self):
        # 线性段边界值必须严格满足分段定义
        assert srgb_oetf(0.0031308) == pytest.approx(12.92 * 0.0031308, rel=0, abs=0.0)
        assert srgb_eotf(0.04045) == pytest.approx(0.04045 / 12.92, rel=0, abs=0.0)

    def test_piecewise_branch_values(self):
        # 线性段内若干点
        for x in (0.0, 0.001, 0.0031308):
            assert srgb_oetf(x) == pytest.approx(12.92 * x, abs=1e-15)
        # 幂段内点，按参考式直接计算
        x = 0.18
        assert srgb_oetf(x) == pytest.approx(
            1.055 * x ** (1.0 / 2.4) - 0.055, abs=1e-15
        )
        y = 0.5
        assert srgb_eotf(y) == pytest.approx(
            ((y + 0.055) / 1.055) ** 2.4, abs=1e-15
        )

    def test_not_gamma_22_approximation(self):
        # 与 2.2 幂近似必须存在明确差异（证明没有用近似式）
        x = 0.5
        approx = x ** (1.0 / 2.2)
        assert abs(srgb_oetf(x) - approx) > 1e-3

    def test_endpoints_exact(self):
        assert srgb_eotf(0.0) == 0.0
        assert srgb_eotf(1.0) == 1.0
        assert srgb_oetf(0.0) == 0.0
        assert srgb_oetf(1.0) == 1.0

    def test_roundtrip_1e12(self):
        back = srgb_oetf(srgb_eotf(GRID))
        assert np.max(np.abs(back - GRID)) <= 1e-12
        back2 = srgb_eotf(srgb_oetf(GRID))
        assert np.max(np.abs(back2 - GRID)) <= 1e-12

    def test_monotonic(self):
        out = srgb_oetf(GRID)
        assert np.all(np.diff(out) >= 0.0)
        out2 = srgb_eotf(GRID)
        assert np.all(np.diff(out2) >= 0.0)

    def test_knee_continuity(self):
        # 分段点两侧表达式应一致（标准定义自洽）
        lin = 12.92 * 0.0031308
        pw = 1.055 * 0.0031308 ** (1.0 / 2.4) - 0.055
        assert abs(lin - pw) < 1e-6
        lin_s = 0.04045 / 12.92
        pw_s = ((0.04045 + 0.055) / 1.055) ** 2.4
        assert abs(lin_s - pw_s) < 1e-6


# ---------------------------------------------------------------------------
# PQ：SMPTE ST 2084
# ---------------------------------------------------------------------------

# 与实现完全独立的参考常量 / 参考式（分数常量在测试内重新字面量给出）
REF_M1, REF_M2 = 0.1593017578125, 78.84375
REF_C1, REF_C2, REF_C3 = 0.8359375, 18.8515625, 18.6875


def ref_pq_oetf(l):
    if l <= 0.0:
        return 0.0
    lm = l ** REF_M1
    return ((REF_C1 + REF_C2 * lm) / (1.0 + REF_C3 * lm)) ** REF_M2


def ref_pq_eotf(n):
    if n <= 0.0:
        return 0.0
    ep = n ** (1.0 / REF_M2)
    return (max(ep - REF_C1, 0.0) / (REF_C2 - REF_C3 * ep)) ** (1.0 / REF_M1)


class TestPQ:
    def test_reference_formula_1e9(self):
        pts = RNG.random(5000)
        got = pq_oetf(pts)
        ref = np.vectorize(ref_pq_oetf)(pts)
        assert np.max(np.abs(got - ref)) <= 1e-9

        got_i = pq_eotf(pts)
        ref_i = np.vectorize(ref_pq_eotf)(pts)
        assert np.max(np.abs(got_i - ref_i)) <= 1e-9

    def test_10000_nit_normalization(self):
        # 10000 nit 归一化：线性 1.0 对应 PQ 信号 1.0
        assert pq_oetf(1.0) == 1.0
        assert pq_eotf(1.0) == 1.0
        assert pq_oetf(0.0) == 0.0
        assert pq_eotf(0.0) == 0.0

    def test_known_midpoint(self):
        # ST 2084 常用校验点：信号 0.5 约对应 0.00922457 归一化线性光
        assert abs(pq_eotf(0.5) - ref_pq_eotf(0.5)) <= 1e-12
        assert 0.009 < pq_eotf(0.5) < 0.010
        # 10000 nit 归一化下，该点约为 92 nit
        assert 90.0 < pq_eotf(0.5) * 10000.0 < 95.0

    def test_roundtrip_machine_precision(self):
        pts = RNG.random(20000)
        back = pq_eotf(pq_oetf(pts))
        assert np.max(np.abs(back - pts)) <= 1e-12

    def test_monotonic(self):
        out = pq_oetf(GRID)
        assert np.all(np.diff(out) >= 0.0)


# ---------------------------------------------------------------------------
# HLG
# ---------------------------------------------------------------------------

class TestHLG:
    def test_piecewise_definition(self):
        # 1/12 以下为 sqrt(3E) 分支
        for x in (0.0, 0.02, 1.0 / 12.0):
            assert hlg_oetf(x) == pytest.approx(math.sqrt(3.0 * x), abs=1e-15)
        # 1/12 以上为对数分支
        a, b, c = 0.17883277, 0.28466892, 0.55991073
        x = 0.5
        assert hlg_oetf(x) == pytest.approx(a * math.log(12 * x - b) + c, abs=1e-9)

    def test_endpoints(self):
        assert hlg_oetf(0.0) == 0.0
        assert hlg_oetf(1.0) == 1.0
        assert hlg_inverse_oetf(0.0) == 0.0
        assert hlg_inverse_oetf(1.0) == 1.0

    def test_oetf_inverse_roundtrip(self):
        back = hlg_inverse_oetf(hlg_oetf(GRID))
        assert np.max(np.abs(back - GRID)) <= 1e-12
        back2 = hlg_oetf(hlg_inverse_oetf(GRID))
        assert np.max(np.abs(back2 - GRID)) <= 1e-12

    def test_branch_boundary(self):
        # E'=0.5 处两分支在 E=1/12 衔接
        assert abs(hlg_oetf(1.0 / 12.0) - 0.5) <= 1e-9
        assert abs(hlg_inverse_oetf(0.5) - 1.0 / 12.0) <= 1e-9

    def test_eotf_roundtrip_and_peak(self):
        sig = GRID
        nits = hlg_eotf(sig, peak_nits=1000.0)
        assert nits[-1] == pytest.approx(1000.0, abs=1e-10)
        back = hlg_inverse_eotf(nits, peak_nits=1000.0)
        assert np.max(np.abs(back - sig)) <= 1e-10

    def test_monotonic(self):
        assert np.all(np.diff(hlg_oetf(GRID)) >= 0.0)


# ---------------------------------------------------------------------------
# 色调映射：sRGB -> PQ（含高光 roll-off）及其逆
# ---------------------------------------------------------------------------

class TestTonemap:
    def test_endpoints(self):
        assert sdr_to_pq(0.0) == 0.0
        assert pq_to_sdr(0.0) == 0.0

    def test_roundtrip_1e3(self):
        pts = RNG.random(50000)
        back = pq_to_sdr(sdr_to_pq(pts))
        err = np.max(np.abs(back - pts))
        assert err <= 1e-3, f"往返误差 {err} 超过 1e-3"

    def test_roundtrip_dense_grid(self):
        back = pq_to_sdr(sdr_to_pq(GRID))
        assert np.max(np.abs(back - GRID)) <= 1e-3

    def test_output_in_pq_range(self):
        out = sdr_to_pq(GRID)
        assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_monotonic(self):
        out = sdr_to_pq(GRID)
        assert np.all(np.diff(out) >= -1e-15)

    def test_highlight_rolloff_exists(self):
        # roll-off 曲线：严格单调、端点精确、末端斜率显著小于 1（压缩高光）
        t = np.linspace(0.0, 1.0, 10000)
        u = shoulder_curve(t)
        assert u[0] == 0.0 and u[-1] == 1.0
        assert np.all(np.diff(u) > 0.0)
        d = np.diff(u) / np.diff(t)
        assert d[-1] < 0.2  # 末端斜率被压下来 -> roll-off
        # 与线性 1:1 不同，确实发生了重映射
        assert np.max(np.abs(u - t)) > 1e-3

    def test_shoulder_inverse_roundtrip(self):
        t = np.linspace(0.0, 1.0, 10000)
        back = shoulder_curve_inverse(shoulder_curve(t))
        assert np.max(np.abs(back - t)) <= 1e-10
        u = np.linspace(0.0, 1.0, 10000)
        back2 = shoulder_curve(shoulder_curve_inverse(u))
        assert np.max(np.abs(back2 - u)) <= 1e-10

    def test_extended_linear_rolloff_roundtrip(self):
        # 扩展线性光（含 >1 的高光，直达 10000 nit）全程可逆
        ext = np.linspace(0.0, 10000.0 / 203.0, 50000)
        pq_norm = linear_to_pq_tonemap(ext)
        assert pq_norm[-1] == pytest.approx(1.0, abs=1e-12)
        back = pq_tonemap_to_linear(pq_norm)
        assert np.max(np.abs(back - ext)) <= 1e-3

    def test_sdr_white_maps_consistently(self):
        # SDR 白（线性 1.0）以下保持线性 1:1
        below = np.linspace(0.0, 1.0, 1000)
        mapped = linear_to_pq_tonemap(below)
        assert np.max(np.abs(mapped - below * 203.0 / 10000.0)) <= 1e-15


# ---------------------------------------------------------------------------
# 16 位量化
# ---------------------------------------------------------------------------

class TestQuantize:
    def test_16bit_roundtrip(self):
        codes = quantize(GRID, 16)
        back = dequantize(codes, 16)
        assert np.max(np.abs(back - GRID)) <= 1.0 / 65535.0 + 1e-16

    def test_endpoint_codes(self):
        assert quantize(0.0, 16) == 0
        assert quantize(1.0, 16) == 65535

    def test_all_codes_exact(self):
        # 每个码值反量化后再量化必须回到自身
        codes = np.arange(65536)
        re = quantize(dequantize(codes, 16), 16)
        assert np.array_equal(re, codes)

    def test_10bit_likewise(self):
        codes = quantize(GRID, 10)
        back = dequantize(codes, 10)
        assert np.max(np.abs(back - GRID)) <= 1.0 / 1023.0 + 1e-16


# ---------------------------------------------------------------------------
# 性能：10 万像素 <= 400 ms（完整 sRGB->PQ 管线）
# ---------------------------------------------------------------------------

class TestPerformance:
    def test_100k_pixels_under_400ms(self):
        pixels = RNG.random(100_000)
        start = time.perf_counter()
        encoded = sdr_to_pq(pixels)
        decoded = pq_to_sdr(encoded)
        elapsed = (time.perf_counter() - start) * 1000.0
        assert decoded.shape == pixels.shape
        assert elapsed <= 400.0, f"耗时 {elapsed:.1f} ms 超过 400 ms 预算"

    def test_srgb_100k_under_400ms(self):
        pixels = RNG.random(100_000)
        start = time.perf_counter()
        srgb_eotf(pixels)
        srgb_oetf(pixels)
        elapsed = (time.perf_counter() - start) * 1000.0
        assert elapsed <= 400.0


# ---------------------------------------------------------------------------
# 错误分类：非法曲线类型 vs 越界亮度，可区分
# ---------------------------------------------------------------------------

class TestErrors:
    def test_unknown_curve_distinguishable(self):
        with pytest.raises(UnknownCurveError):
            TransferFunction("rec709-power")
        with pytest.raises(UnknownCurveError):
            gamma.encode("nope", 0.5)
        # 同时也是 ValueError，兼容通用捕获
        with pytest.raises(ValueError):
            TransferFunction(123)

    def test_out_of_range_distinguishable(self):
        with pytest.raises(LuminanceRangeError):
            srgb_oetf(1.5)
        with pytest.raises(LuminanceRangeError):
            pq_eotf(-0.1)
        with pytest.raises(LuminanceRangeError):
            hlg_oetf(1.1)
        with pytest.raises(LuminanceRangeError):
            linear_to_pq_tonemap(999.0)

    def test_error_types_are_distinct(self):
        assert issubclass(UnknownCurveError, GammaError)
        assert issubclass(LuminanceRangeError, GammaError)
        assert issubclass(NonFiniteError, GammaError)
        # 越界不是非法曲线，反之亦然 —— 两者可区分
        try:
            srgb_oetf(2.0)
        except UnknownCurveError:  # pragma: no cover - 不应走到
            pytest.fail("越界亮度被误报为非法曲线类型")
        except LuminanceRangeError:
            pass

    def test_non_finite_inputs(self):
        with pytest.raises(NonFiniteError):
            srgb_eotf([float("nan"), 0.5])
        with pytest.raises(NonFiniteError):
            pq_oetf([float("inf")])
        with pytest.raises(NonFiniteError):
            sdr_to_pq([0.1, float("-inf")])

    def test_bad_input_type(self):
        with pytest.raises(GammaError):
            srgb_eotf("not-a-number")
        with pytest.raises(GammaError):
            quantize(0.5, bit_depth=0)


# ---------------------------------------------------------------------------
# 形态与并发安全
# ---------------------------------------------------------------------------

class TestShapeAndConcurrency:
    def test_list_io_shape_preserved(self):
        nested = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        out = srgb_oetf(nested)
        assert isinstance(out, list)
        assert len(out) == 2 and len(out[0]) == 3
        back = srgb_eotf(out)
        assert np.allclose(np.asarray(back), np.asarray(nested), atol=1e-12)

    def test_scalar_io(self):
        assert isinstance(srgb_oetf(0.5), float)
        assert srgb_oetf(0.5) == pytest.approx(srgb_oetf(np.float64(0.5)))

    def test_deterministic(self):
        pts = RNG.random(1000)
        a = sdr_to_pq(pts)
        b = sdr_to_pq(pts)
        assert np.array_equal(a, b)

    def test_thread_safety(self):
        pts = RNG.random(20000)
        expected = sdr_to_pq(pts)

        def worker(_):
            return np.max(np.abs(sdr_to_pq(pts) - expected))

        with ThreadPoolExecutor(max_workers=8) as pool:
            errs = list(pool.map(worker, range(16)))
        assert max(errs) == 0.0

    def test_dispatch_class(self):
        for name, fn in ((SRGB, srgb_oetf), (PQ, pq_oetf), (HLG, hlg_oetf)):
            tf = TransferFunction(name)
            assert tf.curve == name
            assert tf.encode(0.7) == pytest.approx(fn(0.7), abs=1e-15)
