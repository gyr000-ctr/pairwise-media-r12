# gamma

传递函数与 HDR 曲线（sRGB / PQ / HLG）及 sRGB→PQ 色调映射。

## 实现内容

代码位于 `gamma/` 包：

- `gamma/transfer.py` — 传递函数与 HDR 曲线主实现
- `gamma/errors.py` — 可区分的异常类型
- `gamma/core.py` — 保留的简单幂 gamma / LUT 接口（`Gamma`）
- `tests/test_transfer.py` — 传递函数/HDR 验收测试
- `tests/test_core.py` — 原有 LUT 接口测试

### 曲线

| 曲线 | OETF（线性→信号） | EOTF（信号→线性） |
| --- | --- | --- |
| sRGB（IEC 61966-2-1 分段式） | `srgb_oetf` / `linear_to_srgb` | `srgb_eotf` / `srgb_to_linear` |
| PQ（SMPTE ST 2084，10000 nit 归一） | `pq_oetf` / `linear_to_pq` | `pq_eotf` / `pq_to_linear` |
| HLG（ARIB STD-B67 / BT.2100） | `hlg_oetf` | `hlg_inverse_oetf` / `hlg_eotf` |

`sRGB` 严格采用分段定义（阈值 0.04045 / 0.0031308、线性斜率 12.92、幂指数 2.4），
**不使用 2.2 幂近似**。标准中两个舍入阈值本身并不严格互逆，因此实现共用一个
精确拐点 `0.04045 / 12.92` 保证 EOTF/OETF 互为逆函数；标准常量全部原样保留，
与标准式的偏差只出现在宽度约 5e-9 的拐点微小区间（< 3e-8）。

### 色调映射

- `sdr_to_pq` / `pq_to_sdr`：sRGB 信号 → PQ 信号的完整管线及其逆；
- `linear_to_pq_tonemap` / `pq_tonemap_to_linear`：扩展线性光（含 >1 的高光）
  与归一化 PQ 线性光之间的映射，SDR 白以下 1:1 线性，SDR 白到 10000 nit 之间
  采用 C1 连续、严格单调的三次 Hermite 曲线做高光 roll-off（牛顿法求逆）。

### 量化

`quantize(values, bit_depth)` / `dequantize(codes, bit_depth)`，默认 16 位。

### 统一入口与异常

- `TransferFunction("srgb" | "pq" | "hlg")` 调度器，及 `encode` / `decode` 函数；
- `UnknownCurveError`：非法曲线类型；
- `LuminanceRangeError`：亮度/信号越界；
- `NonFiniteError`：NaN / 无穷大输入。

三者都继承自 `GammaError`（同时是 `ValueError` 子类），非法曲线与越界亮度可
分别捕获、互不混淆。

## 约束达成情况

| 约束 | 实测 |
| --- | --- |
| sRGB 往返 ≤ 1e-12 | ~4e-16（机器精度） |
| PQ 与参考式差 ≤ 1e-9 | 0.0（同一系数定义） |
| PQ 10000 nit 归一 | 线性 1.0 ↔ 信号 1.0 |
| sRGB→PQ 色调映射往返 ≤ 1e-3 | ~6e-14（含全 HDR 量程） |
| 16 位量化往返 ≤ 1/65535 | 最大 7.63e-6 ≤ 1/65535 |
| 10 万像素 ≤ 400 ms | 完整 sRGB→PQ→sRGB 管线约 6–8 ms |
| 非法曲线 / 越界可区分报错 | `UnknownCurveError` ≠ `LuminanceRangeError` |
| 并发安全 | 纯函数、无可变共享状态，16 线程结果逐位一致 |

## 运行方式

```bash
cd gamma && python -m pytest tests -q
```

依赖：Python 3、numpy、pytest。
