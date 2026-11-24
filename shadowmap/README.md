# shadowmap

## 题面（可直接粘贴）

子目录：shadowmap。深度偏移与 PCF 阴影。约束：同一深度处判为受光（偏移消除自阴影）；全遮挡为 0、全受光为 1（≤1e-6）；PCF 采样数可配且随采样数收敛；非法采样数可区分报错；1 万像素不超过 400 ms；并发安全。验收：cd shadowmap && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd shadowmap && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
