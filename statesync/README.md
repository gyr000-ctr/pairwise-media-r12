# statesync

## 题面（可直接粘贴）

子目录：statesync。快照插值（网络同步）。约束：t=0 取前一快照、t=1 取后一快照（逐字段 ≤1e-9）；数值字段线性插值、非数值字段阶跃保持；t 越界可区分报错；字段缺失报错；10 万次插值不超过 100 ms；并发安全。验收：cd statesync && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd statesync && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：游戏开发
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
