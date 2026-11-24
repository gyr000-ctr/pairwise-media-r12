# gamma

## 题面（可直接粘贴）

子目录：gamma。gamma 与 LUT。约束：LUT 与解析式逐值一致（≤1e-9）；端点 0→0、1→1 精确；单调不减；gamma 非法可区分报错；10 万像素不超过 300 ms；并发安全。验收：cd gamma && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd gamma && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图像处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
