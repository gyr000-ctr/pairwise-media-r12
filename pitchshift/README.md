# pitchshift

## 题面（可直接粘贴）

子目录：pitchshift。相位声码器变调。约束：0 半音恒等（≤1e-9）；输出长度与输入一致；基频比与目标差 ≤0.5%；相位连续（帧间相位差展开正确，无爆音）；非法半音可区分报错；10 万样本不超过 800 ms；并发安全。验收：cd pitchshift && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd pitchshift && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：音视频处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
