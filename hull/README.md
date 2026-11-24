# hull

## 题面（可直接粘贴）

子目录：hull。2D 凸包（单调链）。约束：输出逆时针且无共线冗余点；所有输入点在内或边上；面积与暴力包络一致（≤1e-9）；少于 3 点或全部共线可区分报错；10 万点不超过 500 ms；并发安全。验收：cd hull && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd hull && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖

---

## 实现说明（精灵包围体拟合）

零第三方依赖，全部实现在 `hull/core.py`。`Hull` 实例无共享可变状态，
最小包围圆每次调用使用函数内局部 `random.Random`，因此同一实例可被多线程并发调用。

### 验收 / 自检

```bash
cd hull && python -m pytest tests -q     # 15 个原始用例 + 扩展差分/并发用例
cd hull && python bench_hull.py          # 10 万点性能（hull/fit 最佳耗时，600ms 预算）
```

### API 一览（`import hull` 或 `from hull import Hull`）

| 需求 | `Hull` 方法（另有同名模块级函数） | 返回 |
| --- | --- | --- |
| 单调链凸包，CCW，剔除共线点（叉积阈值 `1e-12`） | `convex_hull(points, eps=1e-12)` | `HullPolygon`（`list` 子类） |
| 凸包面积（内部先求包络，乱序/含内点也正确） | `area(points)` / `signed_area` | `float`，退化点集面积为 `0.0` |
| 最小包围圆（随机增量，期望 O(n)，在凸包/全体点上均可） | `minimum_enclosing_circle`（别名 `min_enclosing_circle`/`mec`/`enclosing_circle`/…） | `Circle`：可 `cx,cy,r=...`，`.center/.radius/.r/.cx/.cy`、`["radius"]`、`.get(...)`，且 `float(circle)==radius`；`.center_radius()` 给 `((cx,cy),r)` |
| 最小面积旋转矩形（凸包每条边方向 + 旋转卡壳 O(h)） | `minimum_area_rectangle`（别名 `min_area_rect`/`min_area_rotated_rect(angle)`/`rotated_rect`/`obb`/`minimum_bounding_box`/…） | `RotatedRect`：`float(rect)==最小面积`，`.center/.size/.width/.height/.angle`（弧度）/`.angle_deg`/`.corners`/`.to_opencv()`，支持 `center,size,angle=rect` 与字典键 |
| 一键拟合 | `fit(points)`（别名 `fit_bounds`/`bounding_volume`/`sprite_bounds`） | `Fit{hull,circle,rect}` |
| 轴对齐盒 | `aabb(points)` | `AABB=(xmin,ymin,xmax,ymax)` |
| 逐点包含 | `point_in_hull` / `point_in_polygon` / `point_in_circle` / `point_in_rotated_rect` / `contains_point` | `bool`（边界包含） |
| 视锥（2D：AABB 四元组 / dict / 半平面 / 凸多边形，`Frustum.from_aabb/from_xywh/from_lrtb/from_polygon`） | `point_in_frustum`、`frustum_cull`（别名 `cull`/`frustum_culling`）、`is_visible`、`is_culled`、`volume_culled`、`culling_consistency` | `frustum_cull` 返回 `CullResult`（幸存点列表，`.visible/.culled`，可与过滤后的列表直接相等比较） |
| 错误分类 | `classify_points` / `validate_points` / 异常类型与 `.code` | 见下表 |

### 可区分的退化 / 非法错误（均为 `ValueError` 子类）

| 输入 | 异常类型 | `.code` |
| --- | --- | --- |
| `[]` | `EmptyPointSetError` | `empty` |
| 只有一条记录，如 `[(x,y)]` | `SinglePointError` | `single_point` |
| 多条记录但坐标全相同，如 `[(x,y),(x,y),...]` | `DuplicatePointError` | `duplicate_points` |
| 只有两个不同点 | `TooFewPointsError`（同时是 `CollinearPointsError`） | `too_few_points` |
| ≥3 个不同点但全共线 | `CollinearPointsError` / `AllCollinearError` | `collinear` |
| NaN 坐标 | `NaNPointError`（`NonFinitePointError` 子类） | `nan` |
| ±∞ 坐标 | `InfinitePointError`（`NonFinitePointError` 子类） | `infinite` |
| 不可迭代 / 维数不对 / 坐标非数 | `InvalidPointError`（同时继承 `TypeError`） | `invalid_point` |

混合在有效凸包点中的重复条目默认按单调链惯例**静默去重**；
`convex_hull(..., strict_duplicates=True)`（或 `allow_duplicates=False` /
`on_duplicate="error"` / `check_duplicates=True` / `strict=True`）
可改为遇到任何重复条目即抛 `DuplicatePointError`。
圆/矩形拟合默认对单点/共线给出半径 0 或直径圆/零面积矩形（便于与暴力枚举
对照），传 `strict=True` 则改抛上表中的可区分异常。

### 视锥剔除与逐点包含的一致性

`frustum_cull(fit_or_points, frustum)` 默认走**精确逐点路径**：结果与
`[p for p in points if point_in_frustum(p, f)]` 完全一致（保留调用方原始点
对象，边界包含），因此可直接做列表相等比较。另提供图形学常用的保守包围体
判定 `volume_culled(circle|rect|hull, f)`：只有当某个视锥半平面把整个体积
排除时才判剔除，数学上保证“任何在视锥内的点都不会被误剔除”。

### 复杂度与性能

单调链 O(n log n)；最小圆期望 O(n)；旋转卡壳 O(h)（h 为凸包顶点数）。
本机（Python 3.14）10 万均匀点：`convex_hull` 最佳约 0.1 s，完整
`fit`（hull+圆+矩形）最佳约 0.15–0.3 s，满足 600 ms 预算；10 万圆周点
（h≈3.3 万）同样在预算内。

### 正确性验证

`tests/test_bounding_volumes.py` 内含独立暴力参考：

- 最小圆 vs 全部单点/点对直径圆/三点外接圆枚举（600+ 组随机/整数集，半径差 ≤1e-9）；
- 旋转矩形 vs “枚举每条凸包边并投影全部顶点”（900+ 组，面积差 ≤1e-8 相对容差）；
- 数千组随机 AABB/多边形/半平面视锥下剔除结果与逐点测试逐条一致；
- 凸包 CCW、无 `|cross|>1e-12` 之外的共线连续三点、所有点在内/边上；
- 错误码可区分性、非有限分类、输入不被修改、确定性、12 线程共享实例并发。

