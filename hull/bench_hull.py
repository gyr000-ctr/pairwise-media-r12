#!/usr/bin/env python3
"""10 万点性能与规格自检脚本（不属于 pytest 收集范围）。

用法：cd hull && python bench_hull.py

报告 convex_hull / minimum_enclosing_circle / minimum_area_rectangle / fit
的最佳耗时，并按题面 600 ms 预算判定。
"""

import math
import random
import sys
import time

import hull


def time_best(fn, *args, repeats=7, **kwargs):
    best = float("inf")
    last = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        last = fn(*args, **kwargs)
        best = min(best, (time.perf_counter() - t0) * 1000.0)
    return best, last


def main():
    h = hull.Hull()
    n = 100_000
    rng = random.Random(20261125)
    uniform = [(rng.uniform(0.0, 1000.0), rng.uniform(0.0, 1000.0)) for _ in range(n)]
    circle = [(math.cos(2.0 * math.pi * i / n), math.sin(2.0 * math.pi * i / n))
              for i in range(n)]

    budget = 600.0
    ok = True

    for label, data in [("uniform-100k", uniform), ("circle-100k", circle)]:
        t_hull, hp = time_best(h.convex_hull, data)
        t_circle, c = time_best(h.minimum_enclosing_circle, data)
        t_rect, rr = time_best(h.minimum_area_rectangle, data)
        t_fit, fit = time_best(h.fit, data)
        print(f"[{label}] hull_vertices={len(hp)}")
        print(f"  convex_hull             best {t_hull:7.1f} ms  (<=500 README / 600 prompt)")
        print(f"  min_enclosing_circle    best {t_circle:7.1f} ms  r={c.radius:.6f}")
        print(f"  min_area_rotated_rect   best {t_rect:7.1f} ms  area={float(rr):.6f}")
        print(f"  fit(all three)          best {t_fit:7.1f} ms  (budget {budget:.0f} ms)")
        within = t_hull <= 500.0 and t_fit <= budget
        ok = ok and within
        print(f"  -> {'WITHIN BUDGET' if within else 'OVER BUDGET'}")

    # 快速正确性抽查
    tri = [(0.0, 0.0), (4.0, 0.0), (0.0, 3.0)]
    assert abs(h.minimum_enclosing_circle(tri).radius - 2.5) < 1e-9
    assert abs(float(h.minimum_area_rectangle(tri)) - 12.0) < 1e-9
    assert abs(h.area([(0, 0), (2, 0), (2, 2), (0, 2), (1, 1)]) - 4.0) < 1e-9
    print("\nsanity: circle r=2.5, rect area=12.0, square area=4.0  OK")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
