"""性能/并发自检脚本（不计入 pytest 验收）。

用法：
    cd statesync && python bench/perf_check.py

对应题面：10 万次插值 <= 200 ms（README 旧口径为 <= 100 ms）。
"""

from __future__ import annotations

import concurrent.futures
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import statesync as ss  # noqa: E402

A = {"x": 0.0, "y": 0.0, "tag": "idle"}
B = {"x": 1.0, "y": 2.0, "tag": "run"}
N = 100_000
LIMIT_S = 0.200


def bench_interpolate_reused() -> float:
    sync = ss.StateSync()
    for i in range(2000):  # warmup
        sync.interpolate(A, B, (i % 100) / 100.0)
    t0 = time.perf_counter()
    for i in range(N):
        sync.interpolate(A, B, (i % 101) / 101.0)
    return time.perf_counter() - t0


def bench_interpolate_fresh() -> float:
    t0 = time.perf_counter()
    for _ in range(N):
        ss.StateSync().interpolate(A, B, 0.5)
    return time.perf_counter() - t0


def bench_buffer_sample() -> float:
    buf = ss.SnapshotBuffer(capacity=32, delay=0.1)
    for i in range(12):
        buf.push({"x": i * 0.1, "y": i * 0.2}, timestamp=i * 0.1)
    t0 = time.perf_counter()
    for i in range(N):
        buf.sample(0.05 + (i % 90) / 100.0)
    return time.perf_counter() - t0


def main() -> int:
    dt_reused = bench_interpolate_reused()
    dt_fresh = bench_interpolate_fresh()
    dt_sample = bench_buffer_sample()
    print("interpolate reused : %6.1f ms / %d  (limit 200 ms)  %s"
          % (dt_reused * 1000, N, "PASS" if dt_reused <= LIMIT_S else "FAIL"))
    print("interpolate fresh  : %6.1f ms / %d  (limit 200 ms)  %s"
          % (dt_fresh * 1000, N, "PASS" if dt_fresh <= LIMIT_S else "FAIL"))
    print("buffer sample      : %6.1f ms / %d" % (dt_sample * 1000, N))

    # 快速并发冒烟：8 线程各 2 万次纯插值，结果必须正确且无异常。
    sync = ss.StateSync()

    def work(_):
        for i in range(20_000):
            t = (i % 1000) / 1000.0
            r = sync.interpolate(A, B, t)
            assert math.isclose(r["x"], t, abs_tol=1e-9)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(work, range(8)))
    print("concurrency smoke  : PASS (8 x 20000 interpolations)")

    return 0 if dt_reused <= LIMIT_S and dt_fresh <= LIMIT_S else 1


if __name__ == "__main__":
    raise SystemExit(main())
