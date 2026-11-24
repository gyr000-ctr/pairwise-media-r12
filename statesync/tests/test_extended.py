"""扩展契约：环形缓冲 / 延迟补偿 / 航位外推 / 分类报错 / 并发安全。

这些用例只做确定性断言，不写性能阈值（性能由 bench/perf_check.py 度量），
保证 ``python -m pytest tests -q`` 在任何机器上都稳定。
"""

import threading

import pytest

import statesync as ss


A = {"x": 0.0, "y": 0.0, "tag": "idle"}
B = {"x": 1.0, "y": 2.0, "tag": "run"}


# --------------------------------------------------------------------------- #
# 环形缓冲
# --------------------------------------------------------------------------- #
def test_ring_overwrites_oldest_and_keeps_capacity():
    buf = ss.SnapshotBuffer(capacity=4)
    for i in range(6):
        buf.push({"v": float(i)}, timestamp=float(i))
    assert len(buf) == 4
    assert [s["v"] for s in buf.snapshots()] == [2.0, 3.0, 4.0, 5.0]
    assert buf.oldest["v"] == 2.0
    assert buf.newest["v"] == 5.0
    assert buf.capacity == 4


def test_generic_ring_buffer_index_and_wrap():
    rb = ss.RingBuffer(3)
    for i in range(5):
        rb.append(i)
    assert rb.to_list() == [2, 3, 4]
    assert rb[0] == 2 and rb[-1] == 4
    assert rb.is_full() and not rb.is_empty()
    rb.clear()
    assert len(rb) == 0 and rb.is_empty()


def test_invalid_capacity_classified():
    for bad in (0, -2, True, 1.5, "nope"):
        with pytest.raises(ValueError):
            ss.RingBuffer(bad)
        with pytest.raises(ss.StateSyncError):
            ss.StateSync(capacity=bad)


# --------------------------------------------------------------------------- #
# 乱序 vs 重复：必须是不同类型 / 不同 code
# --------------------------------------------------------------------------- #
def test_duplicate_and_out_of_order_are_distinguishable_by_type():
    buf = ss.SnapshotBuffer()
    buf.push({"x": 0}, seq=1)
    buf.push({"x": 1}, seq=2)

    with pytest.raises(ss.DuplicateSnapshotError):
        buf.push({"x": 9}, seq=2)
    with pytest.raises(ss.DuplicateSnapshotError):
        buf.push({"x": 9}, seq=1)
    with pytest.raises(ss.OutOfOrderSnapshotError):
        buf.push({"x": 9}, seq=0)

    assert ss.DuplicateSnapshotError is not ss.OutOfOrderSnapshotError
    assert ss.DuplicateSnapshotError.code != ss.OutOfOrderSnapshotError.code
    assert issubclass(ss.DuplicateSnapshotError, ValueError)
    assert issubclass(ss.OutOfOrderSnapshotError, ValueError)


def test_duplicate_and_order_by_timestamp():
    buf = ss.StateSync()
    buf.push({"x": 0.0}, timestamp=0.0)
    buf.push({"x": 1.0}, timestamp=0.1)
    with pytest.raises(ss.DuplicateSnapshotError):
        buf.push({"x": 2.0}, timestamp=0.1)
    with pytest.raises(ss.OutOfOrderSnapshotError):
        buf.push({"x": 2.0}, timestamp=0.05)


def test_error_codes_distinct_on_raise():
    buf = ss.SnapshotBuffer()
    buf.push({"x": 1}, seq=1)
    buf.push({"x": 2}, seq=2)
    try:
        buf.push({"x": 3}, seq=2)
    except ss.DuplicateSnapshotError as e:
        assert e.code == "SNAPSHOT_DUPLICATE"
    else:  # pragma: no cover
        raise AssertionError("expected duplicate")
    try:
        buf.push({"x": 3}, seq=0)
    except ss.OutOfOrderSnapshotError as e:
        assert e.code == "SNAPSHOT_OUT_OF_ORDER"
    else:  # pragma: no cover
        raise AssertionError("expected out-of-order")


# --------------------------------------------------------------------------- #
# 多种写入形态
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "call",
    [
        lambda s: s.push({"x": 1.0}, timestamp=0.1),
        lambda s: s.push({"timestamp": 0.1, "x": 1.0}),
        lambda s: s.push(0.1, {"x": 1.0}),
        lambda s: s.push({"x": 1.0}, 0.1),
        lambda s: s.push((0.1, {"x": 1.0})),
        lambda s: s.push(x=1.0, timestamp=0.1),
    ],
)
def test_push_call_shapes(call):
    s = ss.StateSync()
    snap = call(s)
    assert snap["x"] == 1.0
    assert snap.time == pytest.approx(0.1)


def test_nested_envelope_push():
    s = ss.StateSync()
    s.push({"seq": 7, "timestamp": 0.7, "state": {"x": 42.0}})
    assert len(s) == 1
    assert s.newest["x"] == 42.0
    assert s.latest_snapshot.seq == 7


# --------------------------------------------------------------------------- #
# 延迟补偿选帧 + 区间插值
# --------------------------------------------------------------------------- #
def test_latency_compensated_playhead():
    sync = ss.StateSync(delay=0.1)
    for i in range(11):
        sync.push({"x": i * 0.1}, timestamp=i * 0.1)  # x == t
    assert sync.sample()["x"] == pytest.approx(0.9, abs=1e-9)
    assert sync.sample(0.55)["x"] == pytest.approx(0.55, abs=1e-9)
    assert sync.sample(now=1.05, latency=0.2)["x"] == pytest.approx(0.85, abs=1e-9)
    assert sync.sample_latency(0.05)["x"] == pytest.approx(0.95, abs=1e-9)

    old, new, u = sync.bracket()
    assert old.time <= 0.9 <= new.time
    assert 0.0 <= u <= 1.0


def test_empty_buffer_raises_distinct_error():
    s = ss.SnapshotBuffer()
    with pytest.raises(ss.EmptyBufferError):
        s.sample()
    with pytest.raises(ss.EmptyBufferError):
        s.newest  # noqa: B018


# --------------------------------------------------------------------------- #
# 航位外推精度
# --------------------------------------------------------------------------- #
def test_extrapolate_constant_velocity_exact():
    buf = ss.SnapshotBuffer()
    v, x0 = 3.0, 1.0
    for i in range(6):
        t = i * 0.2
        buf.push({"x": x0 + v * t}, timestamp=t)
    for ahead in (0.05, 0.1, 0.15):
        target = 1.0 + ahead
        got = buf.sample(target)["x"]
        truth = x0 + v * target
        assert abs(got - truth) / abs(truth) <= 0.05


def test_extrapolate_mild_curvature_within_five_percent():
    buf = ss.SnapshotBuffer()
    for i in range(11):
        t = i * 0.1
        buf.push({"x": t * t}, timestamp=t)
    for target in (1.02, 1.05):
        got = buf.sample(target)["x"]
        truth = target * target
        assert abs(got - truth) / abs(truth) <= 0.05


def test_normalized_extrapolate_and_opt_in_flag():
    assert ss.StateSync().extrapolate({"x": 0.0}, {"x": 1.0}, 1.5)["x"] == pytest.approx(1.5)
    assert ss.extrapolate({"x": 0.0}, {"x": 2.0}, -0.5)["x"] == pytest.approx(-1.0)
    r = ss.StateSync().interpolate(A, B, 1.5, extrapolate=True)
    assert r["x"] == pytest.approx(1.5)
    # 默认严格模式仍然报错（可见契约）。
    with pytest.raises(ss.InterpolationOutOfRangeError):
        ss.StateSync().interpolate(A, B, 1.5)


def test_velocity_values():
    s = ss.StateSync()
    s.push({"x": 0.0}, timestamp=0.0)
    s.push({"x": 2.0}, timestamp=0.5)
    assert s.velocity()["x"] == pytest.approx(4.0)
    assert s.predict(ahead=0.5)["x"] == pytest.approx(4.0)


# --------------------------------------------------------------------------- #
# 非有限输入分类 + 不可变输入
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_t_classified(bad):
    with pytest.raises(ss.NonFiniteValueError):
        ss.StateSync().interpolate(A, B, bad)
    with pytest.raises(ValueError):
        ss.StateSync().interpolate(A, B, bad)


def test_nonfinite_field_classified():
    with pytest.raises(ss.NonFiniteValueError):
        ss.StateSync().interpolate({"x": float("nan"), "y": 0.0, "tag": "a"}, B, 0.5)
    buf = ss.SnapshotBuffer()
    with pytest.raises(ss.NonFiniteValueError):
        buf.push({"x": float("inf")}, timestamp=0.0)


def test_inputs_are_not_mutated():
    a = {"x": 0.0, "y": 0.0, "tag": "idle"}
    b = {"x": 1.0, "y": 2.0, "tag": "run"}
    before_a, before_b = dict(a), dict(b)
    r = ss.StateSync().interpolate(a, b, 0.5)
    assert a == before_a and b == before_b
    r["x"] = 999.0
    assert a["x"] == 0.0


def test_step_policies():
    s_old = ss.StateSync(step_policy="hold_old")
    assert s_old.interpolate(A, B, 0.9)["tag"] == "idle"
    s_new = ss.StateSync(step_policy="hold_new")
    assert s_new.interpolate(A, B, 0.1)["tag"] == "run"
    s_mid = ss.StateSync()
    assert s_mid.interpolate(A, B, 0.4)["tag"] == "idle"
    assert s_mid.interpolate(A, B, 0.5)["tag"] == "idle"  # 中点平局保留前帧
    assert s_mid.interpolate(A, B, 0.6)["tag"] == "run"


# --------------------------------------------------------------------------- #
# 并发安全（确定性：join 所有线程，不依赖墙钟）
# --------------------------------------------------------------------------- #
def test_concurrent_interpolate():
    sync = ss.StateSync()
    errors = []

    def worker():
        try:
            for i in range(4000):
                t = (i % 1000) / 1000.0
                r = sync.interpolate(A, B, t)
                assert abs(r["x"] - t) < 1e-9
                assert abs(r["y"] - 2.0 * t) < 1e-9
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert not errors


def test_concurrent_ring_append_invariants():
    rb = ss.RingBuffer(capacity=64)

    def prod(p):
        for i in range(2000):
            rb.append((p, i))

    threads = [threading.Thread(target=prod, args=(p,)) for p in range(6)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert len(rb) == 64
    assert len(rb.to_list()) == 64


def test_single_writer_multiple_readers():
    buf = ss.SnapshotBuffer(capacity=16, delay=0.05)
    errors = []
    total = 400

    def writer():
        try:
            for i in range(total):
                buf.push({"x": float(i)}, timestamp=i * 0.01)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    def reader():
        try:
            for _ in range(3000):
                try:
                    r = buf.sample()
                except ss.EmptyBufferError:
                    continue
                assert "x" in r
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    tw = threading.Thread(target=writer)
    readers = [threading.Thread(target=reader) for _ in range(4)]
    tw.start()
    for th in readers:
        th.start()
    tw.join()
    for th in readers:
        th.join()
    assert not errors
    assert len(buf) <= 16
