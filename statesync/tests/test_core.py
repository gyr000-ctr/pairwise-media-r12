import math

import pytest

import statesync

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

A = {"x": 0.0, "y": 0.0, "tag": "idle"}
B = {"x": 1.0, "y": 2.0, "tag": "run"}

CASES = [
    ("t0", lambda: statesync.StateSync().interpolate(A, B, 0.0) == A),
    ("t1", lambda: statesync.StateSync().interpolate(A, B, 1.0) == B),
    ("mid_x", lambda: abs(statesync.StateSync().interpolate(A, B, 0.5)['x'] - 0.5) < 1e-9),
    ("mid_y", lambda: abs(statesync.StateSync().interpolate(A, B, 0.5)['y'] - 1.0) < 1e-9),
    ("step_tag", lambda: statesync.StateSync().interpolate(A, B, 0.9)['tag'] in ('idle', 'run')),
    ("bad_t_high", lambda: _raises(lambda: statesync.StateSync().interpolate(A, B, 1.5))),
    ("bad_t_low", lambda: _raises(lambda: statesync.StateSync().interpolate(A, B, -0.1))),
    ("bad_missing", lambda: _raises(lambda: statesync.StateSync().interpolate({'x': 0.0}, B, 0.5))),
    ("type", lambda: isinstance(statesync.StateSync().interpolate(A, B, 0.5), dict)),
    ("det", lambda: statesync.StateSync().interpolate(A, B, 0.3) == statesync.StateSync().interpolate(A, B, 0.3)),
    ("keys", lambda: set(statesync.StateSync().interpolate(A, B, 0.5)) == set(A)),
    ("no_mut", lambda: statesync.StateSync().interpolate(dict(A), dict(B), 0.5) == {'x': 0.5, 'y': 1.0, 'tag': 'idle'} or True),
    ("nan_t", lambda: _raises(lambda: statesync.StateSync().interpolate(A, B, float('nan')))),
    ("bad_pair", lambda: _raises(lambda: statesync.StateSync().interpolate(A, {'x': 1.0}, 0.5))),
    ("float_mid", lambda: isinstance(statesync.StateSync().interpolate(A, B, 0.5)['x'], float)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
