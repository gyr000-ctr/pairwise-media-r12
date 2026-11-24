import math

import pytest

import pitchshift

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

S = [0.1 * (i % 10) for i in range(2000)]

CASES = [
    ("zero_identity", lambda: all(abs(a - b) < 1e-9 for a, b in zip(pitchshift.PitchShift().shift(S, 0.0), S))),
    ("length", lambda: len(pitchshift.PitchShift().shift(S, 12.0)) == len(S)),
    ("ratio_up", lambda: len(pitchshift.PitchShift().shift(S, 7.0)) == len(S)),
    ("bad_frame", lambda: _raises(lambda: pitchshift.PitchShift(0))),
    ("bad_range", lambda: _raises(lambda: pitchshift.PitchShift().shift(S, 999.0))),
    ("empty", lambda: pitchshift.PitchShift().shift([], 0.0) == []),
    ("type", lambda: isinstance(pitchshift.PitchShift().shift(S, 2.0), list)),
    ("det", lambda: pitchshift.PitchShift().shift(S[:200], 3.0) == pitchshift.PitchShift().shift(S[:200], 3.0)),
    ("range", lambda: all(abs(v) <= 1.0 for v in pitchshift.PitchShift().shift(S, 5.0))),
    ("nan", lambda: _raises(lambda: pitchshift.PitchShift().shift([float('nan')], 1.0))),
    ("negative_semi", lambda: len(pitchshift.PitchShift().shift(S, -3.0)) == len(S)),
    ("short_input", lambda: isinstance(pitchshift.PitchShift().shift([0.1, 0.2], 1.0), list)),
    ("no_jump", lambda: max(abs(a - b) for a, b in zip(pitchshift.PitchShift().shift(S, 1.0), pitchshift.PitchShift().shift(S, 1.0)[1:])) < 1.0),
    ("bad_len_type", lambda: _raises(lambda: pitchshift.PitchShift().shift('abc', 1.0))),
    ("finite", lambda: all(math.isfinite(v) for v in pitchshift.PitchShift().shift(S[:500], 2.0))),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
