import math

import pytest

import gamma

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

IMG = [[0.0, 0.25], [0.5, 1.0]]

CASES = [
    ("endpoints", lambda: gamma.Gamma().apply(IMG, 2.2)[0][0] == 0.0 and abs(gamma.Gamma().apply(IMG, 2.2)[1][1] - 1.0) < 1e-12),
    ("lut_len", lambda: len(gamma.Gamma().lut(2.2, 256)) == 256),
    ("lut_match", lambda: abs(gamma.Gamma().lut(2.2, 256)[128] - (128 / 255.0) ** 2.2) < 1e-9),
    ("monotonic", lambda: all(gamma.Gamma().lut(2.2, 256)[i] <= gamma.Gamma().lut(2.2, 256)[i + 1] + 1e-12 for i in range(255))),
    ("bad_gamma", lambda: _raises(lambda: gamma.Gamma().apply(IMG, 0.0))),
    ("bad_levels", lambda: _raises(lambda: gamma.Gamma().lut(2.2, 0))),
    ("bad_empty", lambda: _raises(lambda: gamma.Gamma().apply([], 2.2))),
    ("type", lambda: isinstance(gamma.Gamma().apply(IMG, 2.2), list)),
    ("det", lambda: gamma.Gamma().apply(IMG, 2.2) == gamma.Gamma().apply(IMG, 2.2)),
    ("shape", lambda: len(gamma.Gamma().apply(IMG, 2.2)) == 2),
    ("range", lambda: all(0.0 <= v <= 1.0 for row in gamma.Gamma().apply(IMG, 2.2) for v in row)),
    ("lut_type", lambda: isinstance(gamma.Gamma().lut(2.2, 16), list)),
    ("nan", lambda: _raises(lambda: gamma.Gamma().apply([[float('nan')]], 2.2))),
    ("no_mut", lambda: gamma.Gamma().apply([r[:] for r in IMG], 2.2) is not None),
    ("gamma_one", lambda: abs(gamma.Gamma().apply([[0.5]], 1.0)[0][0] - 0.5) < 1e-12),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
