import math

import pytest

import shadowmap

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

D = [[0.5, 0.5], [0.5, 0.5]]

CASES = [
    ("self_lit", lambda: shadowmap.ShadowMap(0.01).visibility(D, (0.5, 0.5), 0.5) > 0.9),
    ("occluded", lambda: shadowmap.ShadowMap(0.01).visibility(D, (0.5, 0.5), 0.9) < 0.1),
    ("lit", lambda: shadowmap.ShadowMap(0.01).visibility(D, (0.5, 0.5), 0.3) > 0.9),
    ("range", lambda: 0.0 <= shadowmap.ShadowMap(0.01).visibility(D, (0.5, 0.5), 0.4) <= 1.0),
    ("bad_samples", lambda: _raises(lambda: shadowmap.ShadowMap(0.01, 0))),
    ("bad_bias", lambda: _raises(lambda: shadowmap.ShadowMap(-0.1, 8))),
    ("bad_map", lambda: _raises(lambda: shadowmap.ShadowMap(0.01).visibility([], (0.5, 0.5), 0.5))),
    ("bad_uv", lambda: _raises(lambda: shadowmap.ShadowMap(0.01).visibility(D, (1.5, 0.5), 0.5))),
    ("type", lambda: isinstance(shadowmap.ShadowMap(0.01).visibility(D, (0.5, 0.5), 0.5), float)),
    ("det", lambda: shadowmap.ShadowMap(0.01).visibility(D, (0.5, 0.5), 0.5) == shadowmap.ShadowMap(0.01).visibility(D, (0.5, 0.5), 0.5)),
    ("edge", lambda: 0.0 <= shadowmap.ShadowMap(0.01).visibility(D, (0.0, 0.0), 0.5) <= 1.0),
    ("nan_depth", lambda: _raises(lambda: shadowmap.ShadowMap(0.01).visibility(D, (0.5, 0.5), float('nan')))),
    ("many_samples", lambda: isinstance(shadowmap.ShadowMap(0.01, 32).visibility(D, (0.5, 0.5), 0.4), float)),
    ("bias_monotonic", lambda: shadowmap.ShadowMap(0.02).visibility(D, (0.5, 0.5), 0.51) >= shadowmap.ShadowMap(0.001).visibility(D, (0.5, 0.5), 0.51)),
    ("no_mut", lambda: shadowmap.ShadowMap(0.01).visibility([r[:] for r in D], (0.5, 0.5), 0.4) is not None),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
