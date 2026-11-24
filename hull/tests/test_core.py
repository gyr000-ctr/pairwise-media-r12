import math

import pytest

import hull

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

PTS = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (1.0, 1.0)]

CASES = [
    ("count", lambda: len(hull.Hull().convex_hull(PTS)) == 4),
    ("area", lambda: abs(hull.Hull().area(PTS) - 4.0) < 1e-9),
    ("ccw", lambda: hull.Hull().area(hull.Hull().convex_hull(PTS)) > 0.0),
    ("inside_removed", lambda: (1.0, 1.0) not in hull.Hull().convex_hull(PTS)),
    ("bad_few", lambda: _raises(lambda: hull.Hull().convex_hull([(0.0, 0.0), (1.0, 1.0)]))),
    ("bad_collinear", lambda: _raises(lambda: hull.Hull().convex_hull([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]))),
    ("type", lambda: isinstance(hull.Hull().convex_hull(PTS), list)),
    ("det", lambda: hull.Hull().convex_hull(PTS) == hull.Hull().convex_hull(PTS)),
    ("no_dup", lambda: len(set(hull.Hull().convex_hull(PTS))) == len(hull.Hull().convex_hull(PTS))),
    ("float_area", lambda: isinstance(hull.Hull().area(PTS), float)),
    ("tri", lambda: len(hull.Hull().convex_hull([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)])) == 3),
    ("no_collinear_pts", lambda: all(abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])) > 1e-12 for a, b, c in zip(hull.Hull().convex_hull(PTS), hull.Hull().convex_hull(PTS)[1:] + hull.Hull().convex_hull(PTS)[:1], hull.Hull().convex_hull(PTS)[2:] + hull.Hull().convex_hull(PTS)[:2]))),
    ("empty", lambda: _raises(lambda: hull.Hull().convex_hull([]))),
    ("neg_coords", lambda: len(hull.Hull().convex_hull([(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)])) == 4),
    ("nan", lambda: _raises(lambda: hull.Hull().convex_hull([(float('nan'), 0.0), (1.0, 0.0), (0.0, 1.0)]))),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
