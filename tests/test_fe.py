import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from hypothesis import given, settings
from hypothesis import strategies as st

from congestion_impact import fe


def data(seed=0, n=600):
    rng = np.random.default_rng(seed)
    a, b = rng.integers(0, 20, n), rng.integers(0, 15, n)
    x = pd.DataFrame({"x1": rng.normal(size=n) + a / 10, "x2": rng.normal(size=n) - b / 9})
    w = rng.uniform(0.5, 3, n)
    y = 0.7 * x["x1"] - 1.2 * x["x2"] + a * 0.3 - b * 0.2 + rng.normal(size=n)
    return y.to_numpy(), x, a, b, w


def test_matches_dummy_variable_wls():
    y, x, a, b, w = data()
    dummies = pd.get_dummies(pd.DataFrame({"a": a, "b": b}).astype(str), drop_first=True)
    full = sm.WLS(y, sm.add_constant(pd.concat([x, dummies.astype(float)], axis=1)), w).fit()
    ours = fe.fit(y, x, [a, b], w, a, b, b)
    np.testing.assert_allclose(ours.coef, full.params[["x1", "x2"]], atol=1e-6)


def test_weight_scale_does_not_matter():
    y, x, a, b, w = data()
    one, two = fe.fit(y, x, [a, b], w, a, b, b), fe.fit(y, x, [a, b], w * 1000, a, b, b)
    np.testing.assert_allclose(one.coef, two.coef, atol=1e-9)
    np.testing.assert_allclose(one.se, two.se, rtol=1e-6)


def test_one_way_cluster_matches_statsmodels_up_to_small_sample_factor():
    y, x, a, b, w = data()
    xd = fe.demean(x.to_numpy(), [a], w / w.mean())
    yd = fe.demean(y, [a], w / w.mean())[:, 0]
    ref = sm.WLS(yd, xd, w).fit(cov_type="cluster", cov_kwds={"groups": a, "use_correction": False})
    # identical clusters on both sides collapse two-way clustering to one-way
    ours = fe.fit(y, x, [a], w, a, a, b)
    np.testing.assert_allclose(ours.se_cluster, ref.bse * np.sqrt(20 / 19), rtol=1e-6)


def test_reported_se_is_the_larger_one():
    y, x, a, b, w = data()
    f = fe.fit(y, x, [a, b], w, a, b, b)
    assert (f.se >= f.se_cluster).all() and (f.se >= f.se_time).all()


def test_contrast_matches_single_coefficient():
    y, x, a, b, w = data()
    f = fe.fit(y, x, [a, b], w, a, b, b)
    assert f.contrast({"x1": 1.0}) == pytest.approx((f.coef[0], f.se[0]))


def test_singletons_are_dropped_repeatedly():
    # removing row 0 (alone in g1) leaves group 1 of g2 with one row, which must go too
    g1, g2 = np.array([0, 1, 1, 2, 2]), np.array([1, 1, 0, 0, 0])
    assert fe.drop_singletons([g1, g2]).tolist() == [False, False, False, True, True]


def test_codes_distinguish_combinations():
    c = fe.codes(pd.Series([1, 1, 2, 1]), pd.Series(["a", "b", "a", "a"]))
    assert c[0] == c[3] and len(set(c)) == 3


@settings(max_examples=25, deadline=None)
@given(st.integers(0, 10_000))
def test_demeaned_columns_have_zero_group_means(seed):
    y, _, a, b, w = data(seed, 200)
    r = fe.demean(y, [a, b], w)[:, 0]
    for g in (a, b):
        assert np.abs(np.bincount(g, w * r)).max() < 1e-6


def test_false_positive_rate_with_shared_weekly_shocks():
    """Weekly shocks common to the treated area make independent-row errors far too small.
    The reported errors must keep false positives near 5%."""
    hits = naive = 0
    rng = np.random.default_rng(3)
    weeks, units = 60, 20
    for _ in range(150):
        unit, week = np.repeat(np.arange(units), weeks), np.tile(np.arange(weeks), units)
        treated = unit < 10
        shock = rng.normal(0, 1, weeks)
        y = shock[week] * treated + rng.normal(0, 1, len(unit))
        x = pd.DataFrame({"d": (treated & (week >= 30)).astype(float)})
        f = fe.fit(y, x, [unit, week], np.ones(len(y)), unit, unit, week, lags=4)
        hits += abs(f.coef[0] / f.se[0]) > 1.96
        resid_se = 2 * np.sqrt((1 + 0.5) / len(y)) * 2
        naive += abs(f.coef[0] / resid_se) > 1.96
    assert hits / 150 < 0.13
    assert naive / 150 > 0.13
