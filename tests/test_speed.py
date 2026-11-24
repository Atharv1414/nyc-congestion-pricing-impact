import numpy as np
import pytest
from conftest import make_panel

from congestion_impact import speed, synth
from congestion_impact.panel import START


def test_recovers_known_effect(panel):
    s = speed.estimate(panel, 2025)["effect"]
    assert s["log"] == pytest.approx(0.05, abs=0.01)
    assert s["low"] < speed.pct(0.05) < s["high"]


def test_placebo_year_is_zero_when_trends_are_parallel(panel):
    s = speed.estimate(panel, 2024)["effect"]
    assert s["low"] < 0 < s["high"]


def test_seasonal_differences_between_areas_do_not_bias():
    # the fixture gives the zone a 50% larger seasonal swing; with no effect we must find none
    assert abs(speed.yoy_did(make_panel(effect=0.0, seed=5), 2025).coef[0]) < 0.01


def test_drift_is_caught_by_placebo_and_removed_by_adjustment():
    p = make_panel(effect=0.05, drift=-0.03, seed=2)
    placebo, effect = speed.estimate(p, 2024), speed.estimate(p, 2025)
    assert placebo["effect"]["log"] == pytest.approx(-0.03, abs=0.01)
    assert effect["effect"]["log"] == pytest.approx(0.02, abs=0.01)  # biased
    adjusted = speed.trend_adjusted(effect, placebo)
    assert adjusted["log"] == pytest.approx(0.05, abs=0.015)
    assert adjusted["se"] > effect["effect"]["se"]


def test_jackknife_controls_false_positives_where_analytic_errors_do_not():
    rejected = {"jackknife": 0, "analytic": 0}
    for seed in range(40):
        p = make_panel(seed=seed)
        e, f = speed.estimate(p, 2024)["effect"], speed.yoy_did(p, 2024)
        rejected["jackknife"] += abs(e["log"] / e["se"]) > 1.96
        rejected["analytic"] += abs(f.coef[0] / f.se[0]) > 1.96
    assert rejected["jackknife"] / 40 <= 0.10
    assert rejected["analytic"] > rejected["jackknife"]


def test_influence_names_the_zone_that_matters():
    p = make_panel(seed=3)
    odd = (p.pu_zone == 2) & (p.do_zone == 2) & (p.year == 2025)
    p.loc[odd, "log_mph"] += 1.0
    assert speed.estimate(p, 2025).influence()["most_influential_zone"] == 2


def test_quarterly_effects_average_to_the_overall(panel):
    f = speed.yoy_did(panel, 2025, by="quarter")
    q = f.coef[:4]
    assert np.mean(q) == pytest.approx(0.05, abs=0.01)


def test_contaminated_control_biases_and_far_control_does_not():
    p = make_panel(effect=0.05, spill=0.04, seed=4)
    far = speed.yoy_did(p, 2025, control="far").coef[0]
    mixed = speed.yoy_did(p, 2025, control="all_outside").coef[0]
    assert far == pytest.approx(0.05, abs=0.01)
    assert mixed < far - 0.005


def test_bounds_and_breakdown_agree():
    e = {"log": 0.05, "se": 0.01}
    m = speed.breakdown(e, [-0.02, 0.01])
    low, _ = speed.bounds(e, [-0.02, 0.01], m)
    assert low == pytest.approx(0, abs=1e-9)
    assert speed.breakdown({"log": 0.01, "se": 0.01}, [0.02]) == 0


def test_flex_rows_excluded_unless_asked(panel):
    flex = panel.assign(metered=False, log_mph=panel["log_mph"] + 1)
    both = __import__("pandas").concat([panel, flex], ignore_index=True)
    a = speed.yoy_did(panel, 2025)
    assert speed.yoy_did(both, 2025).coef[0] == pytest.approx(a.coef[0])
    assert speed.yoy_did(both, 2025, include_flex=True).n == 2 * a.n


def test_synthetic_weights_recover_a_convex_mix():
    rng = np.random.default_rng(0)
    donors = rng.normal(size=(120, 5)).cumsum(axis=0)
    w = synth.weights(0.6 * donors[:, 1] + 0.4 * donors[:, 3] + 7, donors, ridge=1e-9)
    np.testing.assert_allclose(w, [0, 0.6, 0, 0.4, 0], atol=0.01)
    assert w.sum() == pytest.approx(1)


def test_synthetic_control_finds_the_effect():
    wide = synth.area_series(make_panel(effect=0.08, noise=0.01, seed=8), min_trips=50)
    r = synth.run(wide, START)
    assert r["effect"] == pytest.approx(0.08, abs=0.02)


def test_minutes_saved_scales_with_hours(panel):
    m = speed.minutes_saved(panel, {"pct": 10.0, "low": 0.0, "high": 20.0})
    hours = panel[(panel.group == "inside") & (panel.year == 2025)].hours.sum()
    assert m["saved_hours"] == pytest.approx(0.10 * hours)
    assert m["saved_low"] == 0 and m["saved_high"] == pytest.approx(2 * m["saved_minutes_per_trip"])
