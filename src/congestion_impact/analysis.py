"""Run every estimate once and write reports/results.json. The README quotes only this file."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import demand, speed, synth
from .panel import START
from .processed import read_demand, read_panel

BANDS = ("weekday_peak", "weekend_peak", "overnight")


def _rounded(o):
    if isinstance(o, dict):
        return {str(k): _rounded(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_rounded(v) for v in o]
    if isinstance(o, float | np.floating):
        return round(float(o), 4)
    if isinstance(o, np.integer):
        return int(o)
    return o


def quality(data: Path) -> dict:
    w = pd.read_csv(data / "quality_waterfall.csv", dtype={"file_month": str})
    total = int(w["trips"].sum())
    by_status = w.groupby("status")["trips"].sum()
    y25 = w[w["file_month"] >= "2025-01"]
    v7 = y25[y25["vendor_id"] == 7]
    zero = v7[v7["status"].str.startswith("5")]["trips"].sum()
    return {
        "raw_trips": total,
        "waterfall": {k: int(v) for k, v in by_status.items()},
        "vendor7_trips_2025": int(v7["trips"].sum()),
        "vendor7_zero_duration_share": float(
            zero / v7[~v7["status"].str[0].isin(list("1234"))]["trips"].sum()
        ),
    }


def run(data: Path, reports: Path) -> dict:
    panel = read_panel(data)
    r: dict = {"quality": quality(data)}

    main = {y: speed.estimate(panel, y) for y in (2024, 2025)}
    effect, placebo = main[2025]["effect"], main[2024]["effect"]
    analytic = speed.yoy_did(panel, 2025)
    r["speed"] = {
        "placebo_2024": placebo,
        "effect_2025": effect,
        "influence_2025": main[2025].influence(),
        "analytic_se_2025": analytic.se[0],
        "trend_adjusted": speed.trend_adjusted(main[2025], main[2024]),
        "breakdown_m": speed.breakdown(effect, [placebo["log"]]),
        "bounds_m1": speed.bounds(effect, [placebo["log"]], 1.0),
    }

    quarters, r["four_weekly"] = {}, {}
    for y in (2024, 2025):
        q = speed.estimate(panel, y, by="quarter")
        quarters[y] = [q[f"q{k}"] for k in range(1, 5)]
        m = speed.estimate(panel, y, by="month")
        r["four_weekly"][y] = [m[f"m{k:02d}"] for k in range(13)]
    r["quarters"] = quarters
    r["quarter_breakdown_m"] = [
        speed.breakdown(q, [v["log"] for v in quarters[2024]]) for q in quarters[2025]
    ]

    variants = {
        "all_outside_controls": {"control": "all_outside"},
        "include_flex": {"include_flex": True},
        "no_distance_control": {"distance_control": False},
        **{b: {"band": b} for b in BANDS},
    }
    r["robustness"] = {
        y: {
            "main": main[y]["effect"],
            **{k: speed.estimate(panel, y, **kw)["effect"] for k, kw in variants.items()},
        }
        for y in (2024, 2025)
    }
    r["exposure"] = {
        y: {
            t: speed.estimate(panel, y, treated=t)["effect"]
            for t in ("crossing", "through", "near_ring")
        }
        for y in (2024, 2025)
    }

    s = synth.run(synth.area_series(panel), START)
    r["synthetic_control"] = {
        "effect_pct": speed.pct(s["effect"]),
        "pre_rmse_pct": 100 * s["pre_rmse"],
        "p_value": s["p_ratio"],
        "donors": len(s["weights"]),
        "weights": {k: v for k, v in s["weights"].items() if v > 0.02},
    }

    dem = read_demand(data)
    r["demand"] = {
        a: {
            y: {
                "metered": demand.yoy(dem, y, a, ["metered"]),
                "all": demand.yoy(dem, y, a, ["metered", "flex"]),
            }
            for y in (2024, 2025)
        }
        for a in ("inside", "crossing")
    }
    r["flex_share"] = demand.flex_share(dem).to_dict()
    r["time"] = speed.minutes_saved(panel, effect)

    r = _rounded(r)
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "results.json").write_text(json.dumps(r, indent=2) + "\n")
    return r
