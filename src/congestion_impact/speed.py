"""Speed effect: year-over-year difference-in-differences with placebo, dynamics and bounds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import fe

CONTROLS = {
    "far": ["far_control"],
    "all_outside": ["near_ring", "far_control", "outside_mixed"],
}


def quarter(woy: pd.Series) -> np.ndarray:
    """13-week blocks: weeks 2-13, 14-26, 27-39, 40-52."""
    return ((woy - 1) // 13).clip(0, 3).to_numpy() + 1


def yoy_did(
    panel: pd.DataFrame,
    year: int,
    treated: str = "inside",
    control: str = "far",
    by: str | None = None,
    band: str | None = None,
    include_flex: bool = False,
    distance_control: bool = True,
) -> fe.Fit:
    """Change in log speed from `year - 1` to `year` for treated pairs, minus the change for
    control pairs.

    Each pair is compared with itself in the same week of the previous year and the same time
    band (pair x band x week-of-year effect), and against everything else on the road that week
    (week x band effect). Mean log distance absorbs shifts in trip length within a pair.
    """
    groups = [treated, *CONTROLS[control]]
    d = panel[panel["year"].isin([year - 1, year]) & panel["group"].isin(groups)]
    if not include_flex:
        d = d[d["metered"]]
    if band:
        d = d[d["band"] == band]
    post = ((d["group"] == treated) & (d["year"] == year)).to_numpy()
    if by == "quarter":
        q = quarter(d["woy"])
        x = pd.DataFrame({f"q{k}": (post & (q == k)).astype(float) for k in range(1, 5)})
    elif by == "month":
        m = ((d["woy"] - 2) // 4).clip(0, 12).to_numpy()
        x = pd.DataFrame({f"m{k:02d}": (post & (m == k)).astype(float) for k in range(13)})
    else:
        x = pd.DataFrame({"effect": post.astype(float)})
    if distance_control:
        x["log_miles"] = d["log_miles"].to_numpy()
    effects = [
        fe.codes(d["pu_zone"], d["do_zone"], d["band"], d["woy"], d["metered"]),
        fe.codes(d["week"], d["band"], d["metered"]),
    ]
    return fe.fit(
        d["log_mph"].to_numpy(),
        x,
        effects,
        d["trips"].to_numpy(),
        d["pu_zone"].to_numpy(),
        d["do_zone"].to_numpy(),
        d["week"].to_numpy().astype("datetime64[D]").astype("int64"),
    )


@dataclass
class Estimate:
    """Point estimates with delete-one-zone replicates."""

    names: list[str]
    coef: np.ndarray
    reps: np.ndarray  # zones x coefficients
    zones: list[int]
    n: int

    def __getitem__(self, name: str) -> dict[str, float]:
        i = self.names.index(name)
        return describe(self.coef[i], jackknife_se(self.reps[:, i])) | {"n": self.n}

    def influence(self, name: str = "effect") -> dict[str, float]:
        reps = self.reps[:, self.names.index(name)]
        worst = np.abs(reps - self.coef[self.names.index(name)]).argmax()
        return {
            "min_pct": pct(reps.min()),
            "max_pct": pct(reps.max()),
            "zones": len(reps),
            "most_influential_zone": self.zones[worst],
        }


def jackknife_se(reps: np.ndarray) -> float:
    g = len(reps)
    return float(np.sqrt((g - 1) / g * ((reps - reps.mean()) ** 2).sum()))


def describe(log_points: float, se: float) -> dict[str, float]:
    return {
        "pct": pct(log_points),
        "low": pct(log_points - 1.96 * se),
        "high": pct(log_points + 1.96 * se),
        "log": float(log_points),
        "se": float(se),
    }


def estimate(panel: pd.DataFrame, year: int, **kw) -> Estimate:
    """The model in `yoy_did`, with standard errors from deleting one zone at a time (every pair
    that starts or ends there) and re-estimating.

    Only 11 pickup zones serve as the far control. With so few, the analytic clustered errors in
    `fe` are too small: in simulation they reject a true null 17% of the time instead of 5%. A
    pair belongs to two zones, so the jackknife counts some variation twice, which errs safe.
    """
    full = yoy_did(panel, year, **kw)
    groups = [kw.get("treated", "inside"), *CONTROLS[kw.get("control", "far")]]
    used = panel[panel["group"].isin(groups) & panel["year"].isin([year - 1, year])]
    zones = sorted(int(z) for z in set(used["pu_zone"]) | set(used["do_zone"]))
    reps = np.array(
        [
            yoy_did(used[(used["pu_zone"] != z) & (used["do_zone"] != z)], year, **kw).coef
            for z in zones
        ]
    )
    return Estimate(full.names, full.coef, reps, zones, full.n)


def pct(log_points: float) -> float:
    return 100 * (np.exp(log_points) - 1)


def bounds(effect: dict[str, float], violations: list[float], m: float) -> tuple[float, float]:
    """Relative-magnitude bounds in the spirit of Rambachan and Roth (2023).

    If the 2025 parallel-trends violation is at most `m` times the largest one measured in the
    placebo year, the effect lies in this interval (sampling error added on top).
    """
    slack = m * max(abs(v) for v in violations) + 1.96 * effect["se"]
    return pct(effect["log"] - slack), pct(effect["log"] + slack)


def breakdown(effect: dict[str, float], violations: list[float]) -> float:
    """Largest `m` at which the interval in `bounds` still excludes zero (0 if none does)."""
    room = abs(effect["log"]) - 1.96 * effect["se"]
    return max(0.0, room / max(abs(v) for v in violations))


def trend_adjusted(effect: Estimate, placebo: Estimate, name: str = "effect") -> dict:
    """Subtract the placebo-year gap: right if the two areas were drifting apart at a steady
    rate. The same zone is deleted from both years, so their shared 2024 data is allowed for."""
    i, j = effect.names.index(name), placebo.names.index(name)
    common = [z for z in effect.zones if z in placebo.zones]
    e = effect.reps[[effect.zones.index(z) for z in common], i]
    p = placebo.reps[[placebo.zones.index(z) for z in common], j]
    return describe(effect.coef[i] - placebo.coef[j], jackknife_se(e - p))


def minutes_saved(panel: pd.DataFrame, effect: dict) -> dict[str, float]:
    """Vehicle time with a passenger at the estimated effect, metered trips inside the zone."""
    d = panel[(panel["group"] == "inside") & (panel["year"] == 2025) & panel["metered"]]
    hours, trips = d["hours"].sum(), d["trips"].sum()
    saved = {k: hours * effect[k] / 100 for k in ("pct", "low", "high")}
    return {
        "trips": int(trips),
        "mean_minutes": 60 * hours / trips,
        "saved_minutes_per_trip": 60 * saved["pct"] / trips,
        "saved_low": 60 * saved["low"] / trips,
        "saved_high": 60 * saved["high"] / trips,
        "saved_hours": saved["pct"],
    }
