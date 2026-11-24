"""Synthetic control for speed inside the zone, with placebo-in-space inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def area_series(panel: pd.DataFrame, min_trips: int = 150) -> pd.DataFrame:
    """Weekly log speed for the zone as a whole and for each pickup zone outside it.

    Pair x band means are removed first, so a change in which journeys are made cannot look like
    a change in speed.
    """
    d = panel[
        panel["metered"]
        & panel["group"].isin(["inside", "far_control", "near_ring", "outside_mixed"])
    ].copy()
    key = ["pu_zone", "do_zone", "band"]
    pre = d[d["year"] < 2025]
    base = (pre["log_mph"] * pre["trips"]).groupby([pre[k] for k in key]).sum() / pre.groupby(key)[
        "trips"
    ].sum()
    d = d.join(base.rename("base"), on=key).dropna(subset=["base"])
    d["dev"] = (d["log_mph"] - d["base"]) * d["trips"]
    d["unit"] = np.where(d["group"] == "inside", "inside", "z" + d["pu_zone"].astype(str))
    g = d.groupby(["unit", "week"])[["dev", "trips"]].sum()
    wide = (g["dev"] / g["trips"]).unstack("unit")
    volume = g["trips"].unstack("unit")
    keep = wide.columns[(volume.min() >= min_trips) & wide.notna().all()]
    return wide[keep]


def weights(target: np.ndarray, donors: np.ndarray, ridge: float = 1e-4) -> np.ndarray:
    """Non-negative weights summing to one, after demeaning (an intercept shift is allowed)."""
    t, x = target - target.mean(), donors - donors.mean(axis=0)
    k = x.shape[1]
    q = x.T @ x / len(t) + ridge * np.eye(k) * np.var(t)
    c = x.T @ t / len(t)
    res = minimize(
        lambda w: 0.5 * w @ q @ w - c @ w,
        np.full(k, 1 / k),
        jac=lambda w: q @ w - c,
        bounds=[(0, 1)] * k,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(k)},
        method="SLSQP",
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not res.success:
        raise RuntimeError(res.message)
    return res.x


def fit_unit(wide: pd.DataFrame, unit: str, donors: list[str], start: pd.Timestamp) -> dict:
    pre = wide.index < start
    y, x = wide[unit].to_numpy(), wide[donors].to_numpy()
    w = weights(y[pre], x[pre])
    synth = x @ w
    gap = y - synth - (y[pre] - synth[pre]).mean()
    pre_rmse, post_rmse = np.sqrt((gap[pre] ** 2).mean()), np.sqrt((gap[~pre] ** 2).mean())
    return {
        "gap": pd.Series(gap, wide.index),
        "weights": pd.Series(w, donors),
        "effect": gap[~pre].mean(),
        "pre_rmse": pre_rmse,
        "ratio": post_rmse / pre_rmse,
    }


def run(wide: pd.DataFrame, start: pd.Timestamp, treated: str = "inside") -> dict:
    donors = [c for c in wide.columns if c != treated]
    main = fit_unit(wide, treated, donors, start)
    placebos = {d: fit_unit(wide, d, [x for x in donors if x != d], start) for d in donors}
    ratios = np.array([p["ratio"] for p in placebos.values()])
    main["p_ratio"] = (1 + (ratios >= main["ratio"]).sum()) / (1 + len(ratios))
    main["placebos"] = placebos
    return main
