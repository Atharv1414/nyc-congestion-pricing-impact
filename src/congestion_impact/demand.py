"""Trips: did people stop taking taxis into the zone? The answer depends on which trips count."""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .panel import add_calendar

AREA = {
    "inside": "inside",
    "crossing": "crossing",
    "through": "through",
    "near_ring": "outside",
    "far_control": "outside",
    "outside_mixed": "outside",
}


def weekly_demand(agg: Path, pairs: pd.DataFrame) -> pd.DataFrame:
    con = duckdb.connect()
    con.register("pairs_df", pairs)
    d = con.sql(f"""
        select date_trunc('week', day)::timestamp as week, p."group", c.fare_type,
               sum(c.trips)::int as trips
        from '{agg / "zone_counts" / "*.parquet"}' c join pairs_df p using (pu_zone, do_zone)
        group by all
    """).df()
    d["area"] = d["group"].map(AREA)
    d = d.groupby(["week", "area", "fare_type"], as_index=False)["trips"].sum()
    return add_calendar(d).reset_index(drop=True)


def yoy(
    demand: pd.DataFrame,
    year: int,
    area: str,
    fare_types: list[str],
    block: int = 4,
    reps: int = 2000,
    seed: int = 7,
) -> dict[str, float]:
    """Year-over-year log change in weekly trips, treated area minus outside area.

    Intervals resample blocks of consecutive weeks, because neighbouring weeks are alike.
    """
    d = demand[demand["fare_type"].isin(fare_types)]
    w = d.groupby(["area", "year", "woy"])["trips"].sum()
    change = np.log(w.xs(year, level="year")) - np.log(w.xs(year - 1, level="year"))
    diff = (change[area] - change["outside"]).sort_index().to_numpy()
    rng = np.random.default_rng(seed)
    n = len(diff)
    starts = rng.integers(0, n, (reps, int(np.ceil(n / block))))
    idx = (starts[:, :, None] + np.arange(block)) % n
    boot = diff[idx.reshape(reps, -1)[:, :n]].mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    pct = lambda v: float(100 * (np.exp(v) - 1))  # noqa: E731
    return {
        "pct": pct(diff.mean()),
        "low": pct(low),
        "high": pct(high),
        "own_change_pct": pct(change[area].mean()),
        "outside_change_pct": pct(change["outside"].mean()),
    }


def flex_share(demand: pd.DataFrame) -> pd.DataFrame:
    s = demand.groupby(["area", "year", "fare_type"])["trips"].sum().unstack()
    return (s["flex"] / s.sum(axis=1)).unstack()
