"""Build the analysis panel: who is treated, who is a control, and which weeks count."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

START = pd.Timestamp("2025-01-05")
MIN_PAIR_TRIPS = 600

# The ring next to the zone (60th to about 96th Street). Drivers avoiding the charge may
# divert here, so it is analysed separately rather than trusted as a control.
NEAR_RING = (43, 140, 141, 142, 143, 236, 237, 238, 239, 262, 263)


def zone_fee_share(zone_counts: str, con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Share of metered trips starting and ending in the same zone that paid the zone fee."""
    return con.sql(f"""
        select pu_zone as zone, sum(trips) as trips, sum(cbd_fee_trips) / sum(trips) as fee_share
        from '{zone_counts}'
        where day >= date '{START.date()}' and fare_type = 'metered' and pu_zone = do_zone
        group by 1 order by 1
    """).df()


def charged_zones(fee: pd.DataFrame, lookup: pd.DataFrame, min_trips: int = 10) -> set[int]:
    """Zones inside the charging zone, read from the fees actually charged.

    The shares split into two clear clusters: 77% and above inside the zone, 45% and below
    outside it (the highest of those straddle 60th Street). 0.6 sits in the empty gap between.
    """
    manhattan = set(lookup.loc[lookup["Borough"] == "Manhattan", "LocationID"])
    hit = fee[(fee["fee_share"] > 0.6) & (fee["trips"] >= min_trips)]
    return set(hit["zone"].astype(int)) & manhattan


def classify_pairs(
    zone_counts: str, inside: set[int], lookup: pd.DataFrame, con: duckdb.DuckDBPyConnection
) -> pd.DataFrame:
    """One row per Manhattan origin-destination pair with its exposure group."""
    pairs = con.sql(f"""
        select pu_zone, do_zone, sum(trips) as trips,
               sum(cbd_fee_trips) / sum(trips) as fee_share
        from '{zone_counts}'
        where day >= date '{START.date()}' and fare_type = 'metered'
        group by 1, 2
    """).df()
    manhattan = set(lookup.loc[lookup["Borough"] == "Manhattan", "LocationID"])
    pairs = pairs[pairs["pu_zone"].isin(manhattan) & pairs["do_zone"].isin(manhattan)].copy()
    ends = pairs["pu_zone"].isin(inside).astype(int) + pairs["do_zone"].isin(inside).astype(int)
    near = pairs["pu_zone"].isin(NEAR_RING) & pairs["do_zone"].isin(NEAR_RING)
    far = ~pairs["pu_zone"].isin(NEAR_RING) & ~pairs["do_zone"].isin(NEAR_RING)
    group = pd.Series("outside_mixed", index=pairs.index)
    group[(ends == 0) & near] = "near_ring"
    group[(ends == 0) & far] = "far_control"
    # both ends outside, but the usual route cuts through the zone
    group[(ends == 0) & (pairs["fee_share"] > 0.05)] = "through"
    group[ends == 1] = "crossing"
    group[ends == 2] = "inside"
    pairs["group"] = group
    return pairs.reset_index(drop=True)


def add_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    """ISO year and week; weeks 2-52 only, so every year has the same 51 whole weeks and the
    launch (Sunday 5 January 2025, the last day of week 1) splits cleanly between weeks."""
    iso = frame["week"].dt.isocalendar()
    out = frame.assign(year=iso["year"].astype(int), woy=iso["week"].astype(int))
    return out[out["woy"].between(2, 52) & out["year"].between(2023, 2025)]


def build_panel(agg: Path, raw: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (speed panel, pair table, zone fee table)."""
    con = duckdb.connect()
    lookup = pd.read_csv(raw / "taxi_zone_lookup.csv")
    counts = str(agg / "zone_counts" / "*.parquet")
    fee = zone_fee_share(counts, con)
    inside = charged_zones(fee, lookup)
    pairs = classify_pairs(counts, inside, lookup, con)
    con.register("pairs_df", pairs)
    cells = con.sql(f"""
        select s.pu_zone, s.do_zone, p."group", s.week::timestamp as week, s.band,
               s.metered,
               sum(s.trips)::int as trips,
               sum(s.sum_log_mph) / sum(s.trips) as log_mph,
               sum(s.sum_log_miles) / sum(s.trips) as log_miles,
               sum(s.miles) as miles, sum(s.hours) as hours
        from '{agg / "speed_cells" / "*.parquet"}' s
        join pairs_df p using (pu_zone, do_zone)
        group by all
    """).df()
    # sparse pairs carry noise, not information about any one week
    metered = cells["trips"].where(cells["metered"], 0)
    volume = metered.groupby([cells["pu_zone"], cells["do_zone"]]).transform("sum")
    cells = cells[volume >= MIN_PAIR_TRIPS]
    fee = fee.merge(lookup, left_on="zone", right_on="LocationID")
    return add_calendar(cells).reset_index(drop=True), pairs, fee
