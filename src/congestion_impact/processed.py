"""Write and read the compact tables committed under data/."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from .demand import weekly_demand
from .panel import build_panel


def write(agg: Path, raw: Path, data: Path) -> None:
    data.mkdir(parents=True, exist_ok=True)
    panel, pairs, fee = build_panel(agg, raw)
    # four decimals is 0.01% of speed; full precision would triple the file size
    for col in ("log_mph", "log_miles", "hours"):
        panel[col] = panel[col].round(4).astype("float32")
    for col in ("pu_zone", "do_zone"):
        panel[col] = panel[col].astype("int16")
    for col in ("group", "band"):
        panel[col] = panel[col].astype("category")
    panel = panel.drop(columns=["year", "woy", "miles"]).sort_values(
        ["group", "pu_zone", "do_zone", "band", "metered", "week"]
    )
    panel.to_parquet(data / "speed_panel.parquet", index=False, compression="zstd")
    weekly_demand(agg, pairs).drop(columns=["year", "woy"]).to_csv(
        data / "weekly_demand.csv", index=False
    )
    cols = ["zone", "Zone", "Borough", "trips", "fee_share"]
    fee[cols].round(4).to_csv(data / "zone_fee_share.csv", index=False)
    duckdb.sql(f"""
        select file_month, status, vendor_id, trips
        from (select *, regexp_extract(filename, '(\\d{{4}}-\\d{{2}})-\\d{{2}}', 1) as file_month
              from read_parquet('{agg / "quality_waterfall" / "*.parquet"}', filename = true))
        order by all
    """).df().to_csv(data / "quality_waterfall.csv", index=False)


def read_panel(data: Path) -> pd.DataFrame:
    from .panel import add_calendar

    p = pd.read_parquet(data / "speed_panel.parquet")
    for col in ("group", "band"):
        p[col] = p[col].astype(str)
    return add_calendar(p).reset_index(drop=True)


def read_demand(data: Path) -> pd.DataFrame:
    from .panel import add_calendar

    return add_calendar(pd.read_csv(data / "weekly_demand.csv", parse_dates=["week"]))
