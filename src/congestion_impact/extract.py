"""Turn 128 million raw TLC trips into the small aggregates the analysis reads.

Months are processed one at a time, so memory use does not grow with the number of months.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

SQL = Path(__file__).resolve().parents[2] / "sql"
OUTPUTS = ("speed_cells", "zone_counts", "quality_waterfall")


def month_of(path: Path) -> str:
    """yellow_tripdata_2025-01.parquet -> 2025-01-01"""
    stem = path.stem.rsplit("_", 1)[-1]
    return pd.Timestamp(f"{stem}-01").date().isoformat()


def extract_month(con: duckdb.DuckDBPyConnection, path: Path) -> dict[str, duckdb.DuckDBPyRelation]:
    columns = {r[0] for r in con.sql(f"describe select * from '{path}'").fetchall()}
    fee = "coalesce(cbd_congestion_fee, 0)" if "cbd_congestion_fee" in columns else "0"
    month = month_of(path)
    con.execute((SQL / "clean_trips.sql").read_text().format(path=path, month=month, cbd_fee=fee))
    return {name: con.sql((SQL / f"{name}.sql").read_text()) for name in OUTPUTS}


def extract(raw: Path, dest: Path, memory: str = "1500MB", threads: int = 3) -> None:
    """Each month is written to its own file, so nothing but one month is ever held in memory and
    an interrupted run picks up where it stopped."""
    files = sorted(raw.glob("yellow_tripdata_*.parquet"))
    if not files:
        raise SystemExit(f"no yellow_tripdata_*.parquet files in {raw}")
    con = duckdb.connect()
    con.execute(f"set memory_limit='{memory}'")
    con.execute(f"set threads={threads}")
    con.execute("set enable_progress_bar=false")
    for path in files:
        marker = dest / OUTPUTS[-1] / f"{month_of(path)}.parquet"  # written last
        if marker.exists():
            continue
        for name, relation in extract_month(con, path).items():
            target = dest / name / marker.name
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(".partial")
            relation.write_parquet(str(partial))
            partial.replace(target)
        print(f"{path.name} done", flush=True)


def verify(raw: Path, dest: Path) -> list[str]:
    """Reconcile every month against the raw file. Returns the months that do not add up."""
    con = duckdb.connect()
    bad = []
    for path in sorted(raw.glob("yellow_tripdata_*.parquet")):
        month = month_of(path)
        one = lambda q: con.execute(q).fetchone()[0]  # noqa: E731
        try:
            total = one(f"select count(*) from '{path}'")
            fall = f"'{dest / 'quality_waterfall' / month}.parquet'"
            kept = one(f"select sum(trips) from {fall} where status like 'kept%'")
            valid = one(
                f"select sum(trips) from {fall} where left(status, 1) not in ('1', '2', '3', '4')"
            )
            ok = (
                one(f"select sum(trips) from {fall}") == total
                and one(f"select sum(trips) from '{dest / 'speed_cells' / month}.parquet'") == kept
                and one(f"select sum(trips) from '{dest / 'zone_counts' / month}.parquet'") == valid
            )
        except duckdb.Error:
            ok = False
        if not ok:
            bad.append(month)
    return bad
