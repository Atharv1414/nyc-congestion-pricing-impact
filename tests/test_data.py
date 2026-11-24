"""Extraction rules on hand-made trips, panel rules, and checks on the committed data."""

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from congestion_impact import demand, extract, panel, processed

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def trip(**kw):
    base = dict(
        VendorID=2,
        tpep_pickup_datetime="2025-01-08 10:00",
        minutes=12,
        trip_distance=2.0,
        PULocationID=161,
        DOLocationID=162,
        payment_type=1,
        fare_amount=12.0,
        cbd_congestion_fee=0.75,
    )
    return {**base, **kw}


CASES = {
    "kept: metered": trip(),
    "kept: flex fare (upfront price)": trip(payment_type=0),
    "1 pickup outside the file month": trip(tpep_pickup_datetime="2009-01-01 00:00"),
    "2 unknown or out-of-city zone": trip(DOLocationID=265),
    "3 refund, void or zero fare": trip(fare_amount=-12.0),
    "4 no charge, dispute or unknown": trip(payment_type=4),
    "5 zero or negative duration": trip(VendorID=7, minutes=0),
    "6 under 1 minute or over 3 hours": trip(minutes=181),
    "7 under 0.2 or over 60 miles": trip(trip_distance=0.1),
    "8 implausible speed": trip(trip_distance=30.0),
}


@pytest.fixture
def raw(tmp_path):
    rows = pd.DataFrame(CASES.values())
    rows["tpep_pickup_datetime"] = pd.to_datetime(rows["tpep_pickup_datetime"])
    rows["tpep_dropoff_datetime"] = rows["tpep_pickup_datetime"] + pd.to_timedelta(
        rows.pop("minutes"), unit="min"
    )
    d = tmp_path / "raw"
    d.mkdir()
    rows.to_parquet(d / "yellow_tripdata_2025-01.parquet")
    # older files have no zone-fee column
    old = rows.drop(columns="cbd_congestion_fee").assign(
        tpep_pickup_datetime=lambda x: x.tpep_pickup_datetime - pd.Timedelta(days=31),
        tpep_dropoff_datetime=lambda x: x.tpep_dropoff_datetime - pd.Timedelta(days=31),
    )
    old.to_parquet(d / "yellow_tripdata_2024-12.parquet")
    return d


def test_every_exclusion_rule_fires_once(raw, tmp_path):
    out = tmp_path / "agg"
    extract.extract(raw, out)
    w = pd.read_parquet(out / "quality_waterfall" / "2025-01-01.parquet")
    assert w.groupby("status").trips.sum().to_dict() == dict.fromkeys(CASES, 1)
    assert extract.verify(raw, out) == []


def test_zero_duration_trips_count_as_demand_but_not_speed(raw, tmp_path):
    out = tmp_path / "agg"
    extract.extract(raw, out)
    z = pd.read_parquet(out / "zone_counts" / "2025-01-01.parquet")
    s = pd.read_parquet(out / "speed_cells" / "2025-01-01.parquet")
    assert z.trips.sum() == 6 and s.trips.sum() == 2
    assert pd.read_parquet(out / "zone_counts" / "2024-12-01.parquet").cbd_fee_trips.sum() == 0


def test_interrupted_run_resumes_and_damage_is_detected(raw, tmp_path):
    out = tmp_path / "agg"
    extract.extract(raw, out)
    target = out / "speed_cells" / "2024-12-01.parquet"
    stamp = target.stat().st_mtime_ns
    extract.extract(raw, out)
    assert target.stat().st_mtime_ns == stamp
    target.write_bytes(b"PAR1")
    assert extract.verify(raw, out) == ["2024-12-01"]


def test_charged_zones_come_from_fees():
    fee = pd.DataFrame(
        {"zone": [1, 2, 3, 4], "trips": [500, 500, 3, 500], "fee_share": [0.99, 0.45, 1.0, 0.98]}
    )
    lookup = pd.DataFrame(
        {"LocationID": [1, 2, 3, 4], "Borough": ["Manhattan", "Manhattan", "Manhattan", "Queens"]}
    )
    assert panel.charged_zones(fee, lookup) == {1}


def test_calendar_drops_launch_week_and_aligns_years():
    f = pd.DataFrame({"week": pd.date_range("2022-12-26", "2025-12-29", freq="W-MON")})
    c = panel.add_calendar(f)
    assert c.groupby("year").size().to_dict() == {2023: 51, 2024: 51, 2025: 51}
    assert pd.Timestamp("2024-12-30") not in set(c.week)  # contains 5 January 2025


def test_committed_zone_definition_is_unambiguous():
    fee = pd.read_csv(DATA / "zone_fee_share.csv")
    m = fee[(fee.Borough == "Manhattan") & (fee.trips >= 10)]
    assert (m.fee_share > 0.6).sum() == 38
    # nothing sits near the threshold, so moving it would not move a zone
    assert not m.fee_share.between(0.5, 0.75).any()


def test_committed_panel_is_clean():
    p = processed.read_panel(DATA)
    assert not p.duplicated(["pu_zone", "do_zone", "band", "metered", "week"]).any()
    assert np.isfinite(p[["log_mph", "log_miles"]]).all().all()
    assert p.groupby("year").week.nunique().tolist() == [51, 51, 51]
    assert set(p.group) == set(demand.AREA)


def test_waterfall_accounts_for_all_trips():
    w = pd.read_csv(DATA / "quality_waterfall.csv")
    r = json.loads((ROOT / "reports" / "results.json").read_text())["quality"]
    assert w.trips.sum() == r["raw_trips"] == sum(r["waterfall"].values())
    assert w.file_month.nunique() == 36


def test_demand_yoy_is_exact_on_constructed_series():
    weeks = pd.date_range("2024-01-08", "2025-12-22", freq="W-MON")
    rows = [
        dict(
            week=w,
            area=a,
            fare_type="metered",
            trips=1000 * (1.1 if a == "inside" and w.year == 2025 else 1),
        )
        for w in weeks
        for a in ("inside", "outside")
    ]
    d = panel.add_calendar(pd.DataFrame(rows))
    d = d[d.year.isin([2024, 2025])]
    d.loc[(d.area == "inside") & (d.year == 2025), "trips"] = 1100
    d.loc[(d.area == "inside") & (d.year == 2024), "trips"] = 1000
    y = demand.yoy(d, 2025, "inside", ["metered"])
    assert y["pct"] == pytest.approx(10) and y["low"] == pytest.approx(10)


def test_readme_quotes_current_results():
    r = json.loads((ROOT / "reports" / "results.json").read_text())
    readme = (ROOT / "README.md").read_text()
    s = r["speed"]
    for v in (
        s["effect_2025"]["pct"],
        s["trend_adjusted"]["pct"],
        s["placebo_2024"]["pct"],
        r["quarters"]["2025"][0]["pct"],
        r["quarters"]["2025"][3]["pct"],
        r["robustness"]["2025"]["weekend_peak"]["pct"],
        r["synthetic_control"]["effect_pct"],
        r["demand"]["inside"]["2025"]["all"]["pct"],
    ):
        assert f"{v:+.1f}%" in readme, v


def test_sql_files_parse():
    con = duckdb.connect()
    for f in (ROOT / "sql").glob("*.sql"):
        assert con.extract_statements(
            f.read_text().format(path="x", month="2025-01-01", cbd_fee="0")
        )
