from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_panel(effect=0.05, drift=0.0, noise=0.03, seed=1, spill=0.0) -> pd.DataFrame:
    """A small world with a known answer: speed effect `effect` inside the zone from 2025, and an
    optional steady drift (log points a year) between the zone and the control."""
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2023-01-09", "2025-12-22", freq="W-MON")
    iso = weeks.isocalendar()
    year, woy = iso["year"].to_numpy().astype(int), iso["week"].to_numpy().astype(int)
    keep = (woy >= 2) & (woy <= 52)
    weeks, year, woy = weeks[keep], year[keep], woy[keep]
    rows = []
    for zone in range(1, 31):
        group = "inside" if zone <= 15 else "near_ring" if zone <= 20 else "far_control"
        for dest in range(zone, zone + 3):
            for band, shift in (("weekday_peak", 0.0), ("overnight", 0.4)):
                pair = rng.normal(0, 0.3)
                season = 0.1 * np.sin(2 * np.pi * woy / 52) * (1.5 if group == "inside" else 1)
                common = 0.02 * np.cos(np.arange(len(weeks)))
                treat = (group == "inside") * (effect * (year == 2025) + drift * (year - 2023))
                treat = treat + (group == "near_ring") * spill * (year == 2025)
                miles = rng.normal(0.5, 0.2, len(weeks))
                y = 2 + pair + shift + season + common + treat + 0.3 * miles
                rows.append(
                    pd.DataFrame(
                        {
                            "pu_zone": zone,
                            "do_zone": dest,
                            "group": group,
                            "week": weeks,
                            "band": band,
                            "metered": True,
                            "trips": rng.integers(60, 120, len(weeks)),
                            "log_mph": y + rng.normal(0, noise, len(weeks)),
                            "log_miles": miles,
                            "hours": 10.0,
                            "year": year,
                            "woy": woy,
                        }
                    )
                )
    return pd.concat(rows, ignore_index=True)


@pytest.fixture(scope="session")
def panel():
    return make_panel()
