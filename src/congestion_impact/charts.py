"""Figures for the README, drawn from reports/results.json and the quality table."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE, INK, MUTED, GRID, SURFACE = (
    "#2a78d6",
    "#eb6834",
    "#0b0b0b",
    "#52514e",
    "#e4e3df",
    "#fcfcfb",
)
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10.5,
        "axes.edgecolor": GRID,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.titlelocation": "left",
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "svg.hashsalt": "congestion",
        "xtick.bottom": False,
        "ytick.left": False,
    }
)


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.2, metadata={"Software": None})
    plt.close(fig)


def dynamics(r: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, constrained_layout=True)
    panels = (("2024", "Placebo: 2024 against 2023", MUTED), ("2025", "2025 against 2024", BLUE))
    for ax, (year, title, colour) in zip(axes, panels, strict=True):
        rows = r["four_weekly"][year]
        x = range(1, 14)
        ax.axhline(0, color=MUTED, lw=1)
        ax.fill_between(
            x, [v["low"] for v in rows], [v["high"] for v in rows], color=colour, alpha=0.16, lw=0
        )
        ax.plot(x, [v["pct"] for v in rows], color=colour, lw=2, marker="o", ms=5)
        ax.set_title(title, fontsize=11.5)
        ax.set_xticks([1, 4, 7, 10, 13], ["Jan", "Apr", "Jul", "Sep", "Dec"])
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Speed inside the zone relative to\nfar control, change on year (%)")
    fig.suptitle(
        "Taxis got faster inside the zone at launch, and the gain faded",
        x=0.01,
        ha="left",
        fontweight="bold",
        fontsize=14,
    )
    fig.text(
        0.01,
        -0.06,
        "Four-week estimates with 95% intervals. The left panel should be flat at "
        "zero if the two areas were on parallel paths. It is not.",
        color=MUTED,
        fontsize=9.5,
    )
    _save(fig, out / "dynamics.png")


LABELS = {
    "main": "Main estimate",
    "all_outside_controls": "All outside zones as control",
    "include_flex": "Including flex-fare trips",
    "no_distance_control": "No distance control",
    "weekday_peak": "Weekday peak ($9)",
    "weekend_peak": "Weekend peak ($9)",
    "overnight": "Overnight ($2.25)",
}


def robustness(r: dict, out: Path) -> None:
    rows = [(LABELS[k], v) for k, v in r["robustness"]["2025"].items()]
    rows.insert(1, ("Adjusted for the 2024 drift", r["speed"]["trend_adjusted"]))
    fig, ax = plt.subplots(figsize=(9, 4.6), constrained_layout=True)
    ax.axvline(0, color=MUTED, lw=1)
    for i, (_label, v) in enumerate(rows):
        y = len(rows) - i
        colour = BLUE if i == 0 else MUTED
        ax.plot([v["low"], v["high"]], [y, y], color=colour, lw=2, solid_capstyle="round")
        ax.plot(v["pct"], y, "o", color=colour, ms=8, mec=SURFACE, mew=1.5)
        ax.text(9.1, y, f"{v['pct']:+.1f}%", va="center", color=INK)
    ax.set_yticks(range(len(rows), 0, -1), [label for label, _ in rows])
    ax.set_xlim(-2, 10)
    ax.set_xlabel("Effect on taxi speed inside the zone in 2025 (%), with 95% interval")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Positive in every regression, and largest at weekends")
    _save(fig, out / "robustness.png")


def quality(data: Path, out: Path) -> None:
    w = pd.read_csv(data / "quality_waterfall.csv", dtype={"file_month": str})
    w["month"] = pd.to_datetime(w["file_month"])
    flex = w[w["status"].str.contains("flex")].groupby("month")["trips"].sum()
    kept = w[w["status"].str.startswith("kept")].groupby("month")["trips"].sum()
    share = 100 * flex / kept
    fig, ax = plt.subplots(figsize=(9, 3.8), constrained_layout=True)
    ax.plot(share.index, share, color=BLUE, lw=2)
    ax.axvline(pd.Timestamp("2025-01-05"), color=MUTED, lw=1, ls="--")
    ax.text(pd.Timestamp("2025-01-15"), 1, "charge starts", color=MUTED, fontsize=9.5)
    ax.text(share.index[-1], share.iloc[-1] + 0.8, f"{share.iloc[-1]:.0f}%", ha="right")
    ax.set_ylim(0, None)
    ax.set_ylabel("Share of usable trips (%)")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Flex fares rose sharply just as the charge began")
    _save(fig, out / "flex_share.png")


def demand(r: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.6), constrained_layout=True)
    d = r["demand"]["inside"]["2025"]
    rows = [
        ("Before and after,\nmetered trips", d["metered"]["own_change_pct"], None, ORANGE),
        ("Before and after,\nall trips", d["all"]["own_change_pct"], None, ORANGE),
        ("Against control,\nmetered trips", d["metered"]["pct"], d["metered"], BLUE),
        ("Against control,\nall trips", d["all"]["pct"], d["all"], BLUE),
    ]
    ax.axvline(0, color=MUTED, lw=1)
    for i, (_label, v, ci, colour) in enumerate(rows):
        y = -i
        if ci:
            ax.plot([ci["low"], ci["high"]], [y, y], color=colour, lw=2, solid_capstyle="round")
        ax.plot(v, y, "o", color=colour, ms=9, mec=SURFACE, mew=1.5)
        left = ci is None and v < 0  # keep the label clear of the zero line
        ax.text(
            v - 0.4 if left else (ci["high"] if ci else v) + 0.5,
            y,
            f"{v:+.1f}%",
            va="center",
            ha="right" if left else "left",
        )
    ax.set_yticks([-i for i in range(4)], [x[0] for x in rows])
    ax.set_ylim(-3.6, 0.6)
    ax.set_xlim(-3, 13.5)
    ax.set_xlabel("Change in trips inside the zone, 2025 on 2024 (%)")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    spread = d["all"]["own_change_pct"] - d["metered"]["own_change_pct"]
    ax.set_title(f"Simple before-and-after counts disagree by {spread:.0f} points")
    _save(fig, out / "demand.png")


def draw_all(r: dict, data: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    dynamics(r, out)
    robustness(r, out)
    quality(data, out)
    demand(r, out)
