from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="congestion-impact")
    p.add_argument("--data", type=Path, default=ROOT / "data")
    sub = p.add_subparsers(dest="command", required=True)
    e = sub.add_parser("extract", help="aggregate the raw TLC files (slow, needs 2.2 GB of input)")
    e.add_argument("raw", type=Path)
    e.add_argument("--agg", type=Path, default=ROOT / "build")
    sub.add_parser("analyse", help="run every estimate and redraw the charts")
    a = p.parse_args(argv)
    if a.command == "extract":
        from . import extract, processed

        extract.extract(a.raw, a.agg)
        bad = extract.verify(a.raw, a.agg)
        if bad:
            raise SystemExit(f"these months do not reconcile to the raw files: {', '.join(bad)}")
        processed.write(a.agg, a.raw, a.data)
    else:
        from . import analysis, charts

        analysis.run(a.data, ROOT / "reports")
        # re-read so the charts draw from exactly what was written
        charts.draw_all(
            json.loads((ROOT / "reports" / "results.json").read_text()),
            a.data,
            ROOT / "docs" / "figures",
        )


if __name__ == "__main__":
    main()
