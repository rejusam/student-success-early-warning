"""Weekly batch job: the top 20% of each module presentation by risk, with reasons.

For a prediction week and a presentation, build the cohort with the same feature query as
training (only data up to the week's cutoff day), score it with the released bundle and write
one CSV per module, plus a manifest.json with the model versions and counts:
  <out>/<presentation>/week-<k>/<module>.csv
Columns: code_module, code_presentation, week, rank, id_student, risk, reasons.
"""

import argparse
import json
import sys
from pathlib import Path

import duckdb

from early_warning.features import WEEKS, load_week
from early_warning.ingest import register_tables
from early_warning.scoring import LIST_FRACTION, load_bundle, risk_lists
from early_warning.views import create_views


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--bundle", type=Path, default=Path("build/model"))
    parser.add_argument("--week", type=int, choices=WEEKS, required=True)
    parser.add_argument("--presentation", required=True, help="for example 2014J")
    parser.add_argument("--out", type=Path, default=Path("build/risk-lists"))
    args = parser.parse_args(argv)

    bundle = load_bundle(args.bundle)
    con = duckdb.connect()
    register_tables(con, args.processed)
    create_views(con)
    df = load_week(con, args.week)
    con.close()
    df = df[df.code_presentation == args.presentation]
    if df.empty:
        parser.error(f"no students in presentation {args.presentation} at week {args.week}")

    lists = risk_lists(bundle, args.week, df)
    folder = args.out / args.presentation / f"week-{args.week}"
    folder.mkdir(parents=True, exist_ok=True)
    counts = {}
    for module, part in lists.groupby("code_module"):
        part.to_csv(folder / f"{module}.csv", index=False)
        counts[module] = {"cohort": int((df.code_module == module).sum()), "listed": len(part)}
    manifest = {
        "presentation": args.presentation,
        "week": args.week,
        "list": f"top {LIST_FRACTION:.0%} of each module presentation by risk",
        "model_version": bundle.manifest["weeks"][str(args.week)]["version"],
        "released_at_commit": bundle.manifest["released_at_commit"],
        "modules": counts,
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(counts)} lists to {folder}")
    return 0


if __name__ == "__main__":  # the image runs: python -m early_warning.batch ...
    sys.exit(main())
