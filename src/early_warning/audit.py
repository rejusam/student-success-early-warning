"""Day 4: fairness audit of the weekly models by student group, on the test presentation.

Models audited:
- lr_main (the model in use) and lr_with_demographics (comparison): refitted as on Day 2. The
  run stops if their metrics differ from reports/baseline_metrics.json.
- gbm (the Day 3 challenger): the registered MLflow model, loaded and not refitted. The run
  stops if its metrics differ from reports/gbm_metrics.json.

Nothing here changes a model. Modes:
- test: train 2013B, 2013J, 2014B; audit 2014J; MLflow store mlruns/.
- smoke: the test sample, 20 resamples, a size floor of 1. Checks the code only. Needs the output
  and MLflow store of `train_gbm.py --mode smoke`.
"""

import argparse
import json
from pathlib import Path

import duckdb
import mlflow
import numpy as np
import pandas as pd

from early_warning import fairness
from early_warning.baseline import build_pipeline
from early_warning.features import WEEKS, load_week
from early_warning.ingest import register_tables
from early_warning.metrics import evaluate
from early_warning.registry import default_tracking_uri, model_input, model_name
from early_warning.views import create_views

MODES = {
    # mode: (train, test, resamples, min group size, min at-risk students)
    "test": (("2013B", "2013J", "2014B"), ("2014J",), 1000, fairness.MIN_N, fairness.MIN_AT_RISK),
    "smoke": (("2013J",), ("2013J",), 20, 1, 1),
}
MODELS = ["lr_main", "gbm", "lr_with_demographics"]
SERVED = "lr_main"
SEED = 0


def check_matches(fresh: dict, saved: dict, models: list[str], filename: str) -> None:
    """Stop if a model's metrics now differ from the saved report: a sign the data, the code or
    the registered model changed since that report was written."""
    for week, by_model in fresh.items():
        for model in models:
            if by_model[model] != saved["metrics"][str(week)][model]:
                raise ValueError(f"week {week} {model}: this run differs from {filename}")


def load_gbm(week: int, alias: str):
    """The registered Day 3 model for the week, as it was registered (not refitted)."""
    return mlflow.pyfunc.load_model(f"models:/{model_name(week)}@{alias}")


def score_week(df: pd.DataFrame, train: tuple, test: tuple, week: int, alias: str):
    """The test presentation's cohort with a score column per audited model."""
    train_df = df[df.code_presentation.isin(train)]
    test_df = df[df.code_presentation.isin(test)].copy()
    y = train_df.at_risk.astype(int)
    test_df["lr_main"] = build_pipeline(False).fit(train_df, y).predict_proba(test_df)[:, 1]
    test_df["lr_with_demographics"] = (
        build_pipeline(True).fit(train_df, y).predict_proba(test_df)[:, 1]
    )
    # model_input() keeps only the model's own columns, so no demographic field reaches gbm.
    test_df["gbm"] = np.asarray(load_gbm(week, alias).predict(model_input(test_df)))[:, 1]
    return test_df


def audit_week(scored: pd.DataFrame, n_boot: int, min_n: int, min_at_risk: int) -> dict:
    """Point values, bootstrap ranges, flags, and the paired gbm - lr_main recall gap."""
    df = fairness.add_groups(scored)
    attributes = [fairness.ALL, *fairness.ATTRIBUTES, fairness.INTERSECTION]
    sizes = {a: df.groupby(a).size().rename("n").to_frame() for a in attributes}
    references = {a: fairness.reference_group(sizes[a]) for a in attributes}
    point, overall = fairness.audit_once(df, MODELS, attributes, references)
    reps, overall_reps = fairness.bootstrap(df, MODELS, attributes, references, n_boot, SEED)
    table = fairness.finish(point, fairness.intervals(reps), min_n, min_at_risk)
    table.insert(3, "reference", table["attribute"].map(references))

    diff = overall_reps["gbm"] - overall_reps[SERVED]
    gap = {
        "gbm_recall": round(overall["gbm"], 4),
        "lr_main_recall": round(overall[SERVED], 4),
        "difference": round(overall["gbm"] - overall[SERVED], 4),
        "difference_lo": round(float(diff.quantile(0.025)), 4),
        "difference_hi": round(float(diff.quantile(0.975)), 4),
        "share_of_resamples_gbm_ahead": round(float((diff > 0).mean()), 4),
    }
    return {"table": table, "gap": gap}


def summarise(table: pd.DataFrame) -> dict:
    """Per model and attribute: reference group, recall range over groups that are not small,
    the lowest-recall group, and the flagged groups."""
    out = {}
    for (model, attribute), part in table.groupby(["model", "attribute"], sort=False):
        judged = part[~part["small"]]
        lowest = judged.loc[judged["recall"].idxmin()] if len(judged) else None
        out.setdefault(model, {})[attribute] = {
            "reference": part["reference"].iloc[0],
            "groups": len(part),
            "small_groups": int(part["small"].sum()),
            "recall_min": round(float(judged["recall"].min()), 4) if len(judged) else None,
            "recall_max": round(float(judged["recall"].max()), 4) if len(judged) else None,
            "lowest_recall_group": None if lowest is None else lowest["group"],
            "flagged": [
                {"group": row.group, "flag": row.flag} for row in part.itertuples() if row.flag
            ],
        }
    return out


def flag_counts(table: pd.DataFrame) -> dict:
    """Per model: how many groups are flagged, for any reason, for recall and for calibration.
    The "All students" row is left out, because it is the whole cohort, not a group."""
    groups = table[table["attribute"] != fairness.ALL]
    return {
        model: {
            "groups_audited": len(part),
            "flagged": int((part["flag"] != "").sum()),
            "recall_gap": int(part["flag"].str.contains("recall gap").sum()),
            "calibration": int(part["flag"].str.contains("calibration").sum()),
        }
        for model, part in groups.groupby("model", sort=False)
    }


def fmt(value, low=None, high=None, digits: int = 3) -> str:
    if pd.isna(value):
        return "n/a"
    if low is None:
        return f"{value:.{digits}f}"
    return f"{value:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def blank_small_cells(table: pd.DataFrame) -> pd.DataFrame:
    """Intersection cells under the size floor keep their counts only (design decision 7)."""
    out = table.copy()
    ranges = [f"{m}_{end}" for m in fairness.MEASURES for end in ("lo", "hi")]
    blank = [*fairness.MEASURES, "at_risk_rate", *ranges]
    out.loc[(out["attribute"] == fairness.INTERSECTION) & out["small"], blank] = np.nan
    return out


def markdown(tables: dict, gaps: dict, settings: dict) -> str:
    """The audit tables in Markdown, ranges in brackets. Three decimals for groups, four for
    the model gap, which is small."""
    lines = [
        "# Fairness audit tables",
        "",
        "Written by `scripts/audit_fairness.py`. Full precision in `fairness_groups.csv` and",
        "`fairness_intersectional.csv`. Ranges are 95% bootstrap intervals "
        f"({settings['n_boot']} resamples).",
        "Recall and precision are for the top-20% list of each module presentation. Calibration",
        "gap = mean predicted risk - observed at-risk rate. small = fewer than "
        f"{settings['min_n']} students or {settings['min_at_risk']} at-risk students.",
        "",
        "## gbm minus lr_main, recall@top-20%, all students",
        "",
        "| Week | gbm | lr_main | Difference [95% range] | Share of resamples gbm ahead |",
        "|---|---|---|---|---|",
    ]
    for week, gap in gaps.items():
        lines.append(
            f"| {week} | {gap['gbm_recall']:.4f} | {gap['lr_main_recall']:.4f} | "
            f"{fmt(gap['difference'], gap['difference_lo'], gap['difference_hi'], 4)} | "
            f"{gap['share_of_resamples_gbm_ahead']:.3f} |"
        )
    for week, table in tables.items():
        lines += ["", f"## Week {week}"]
        for attribute, part in table.groupby("attribute", sort=False):
            wide = part.pivot(index="group", columns="model")
            lines += [
                "",
                f"### {attribute} (reference: {part['reference'].iloc[0]})",
                "",
                "| Group | n | At risk | lr_main recall | gbm recall | lr_with_demographics "
                "recall | lr_main precision | lr_main calibration gap | Flags |",
                "|---|---|---|---|---|---|---|---|---|",
            ]
            for group, row in wide.iterrows():
                flags = [f"{m}: {row[('flag', m)]}" for m in MODELS if row[("flag", m)]]
                label = f"{group} (small)" if row[("small", SERVED)] else group
                recalls = [
                    fmt(row[("recall", m)], row[("recall_lo", m)], row[("recall_hi", m)])
                    for m in MODELS
                ]
                lines.append(
                    f"| {label} | {int(row[('n', SERVED)])} | "
                    f"{fmt(row[('at_risk_rate', SERVED)])} | {' | '.join(recalls)} | "
                    + fmt(
                        row[("precision", SERVED)],
                        row[("precision_lo", SERVED)],
                        row[("precision_hi", SERVED)],
                    )
                    + " | "
                    + fmt(
                        row[("calibration_gap", SERVED)],
                        row[("calibration_gap_lo", SERVED)],
                        row[("calibration_gap_hi", SERVED)],
                    )
                    + f" | {'; '.join(flags)} |"
                )
    return "\n".join(lines) + "\n"


def write_outputs(out: Path, tables: dict, gaps: dict, settings: dict, header: dict) -> None:
    tables = {week: blank_small_cells(table) for week, table in tables.items()}
    rows = pd.concat([t.assign(week=w) for w, t in tables.items()], ignore_index=True)
    rows = rows[["week", *[c for c in rows.columns if c != "week"]]].round(4)
    single = rows[rows.attribute != fairness.INTERSECTION]
    single.to_csv(out / "fairness_groups.csv", index=False)
    cross = rows[rows.attribute == fairness.INTERSECTION]
    cross.to_csv(out / "fairness_intersectional.csv", index=False)

    summary = {
        **header,
        "settings": settings,
        "flag_counts": {str(w): flag_counts(t) for w, t in tables.items()},
        "by_week": {str(w): summarise(t) for w, t in tables.items()},
    }
    (out / "fairness_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    gap_report = {
        **header,
        "note": "Paired bootstrap of recall@top-20%, same resamples for both models.",
        "n_boot": settings["n_boot"],
        "by_week": {str(w): g for w, g in gaps.items()},
    }
    (out / "model_gap_bootstrap.json").write_text(json.dumps(gap_report, indent=2) + "\n")
    (out / "fairness_report.md").write_text(markdown(tables, gaps, settings))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("reports"))
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--mlflow-dir", type=Path, default=Path("mlruns"))
    parser.add_argument("--tracking-uri", help="default: SQLite file in --mlflow-dir")
    args = parser.parse_args(argv)

    train, test, n_boot, min_n, min_at_risk = MODES[args.mode]
    smoke = args.mode == "smoke"
    if smoke and args.out.resolve() == Path("reports").resolve():
        parser.error("smoke mode must not write to reports/: pass --out")
    if smoke and args.mlflow_dir.resolve() == Path("mlruns").resolve():
        parser.error("smoke mode must not read mlruns/: pass --mlflow-dir")

    # The registered gbm carries alias champion if it passed the gate, challenger if not.
    gate = json.loads((args.out / "gate.json").read_text())["gate"]
    alias = "champion" if gate["passed"] else "challenger"
    mlflow.set_tracking_uri(args.tracking_uri or default_tracking_uri(args.mlflow_dir))

    con = duckdb.connect()
    register_tables(con, args.processed)
    create_views(con)
    scored = {week: score_week(load_week(con, week), train, test, week, alias) for week in WEEKS}
    con.close()

    fresh = {w: {m: evaluate(df, m) for m in MODELS} for w, df in scored.items()}
    gbm_saved = json.loads((args.out / "gbm_metrics.json").read_text())
    check_matches(fresh, gbm_saved, ["gbm"], "gbm_metrics.json")
    if not smoke:
        saved = json.loads((args.out / "baseline_metrics.json").read_text())
        check_matches(fresh, saved, ["lr_main", "lr_with_demographics"], "baseline_metrics.json")

    tables, gaps = {}, {}
    for week, df in scored.items():
        result = audit_week(df, n_boot, min_n, min_at_risk)
        tables[week], gaps[week] = result["table"], result["gap"]
        print(f"week {week}: audited")

    settings = {
        "n_boot": n_boot,
        "seed": SEED,
        "list": "top 20% of each module presentation",
        "min_n": min_n,
        "min_at_risk": min_at_risk,
        "recall_ratio_limits": [fairness.RATIO_LOW, fairness.RATIO_HIGH],
        "calibration_gap_limit": fairness.CALIBRATION_LIMIT,
        "gbm_alias": alias,
    }
    header = {"train": list(train), "test": list(test), "weeks": list(WEEKS), "models": MODELS}
    write_outputs(args.out, tables, gaps, settings, header)
    print(f"Wrote fairness reports to {args.out}")
    return 0
