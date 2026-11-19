"""Day 6: monitoring report for a presentation scored by the released model.

For each prediction week:
1. Drift (known each week, before any outcome): PSI of every input and of the risk score
   against the training cohort, from the reference saved in the bundle (drift.py). This is
   the same check the batch job writes next to each week's lists.
2. Realised performance (known once the presentation has ended): recall and precision of the
   top-20% lists, observed at-risk rate, mean predicted risk, Brier score and ECE, compared
   with what was expected before release.

Rules (fixed before the first run, see docs/design/day6-monitoring-freeze.md):
- drift status: alert if any PSI >= 0.25, watch if any >= 0.1, else ok.
- outcome status: alert if recall at the top 20% < 0.9 x expected, or if mean predicted risk
  minus observed rate is more than 0.05 either way; else ok.

Modes:
- test: 2014J. Expected values from reports/dev_metrics.json (the same pipeline trained on
  2013B and 2013J, scored on 2014B: the only out-of-time estimate known before release). Stops
  if the realised metrics differ from reports/baseline_metrics.json. Writes
  reports/monitoring.json and reports/monitoring_report.md.
- smoke: the test sample (2013J, which the smoke bundle was trained on). Checks the code only;
  refuses to write to reports/.
"""

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

from early_warning.baseline import MODES as BASELINE_MODES
from early_warning.batch import drift_report
from early_warning.features import WEEKS, load_week
from early_warning.ingest import register_tables
from early_warning.metrics import calibration_metrics, evaluate
from early_warning.scoring import load_bundle, predict_risk
from early_warning.views import create_views

MODES = {
    # mode: (presentation to monitor, presentations the released model was trained on)
    "test": ("2014J", BASELINE_MODES["test"][0]),
    "smoke": ("2013J", ("2013J",)),
}
RECALL_RATIO = 0.9
CALIBRATION_GAP = 0.05
ACTIONS = {
    "watch": "Note it and look again next week.",
    "alert (drift)": (
        "Check the data pipeline for a change in how activity or assessments are recorded, "
        "then compare the lists with last year's in the modules whose inputs moved."
    ),
    "alert (outcomes)": (
        "Retraining review: retrain with the newest presentations, run the evaluation gate and "
        "the fairness audit, and promote only if both pass. Nothing is retrained or promoted "
        "automatically."
    ),
}


def outcomes(df: pd.DataFrame, risk: pd.Series, expected: dict | None) -> dict:
    """Realised performance of one week's lists, compared with the expected recall.

    expected None (smoke mode) means "expect what was realised"."""
    scored = df.assign(lr_main=risk)
    metrics = evaluate(scored, "lr_main")
    if expected is None:
        expected = {
            "recall": metrics["top_20"]["recall"],
            "precision": metrics["top_20"]["precision"],
            "rate": metrics["base_rate"],
        }
    calibration = calibration_metrics(df.at_risk, risk)
    mean_risk = round(float(risk.mean()), 4)
    gap = round(mean_risk - metrics["base_rate"], 4)
    recall = metrics["top_20"]["recall"]
    alert = recall < RECALL_RATIO * expected["recall"] or abs(gap) > CALIBRATION_GAP
    return {
        "status": "alert" if alert else "ok",
        "recall_top20": recall,
        "expected_recall_top20": expected["recall"],
        "precision_top20": metrics["top_20"]["precision"],
        "expected_precision_top20": expected["precision"],
        "observed_rate": metrics["base_rate"],
        "expected_rate": expected["rate"],
        "mean_risk": mean_risk,
        "calibration_gap": gap,
        "brier": calibration["brier"],
        "ece": calibration["ece"],
        "metrics": metrics,
    }


def assessments_due(full: pd.DataFrame, presentation: str, train: tuple[str, ...]) -> dict:
    """Non-exam assessments due by the cutoff day, per module: the mean over the training
    presentations' students, and the value in this presentation. n_due is set by each module's
    assessment calendar, so it moves whenever a calendar changes."""
    reference = full[full.code_presentation.isin(train)].groupby("code_module").n_due.mean()
    current = full[full.code_presentation == presentation].groupby("code_module").n_due.mean()
    return {
        module: {
            "training": round(float(reference[module]), 2) if module in reference else None,
            "current": round(float(current[module]), 2),
        }
        for module in current.index
    }


def run(
    con: duckdb.DuckDBPyConnection,
    bundle,
    presentation: str,
    train: tuple[str, ...],
    expected: dict | None,
) -> dict:
    """Drift and realised performance for every week. expected: {week: {recall, precision,
    rate}}, or None to expect what was realised (smoke mode)."""
    weeks = {}
    for week in WEEKS:
        full = load_week(con, week)
        df = full[full.code_presentation == presentation]
        weeks[str(week)] = {
            "drift": drift_report(bundle, week, df),
            "assessments_due": assessments_due(full, presentation, train),
            "outcomes": outcomes(df, predict_risk(bundle, week, df), expected and expected[week]),
        }
    return weeks


def expected_from(metrics_file: Path) -> dict:
    """Expected recall, precision and at-risk rate of lr_main per week, from a metrics report."""
    saved = json.loads(metrics_file.read_text())["metrics"]
    return {
        week: {
            "recall": saved[str(week)]["lr_main"]["top_20"]["recall"],
            "precision": saved[str(week)]["lr_main"]["top_20"]["precision"],
            "rate": saved[str(week)]["lr_main"]["base_rate"],
        }
        for week in WEEKS
    }


def check_matches_baseline(weeks: dict, baseline_file: Path) -> None:
    """Stop unless the released model's metrics equal the saved Day 2 test result."""
    saved = json.loads(baseline_file.read_text())["metrics"]
    for week, result in weeks.items():
        if result["outcomes"]["metrics"] != saved[week]["lr_main"]:
            raise ValueError(f"week {week}: realised metrics differ from {baseline_file.name}")


def markdown(report: dict) -> str:
    """The report as Markdown tables."""
    weeks = report["weeks"]
    lines = [
        f"# Monitoring report: {report['presentation']}",
        "",
        "Generated by `scripts/monitor.py`. Reference: the training cohort of the released "
        "model at the same week. PSI bands: below 0.1 stable, 0.1 to 0.25 moderate, 0.25 or "
        "more major.",
        "",
        "## Summary",
        "",
        "| Week | Students | Score PSI | Largest input PSI | Drift | Recall@20% "
        "(expected) | Precision@20% (expected) | At-risk rate (expected) | Mean risk | Gap | "
        "Outcomes |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for week, result in weeks.items():
        d, o = result["drift"], result["outcomes"]
        largest = d["largest_input"]
        lines.append(
            f"| {week} | {d['n_current']:,} | {d['score']['psi']} | {largest} "
            f"{d['inputs'][largest]['psi']} | {d['status']} | {o['recall_top20']} "
            f"({o['expected_recall_top20']}) | {o['precision_top20']} "
            f"({o['expected_precision_top20']}) | {o['observed_rate']} ({o['expected_rate']}) "
            f"| {o['mean_risk']} | {o['calibration_gap']} | {o['status']} |"
        )
    names = list(next(iter(weeks.values()))["drift"]["inputs"])
    lines += [
        "",
        "## PSI per input",
        "",
        "| Input | " + " | ".join(f"Week {w}" for w in weeks) + " |",
        "|---|" + "---|" * len(weeks),
    ]
    for name in names:
        cells = [weeks[w]["drift"]["inputs"][name] for w in weeks]
        lines.append(f"| {name} | " + " | ".join(f"{c['psi']} ({c['band']})" for c in cells) + " |")
    lines += ["", "## Mean of each input, training cohort -> this cohort", ""]
    lines += ["| Input | " + " | ".join(f"Week {w}" for w in weeks) + " |"]
    lines += ["|---|" + "---|" * len(weeks)]
    for name in names:
        cells = [weeks[w]["drift"]["inputs"][name] for w in weeks]
        if "reference_mean" not in cells[0]:
            continue
        lines.append(
            f"| {name} | "
            + " | ".join(f"{c['reference_mean']} -> {c['current_mean']}" for c in cells)
            + " |"
        )
    modules = list(next(iter(weeks.values()))["assessments_due"])
    lines += [
        "",
        "## Assessments due by the cutoff day, per module: training mean -> this presentation",
        "",
        "| Module | " + " | ".join(f"Week {w}" for w in weeks) + " |",
        "|---|" + "---|" * len(weeks),
    ]
    for module in modules:
        cells = [weeks[w]["assessments_due"].get(module) for w in weeks]
        lines.append(
            f"| {module} | "
            + " | ".join(f"{c['training']} -> {c['current']}" if c else "" for c in cells)
            + " |"
        )
    lines += ["", "## What a status asks of a person", ""]
    lines += [f"- **{status}**: {action}" for status, action in ACTIONS.items()]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--bundle", type=Path, default=Path("build/model"))
    parser.add_argument("--reports", type=Path, default=Path("reports"), help="read only")
    parser.add_argument("--out", type=Path, default=Path("reports"))
    parser.add_argument("--mode", choices=MODES, required=True)
    args = parser.parse_args(argv)

    smoke = args.mode == "smoke"
    if smoke and args.out.resolve().is_relative_to(Path("reports").resolve()):
        parser.error("smoke mode must not write to reports/: pass --out")

    presentation, train = MODES[args.mode]
    bundle = load_bundle(args.bundle)
    con = duckdb.connect()
    register_tables(con, args.processed)
    create_views(con)
    # smoke: no saved report for the sample, so expect what the model gives
    expected = None if smoke else expected_from(args.reports / "dev_metrics.json")
    weeks = run(con, bundle, presentation, train, expected)
    con.close()
    if not smoke:
        check_matches_baseline(weeks, args.reports / "baseline_metrics.json")

    report = {
        "presentation": presentation,
        "training_presentations": list(train),
        "expected_from": "smoke run (itself)" if smoke else "dev_metrics.json (2014B, lr_main)",
        "rules": {
            "psi_bands": {"moderate_from": 0.1, "major_from": 0.25},
            "outcome_alert": {
                "recall_top20_below_expected_times": RECALL_RATIO,
                "calibration_gap_above": CALIBRATION_GAP,
            },
        },
        "actions": ACTIONS,
        "weeks": weeks,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "monitoring.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out / "monitoring_report.md").write_text(markdown(report))
    print(f"Wrote {args.out / 'monitoring.json'} and monitoring_report.md")
    return 0
