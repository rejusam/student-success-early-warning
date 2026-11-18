import json

import duckdb
import mlflow
import pandas as pd
import pytest

from early_warning.audit import (
    blank_small_cells,
    check_matches,
    flag_counts,
    main,
    markdown,
    score_week,
    summarise,
)
from early_warning.features import DEMOGRAPHIC_FEATURES, load_week
from early_warning.ingest import register_tables
from early_warning.registry import default_tracking_uri
from early_warning.train_gbm import main as train_gbm_main
from early_warning.views import create_views

AUDIT_FILES = {
    "fairness_groups.csv",
    "fairness_intersectional.csv",
    "fairness_summary.json",
    "fairness_report.md",
    "model_gap_bootstrap.json",
}


@pytest.fixture(scope="module")
def smoke_dirs(parquet_dir, tmp_path_factory):
    """Output and MLflow store of a Day 3 smoke run, which the audit needs."""
    root = tmp_path_factory.mktemp("audit")
    out, store = root / "out", root / "mlruns"
    argv = ["--processed", str(parquet_dir), "--out", str(out), "--mlflow-dir", str(store)]
    assert train_gbm_main([*argv, "--mode", "smoke"]) == 0
    return argv, out


def test_smoke_audit_writes_only_the_audit_files(smoke_dirs):
    argv, out = smoke_dirs
    before = {p.name for p in out.iterdir()}
    assert main([*argv, "--mode", "smoke"]) == 0
    assert {p.name for p in out.iterdir()} - before == AUDIT_FILES

    groups = pd.read_csv(out / "fairness_groups.csv")
    assert set(groups.model) == {"lr_main", "gbm", "lr_with_demographics"}
    assert set(groups.week) == {2, 4, 8, 12}
    assert "disability_x_imd" not in set(groups.attribute)
    assert "All students" in set(groups.group)
    assert set(pd.read_csv(out / "fairness_intersectional.csv").attribute) == {"disability_x_imd"}
    summary = json.loads((out / "fairness_summary.json").read_text())
    assert summary["settings"]["n_boot"] == 20
    assert set(summary["flag_counts"]["8"]) == {"lr_main", "gbm", "lr_with_demographics"}
    gaps = json.loads((out / "model_gap_bootstrap.json").read_text())["by_week"]
    for gap in gaps.values():
        assert gap["difference_lo"] <= gap["difference_hi"]


def test_smoke_refuses_reports_and_mlruns(parquet_dir):
    with pytest.raises(SystemExit):
        main(["--processed", str(parquet_dir), "--out", "reports", "--mode", "smoke"])
    with pytest.raises(SystemExit):
        main(["--processed", str(parquet_dir), "--out", "elsewhere", "--mode", "smoke"])


def test_check_matches_stops_on_a_changed_number():
    saved = {"metrics": {"8": {"gbm": {"auc_pr": 0.7}}}}
    check_matches({8: {"gbm": {"auc_pr": 0.7}}}, saved, ["gbm"], "gbm_metrics.json")
    with pytest.raises(ValueError, match="gbm_metrics.json"):
        check_matches({8: {"gbm": {"auc_pr": 0.71}}}, saved, ["gbm"], "gbm_metrics.json")


def test_served_models_get_no_demographic_input(smoke_dirs, parquet_dir):
    _, out = smoke_dirs
    con = duckdb.connect()
    register_tables(con, parquet_dir)
    create_views(con)
    df = load_week(con, 8)
    con.close()
    mlflow.set_tracking_uri(default_tracking_uri(out.parent / "mlruns"))
    passed = json.loads((out / "gate.json").read_text())["gate"]["passed"]
    alias = "champion" if passed else "challenger"
    scored = score_week(df, ("2013J",), ("2013J",), 8, alias)
    # Shuffling every demographic column must not move lr_main or gbm scores.
    shuffled = df.copy()
    for column in DEMOGRAPHIC_FEATURES:
        shuffled[column] = shuffled[column].sample(frac=1, random_state=1).to_numpy()
    rescored = score_week(shuffled, ("2013J",), ("2013J",), 8, alias)
    pd.testing.assert_series_equal(scored.lr_main, rescored.lr_main)
    pd.testing.assert_series_equal(scored.gbm, rescored.gbm)


def audit_table() -> pd.DataFrame:
    """A hand-made audit table: the whole cohort, two gender groups, two intersection cells."""
    rows = [
        ("all", "All students", 1000, 400, 0.4, "calibration", False),
        ("gender", "F", 400, 150, 0.3, "recall gap", False),
        ("gender", "M", 600, 250, 0.45, "", False),
        ("disability_x_imd", "cell big", 900, 380, 0.5, "recall gap; calibration", False),
        ("disability_x_imd", "cell small", 20, 6, 0.9, "", True),
    ]
    frame = pd.DataFrame(
        rows, columns=["attribute", "group", "n", "n_at_risk", "recall", "flag", "small"]
    )
    frame["model"] = "lr_main"
    frame["reference"] = frame.groupby("attribute")["group"].transform("first")
    for column in [
        "at_risk_rate",
        "flag_rate",
        "precision",
        "mean_risk",
        "calibration_gap",
        "roc_auc",
        "recall_diff",
        "recall_ratio",
    ]:
        frame[column] = 0.1
    for column in [
        "recall",
        "recall_diff",
        "recall_ratio",
        "precision",
        "flag_rate",
        "mean_risk",
        "calibration_gap",
        "roc_auc",
    ]:
        frame[f"{column}_lo"], frame[f"{column}_hi"] = 0.0, 0.2
    return frame


def test_small_intersection_cells_keep_counts_only():
    out = blank_small_cells(audit_table())
    small = out[out.group == "cell small"].iloc[0]
    assert small.n == 20 and small.n_at_risk == 6
    assert pd.isna(small.recall) and pd.isna(small.recall_lo) and pd.isna(small.at_risk_rate)
    big = out[out.group == "cell big"].iloc[0]
    assert big.recall == 0.5
    # single-attribute small groups keep their numbers (they are labelled, not hidden)
    assert out[out.group == "F"].recall.iloc[0] == 0.3

    table = blank_small_cells(audit_table())
    table = pd.concat([table.assign(model=m) for m in ["lr_main", "gbm", "lr_with_demographics"]])
    gap = {
        "gbm_recall": 0.4,
        "lr_main_recall": 0.41,
        "difference": -0.01,
        "difference_lo": -0.0003,
        "difference_hi": 0.02,
        "share_of_resamples_gbm_ahead": 0.3,
    }
    text = markdown({12: table}, {12: gap}, {"n_boot": 10, "min_n": 100, "min_at_risk": 30})
    small_line = next(line for line in text.splitlines() if "cell small" in line)
    assert "0.900" not in small_line and "n/a" in small_line
    assert "-0.0003" in text  # model gap printed with four decimals


def test_flag_counts_leave_out_the_whole_cohort_row():
    counts = flag_counts(audit_table())["lr_main"]
    assert counts == {"groups_audited": 4, "flagged": 2, "recall_gap": 2, "calibration": 1}


def test_summarise_ignores_small_groups_for_range_and_lowest():
    summary = summarise(audit_table())["lr_main"]
    assert summary["gender"]["lowest_recall_group"] == "F"
    assert summary["gender"]["recall_min"] == 0.3 and summary["gender"]["recall_max"] == 0.45
    assert summary["disability_x_imd"]["recall_max"] == 0.5  # 0.9 is a small cell
    assert summary["disability_x_imd"]["small_groups"] == 1
    assert summary["gender"]["flagged"] == [{"group": "F", "flag": "recall gap"}]
