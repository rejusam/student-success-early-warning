"""Day 3: tune, calibrate, explain and gate a gradient boosting model per prediction week.

Modes:
- tune: fit on 2013B and 2013J, choose settings on 2014B. Writes gbm_tuning.json and
  gbm_validation.json. 2014J rows are loaded with the rest but filtered out before any use.
- test: read the chosen settings from gbm_tuning.json, refit on 2013B, 2013J and 2014B,
  calibrate, score 2014J once, apply the gate, log and register the models in MLflow.
  Exits with code 1 if the gate fails.
- smoke: both steps on the test sample with a one-setting grid. Checks the code only.
"""

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

from early_warning.baseline import build_pipeline
from early_warning.calibration import calibrate, fitted_pipeline
from early_warning.features import WEEKS, load_week
from early_warning.gate import check_baseline_matches, check_gate
from early_warning.gbm import GRID, MAX_TREES, build_gbm_pipeline, settings_from, tune_week
from early_warning.ingest import register_tables
from early_warning.metrics import calibration_metrics, evaluate, per_module_auc
from early_warning.reasons import importance, shap_frame, top_reasons
from early_warning.registry import default_tracking_uri, log_and_register
from early_warning.views import create_views

MODES = {
    # mode: (tune train, tune validation, final train, final test)
    "tune": (("2013B", "2013J"), ("2014B",), None, None),
    "test": (None, None, ("2013B", "2013J", "2014B"), ("2014J",)),
    "smoke": (("2013J",), ("2013J",), ("2013J",), ("2013J",)),
}
SMOKE_GRID = {"max_leaf_nodes": [15], "min_samples_leaf": [5], "l2_regularization": [0.0]}
PROBABILITY_MODELS = ("lr_main", "gbm_uncalibrated", "gbm")
EXAMPLE_WEEK = 8


def split(df: pd.DataFrame, presentations: tuple[str, ...]) -> pd.DataFrame:
    return df[df.code_presentation.isin(presentations)].copy()


def score_all(scored: pd.DataFrame, names: tuple[str, ...]) -> dict:
    """Day 2 metrics, per-module AUC and, for probabilities, calibration, for each ranking."""
    out = {"metrics": {}, "per_module": {}, "calibration": {}}
    for name in names:
        out["metrics"][name] = evaluate(scored, name)
        out["per_module"][name] = per_module_auc(scored, name)
        if name in PROBABILITY_MODELS:
            out["calibration"][name] = calibration_metrics(scored.at_risk, scored[name])
    return out


def add_baselines(train_df: pd.DataFrame, scored: pd.DataFrame) -> None:
    """lr_main and clicks_rank exactly as in Day 2 (baseline.run)."""
    lr = build_pipeline(False).fit(train_df, train_df.at_risk.astype(int))
    scored["lr_main"] = lr.predict_proba(scored)[:, 1]
    scored["clicks_rank"] = -scored.clicks_total  # fewest clicks = highest risk


def tune(frames: dict, train: tuple, valid: tuple, grid: dict, max_trees: int) -> dict:
    """Choose settings per week on the validation presentation, and score it with them."""
    tuning, validation = {}, {}
    for week, df in frames.items():
        train_df, valid_df = split(df, train), split(df, valid)
        tuning[week] = tune_week(train_df, valid_df, grid, max_trees)
        gbm = build_gbm_pipeline(**settings_from(tuning[week]["chosen"]))
        gbm.fit(train_df, train_df.at_risk.astype(int))
        valid_df["gbm_uncalibrated"] = gbm.predict_proba(valid_df)[:, 1]
        add_baselines(train_df, valid_df)
        validation[week] = score_all(valid_df, ("gbm_uncalibrated", "lr_main", "clicks_rank"))
    return {"tuning": tuning, "validation": validation}


def final(frames: dict, train: tuple, test: tuple, tuning: dict) -> dict:
    """Refit with the chosen settings, calibrate, score the test presentation once, explain."""
    results, models, importances, example = {}, {}, [], None
    for week, df in frames.items():
        train_df, test_df = split(df, train), split(df, test)
        chosen = tuning[str(week)]["chosen"]
        model = calibrate(build_gbm_pipeline(**settings_from(chosen)), train_df)
        test_df["gbm"] = model.predict_proba(test_df)[:, 1]
        raw = fitted_pipeline(model)
        test_df["gbm_uncalibrated"] = raw.predict_proba(test_df)[:, 1]
        add_baselines(train_df, test_df)
        results[week] = score_all(test_df, ("gbm", "gbm_uncalibrated", "lr_main", "clicks_rank"))
        models[week] = (model, chosen, train_df)

        shap_values, inputs, _ = shap_frame(raw, test_df)
        for feature, value in importance(shap_values).items():
            importances.append({"week": week, "feature": feature, "mean_abs_shap": round(value, 4)})
        if week == EXAMPLE_WEEK:
            module = sorted(test_df.code_module.unique())[0]
            in_module = test_df[test_df.code_module == module]
            top = in_module.sort_values(["gbm", "id_student"], ascending=[False, True]).head(10)
            example = top[["code_module", "code_presentation", "id_student", "gbm", "at_risk"]]
            example = example.rename(columns={"gbm": "risk"}).round({"risk": 4})
            example["reasons"] = top_reasons(shap_values.loc[top.index], inputs.loc[top.index])
    return {
        "results": results,
        "models": models,
        "importance": pd.DataFrame(importances),
        "example": example,
    }


def write_json(path: Path, content: dict) -> None:
    path.write_text(json.dumps(content, indent=2) + "\n")


def by_week(results: dict, part: str) -> dict:
    return {str(week): r[part] for week, r in results.items()}


def run_tune(frames, train, valid, out: Path, grid: dict, max_trees: int) -> dict:
    tuned = tune(frames, train, valid, grid, max_trees)
    header = {"train": list(train), "validation": list(valid), "weeks": list(frames)}
    tuning = {
        **header,
        "note": "Settings chosen by lowest log loss on the validation presentation.",
        "grid": grid,
        "max_trees": max_trees,
        "by_week": {str(w): t for w, t in tuned["tuning"].items()},
    }
    write_json(out / "gbm_tuning.json", tuning)
    write_json(
        out / "gbm_validation.json",
        {
            **header,
            "note": "Validation presentation, used for tuning. Not the test result.",
            "metrics": by_week(tuned["validation"], "metrics"),
            "per_module": by_week(tuned["validation"], "per_module"),
            "calibration": by_week(tuned["validation"], "calibration"),
        },
    )
    return tuning


def run_test(
    frames, train, test, out: Path, tuning: dict, baseline: dict | None, allow_overlap=False
) -> dict:
    if not allow_overlap:
        if set(tuning["train"] + tuning["validation"]) & set(test):
            raise ValueError("the tuning step used the test presentation")
        if set(train) & set(test):
            raise ValueError(f"train {train} and test {test} share a presentation")
    done = final(frames, train, test, tuning["by_week"])
    metrics = {week: r["metrics"] for week, r in done["results"].items()}
    if baseline is not None:
        check_baseline_matches(metrics, baseline)
    gate = check_gate(metrics)

    header = {"train": list(train), "test": list(test), "weeks": list(frames)}
    write_json(
        out / "gbm_metrics.json",
        {
            **header,
            "note": "Single run on the test presentation. gbm is the calibrated model.",
            "metrics": by_week(done["results"], "metrics"),
            "per_module": by_week(done["results"], "per_module"),
        },
    )
    write_json(
        out / "gbm_calibration.json",
        {**header, "calibration": by_week(done["results"], "calibration")},
    )
    write_json(out / "gate.json", {**header, "gate": gate})
    done["importance"].to_csv(out / "gbm_shap_importance.csv", index=False)
    if done["example"] is not None:
        done["example"].to_csv(out / "example_risk_list.csv", index=False)
    return {"gate": gate, "done": done}


def register(done: dict, gate: dict, out: Path, tracking_uri: str, artifact_dir: Path) -> None:
    files = [out / name for name in ("gate.json", "gbm_metrics.json", "gbm_tuning.json")]
    for week, (model, chosen, train_df) in done["models"].items():
        result = done["results"][week]
        gbm, cal = result["metrics"]["gbm"], result["calibration"]["gbm"]
        log_and_register(
            week,
            model,
            train_df,
            params={**chosen, "calibration": "sigmoid, leave one presentation out"},
            metrics={
                "test_recall_top20": gbm["top_20"]["recall"],
                "test_precision_top20": gbm["top_20"]["precision"],
                "test_auc_pr": gbm["auc_pr"],
                "test_roc_auc": gbm["roc_auc"],
                "test_brier": cal["brier"],
                "test_ece": cal["ece"],
            },
            passed=gate["passed"],
            tracking_uri=tracking_uri,
            artifact_dir=artifact_dir,
            report_files=[f for f in files if f.exists()],
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("reports"))
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--mlflow-dir", type=Path, default=Path("mlruns"))
    parser.add_argument("--tracking-uri", help="default: SQLite file in --mlflow-dir")
    args = parser.parse_args(argv)

    tune_train, tune_valid, train, test = MODES[args.mode]
    smoke = args.mode == "smoke"
    if smoke and Path("reports").resolve() in (args.out.resolve(), args.mlflow_dir.resolve()):
        parser.error("smoke mode must not write to reports/: pass --out and --mlflow-dir")
    if smoke and args.mlflow_dir.resolve() == Path("mlruns").resolve():
        parser.error("smoke mode must not write to mlruns/: pass --mlflow-dir")
    con = duckdb.connect()
    register_tables(con, args.processed)
    create_views(con)
    frames = {week: load_week(con, week) for week in WEEKS}
    con.close()
    args.out.mkdir(parents=True, exist_ok=True)

    if tune_train:
        grid, max_trees = (SMOKE_GRID, 20) if smoke else (GRID, MAX_TREES)
        tuning = run_tune(frames, tune_train, tune_valid, args.out, grid, max_trees)
        print(f"Wrote {args.out / 'gbm_tuning.json'}")
    if not train:
        return 0

    if not smoke:
        tuning = json.loads((args.out / "gbm_tuning.json").read_text())
    baseline = None if smoke else json.loads((args.out / "baseline_metrics.json").read_text())
    tested = run_test(frames, train, test, args.out, tuning, baseline, allow_overlap=smoke)

    args.mlflow_dir.mkdir(parents=True, exist_ok=True)
    uri = args.tracking_uri or default_tracking_uri(args.mlflow_dir)
    register(tested["done"], tested["gate"], args.out, uri, args.mlflow_dir / "artifacts")
    passed = tested["gate"]["passed"]
    print(f"Wrote {args.out / 'gate.json'}: gate {'passed' if passed else 'failed'}")
    return 0 if passed or smoke else 1
