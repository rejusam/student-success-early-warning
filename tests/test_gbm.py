import json

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from early_warning.calibration import calibrate, fitted_pipeline
from early_warning.features import DEMOGRAPHIC_FEATURES, WEEKS, load_week
from early_warning.gate import check_baseline_matches, check_gate
from early_warning.gbm import build_gbm_pipeline, grid_settings, settings_from, tune_week
from early_warning.reasons import shap_frame, top_reasons
from early_warning.registry import (
    default_tracking_uri,
    log_and_register,
    model_input,
    model_name,
)
from early_warning.train_gbm import main
from early_warning.views import create_views

SMALL = {"max_iter": 20, "max_leaf_nodes": 7, "min_samples_leaf": 5}


@pytest.fixture
def week8(con):
    create_views(con)
    return load_week(con, 8)


def fit_small(df):
    return build_gbm_pipeline(**SMALL).fit(df, df.at_risk.astype(int))


def test_gbm_inputs_leave_out_target_ids_and_demographics(week8):
    names = list(fit_small(week8)[:-1].get_feature_names_out())
    banned = [*DEMOGRAPHIC_FEATURES, "unregistration", "final_result", "at_risk", "id_student"]
    assert not [n for n in names if any(b in n for b in banned)]
    assert "mean_score" in names
    assert any(n.startswith("code_module_") for n in names)


def test_grid_settings_lists_every_combination():
    grid = {"a": [1, 2], "b": [3, 4, 5]}
    assert len(grid_settings(grid)) == 6
    assert {"a": 2, "b": 5} in grid_settings(grid)


def test_tune_week_picks_lowest_loss_and_tree_count(week8):
    grid = {"max_leaf_nodes": [3, 7], "min_samples_leaf": [5], "l2_regularization": [0.0]}
    out = tune_week(week8, week8, grid, max_trees=15)
    assert len(out["results"]) == 2
    assert out["chosen"]["valid_log_loss"] == min(r["valid_log_loss"] for r in out["results"])
    assert 1 <= out["chosen"]["n_trees"] <= 15
    assert settings_from(out["chosen"])["max_iter"] == out["chosen"]["n_trees"]


def test_calibration_keeps_the_order_of_students(week8):
    model = calibrate(build_gbm_pipeline(**SMALL), week8)
    calibrated = model.predict_proba(week8)[:, 1]
    raw = fitted_pipeline(model).predict_proba(week8)[:, 1]
    assert ((calibrated >= 0) & (calibrated <= 1)).all()
    # a strictly increasing curve: sorting by either score gives the same order
    assert (np.argsort(raw, kind="stable") == np.argsort(calibrated, kind="stable")).all()


def test_shap_values_add_up_to_the_model_score(week8):
    pipe = fit_small(week8)
    values, _, base = shap_frame(pipe, week8)
    raw = pipe.decision_function(week8)
    assert np.abs(values.sum(axis=1).to_numpy() + base - raw).max() < 1e-6


def test_reasons_use_only_upward_pushes_and_skip_module():
    values = pd.DataFrame(
        {"n_missed": [0.9, -0.2], "mean_score": [0.3, -0.1], "code_module_AAA": [2.0, 2.0]}
    )
    inputs = pd.DataFrame(
        {"n_missed": [2.0, 0.0], "mean_score": [np.nan, 80.0], "code_module_AAA": [1.0, 1.0]}
    )
    reasons = top_reasons(values, inputs)
    assert reasons[0] == "assessments due but not submitted: 2; no assessment score yet"
    assert reasons[1] == ""


def gate_metrics(gbm, lr, clicks):
    """metrics[week][ranking]["top_20"]["recall"], same recalls at every week."""

    def entry(value):
        return {"top_20": {"recall": value}}

    return {
        w: {"gbm": entry(gbm), "lr_main": entry(lr), "clicks_rank": entry(clicks)} for w in WEEKS
    }


def test_gate_passes_only_when_both_baselines_are_beaten():
    assert check_gate(gate_metrics(0.5, 0.4, 0.3))["passed"]
    assert not check_gate(gate_metrics(0.5, 0.6, 0.3))["passed"]
    assert not check_gate(gate_metrics(0.5, 0.4, 0.6))["passed"]


def test_gate_tie_fails_and_week_2_is_not_gated():
    assert not check_gate(gate_metrics(0.4, 0.4, 0.3))["passed"]
    metrics = gate_metrics(0.5, 0.4, 0.3)
    metrics[2]["gbm"]["top_20"]["recall"] = 0.0  # losing at week 2 does not matter
    result = check_gate(metrics)
    assert result["passed"]
    assert set(result["weeks"]) == {"4", "8", "12"}


def test_gate_fails_if_one_week_fails():
    metrics = gate_metrics(0.5, 0.4, 0.3)
    metrics[8]["gbm"]["top_20"]["recall"] = 0.35
    result = check_gate(metrics)
    assert not result["passed"]
    assert result["weeks"]["4"]["passed"] and not result["weeks"]["8"]["passed"]


def test_baseline_check_stops_on_a_changed_number():
    fresh = {4: {"lr_main": {"auc_pr": 0.5}, "clicks_rank": {"auc_pr": 0.4}}}
    saved = {"metrics": {"4": {"lr_main": {"auc_pr": 0.5}, "clicks_rank": {"auc_pr": 0.4}}}}
    check_baseline_matches(fresh, saved)
    saved["metrics"]["4"]["lr_main"]["auc_pr"] = 0.51
    with pytest.raises(ValueError):
        check_baseline_matches(fresh, saved)


def test_registry_sets_champion_or_challenger(week8, tmp_path):
    uri = default_tracking_uri(tmp_path)
    model = fit_small(week8)
    common = {
        "model": model,
        "sample": week8,
        "params": {"max_leaf_nodes": 7},
        "metrics": {"test_recall_top20": 0.5},
        "tracking_uri": uri,
        "artifact_dir": tmp_path / "artifacts",
    }
    first = log_and_register(8, passed=True, **common)
    second = log_and_register(8, passed=False, **common)
    client = MlflowClient(tracking_uri=uri)
    assert client.get_model_version_by_alias(model_name(8), "champion").version == first
    assert client.get_model_version_by_alias(model_name(8), "challenger").version == second

    # the champion loads back by alias and gives the same probabilities as the fitted model
    mlflow.set_tracking_uri(uri)
    loaded = mlflow.pyfunc.load_model(f"models:/{model_name(8)}@champion")
    rows = model_input(week8.head(20))
    served = np.asarray(loaded.predict(rows))[:, 1]
    assert np.allclose(served, model.predict_proba(rows)[:, 1])


def test_smoke_run_writes_only_to_given_dirs(tmp_path, parquet_dir):
    out, store = tmp_path / "out", tmp_path / "mlruns"
    argv = ["--processed", str(parquet_dir), "--out", str(out), "--mlflow-dir", str(store)]
    assert main([*argv, "--mode", "smoke"]) == 0
    assert sorted(p.name for p in out.iterdir()) == [
        "example_risk_list.csv",
        "gate.json",
        "gbm_calibration.json",
        "gbm_metrics.json",
        "gbm_shap_importance.csv",
        "gbm_tuning.json",
        "gbm_validation.json",
    ]
    gate = json.loads((out / "gate.json").read_text())
    assert set(gate["gate"]["weeks"]) == {"4", "8", "12"}
    assert (store / "mlflow.db").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["mlruns", "out"]


def test_smoke_refuses_the_real_output_dirs(parquet_dir):
    with pytest.raises(SystemExit):
        main(["--processed", str(parquet_dir), "--mode", "smoke"])
