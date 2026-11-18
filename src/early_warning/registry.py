"""Log a weekly model and its evidence to MLflow, register it, and set its alias.

The tracking store is a local SQLite file by default (mlruns/mlflow.db, not in git). A hosted
MLflow server only needs a different tracking URI. Each week's model is registered as
early-warning-week-<k>. The alias "champion" marks the version in use, the one that is
served; "challenger" marks one that did not pass the gate. lr_main, the model in use from
Day 2, is registered as champion by release.py (tag role=incumbent).
"""

import subprocess
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from early_warning.download import OULAD_SHA256
from early_warning.features import model_input

EXPERIMENT = "early-warning"

# MLflow saves scikit-learn models with skops, which refuses to load any type not on an
# allow-list (safer than pickle). These are the types inside our own models. Only models this
# project trained are loaded, so they are trusted here and nowhere else.
TRUSTED_TYPES = [
    "early_warning.gbm.to_float",
    "numpy.dtype",  # stored by the logistic regression pipeline (lr_main)
    "sklearn.calibration._CalibratedClassifier",
    "sklearn.calibration._SigmoidCalibration",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
]


def model_name(week: int) -> str:
    return f"early-warning-week-{week}"


def default_tracking_uri(directory: Path) -> str:
    return f"sqlite:///{directory.resolve() / 'mlflow.db'}"


def git_commit() -> str:
    """Current commit, with '-dirty' if there are uncommitted changes. 'unknown' outside git."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return commit + ("-dirty" if dirty else "")


def log_and_register(
    week: int,
    model,
    sample: pd.DataFrame,
    params: dict,
    metrics: dict,
    passed: bool | None,
    tracking_uri: str,
    artifact_dir: Path,
    report_files: list[Path] = (),
    alias: str | None = None,
    tags: dict | None = None,
) -> int:
    """Log one weekly run, register the model and set its alias. Returns the version number.

    The alias is "champion" if the gate passed, "challenger" if not, unless one is given.
    passed=None means no gate applies (the incumbent model), so no gate result is logged.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    if client.get_experiment_by_name(EXPERIMENT) is None:
        client.create_experiment(EXPERIMENT, artifact_location=artifact_dir.resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT)

    example = model_input(sample.head(5))
    signature = infer_signature(example, model.predict_proba(example)[:, 1])
    with mlflow.start_run(run_name=f"week-{week}"):
        run_tags = {"week": week, "git_commit": git_commit(), "oulad_sha256": OULAD_SHA256}
        mlflow.set_tags({**run_tags, **(tags or {})})
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if passed is not None:
            mlflow.log_metric("gate_passed", int(passed))
        for path in report_files:
            mlflow.log_artifact(str(path), artifact_path="reports")
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=example,
            skops_trusted_types=TRUSTED_TYPES,
            pyfunc_predict_fn="predict_proba",  # serving returns probabilities, not 0/1
            registered_model_name=model_name(week),
        )
    version = int(info.registered_model_version)
    alias = alias or ("champion" if passed else "challenger")
    client.set_registered_model_alias(model_name(week), alias, str(version))
    return version
