"""Day 5: make the model in use ready to serve.

Steps, for each prediction week:
1. If the registry has no "champion" for the week, fit lr_main exactly as on Day 2, check its
   test metrics against reports/baseline_metrics.json, and register it as champion
   (tag role=incumbent). An existing champion is left as it is, so a second run adds nothing.
2. Download the champion's model file into a bundle folder, with a manifest.json that records
   where it came from and its SHA-256. The bundle also gets reference.json: the training
   cohort's profile of every input and of the model's scores, for drift checks (drift.py).
   Only a logistic regression pipeline can be released, because the service explains scores
   with the model's coefficients.
3. Load the bundle the way the service does, rescore the test presentation, and stop if any
   metric differs from the expected ones. So the files that ship give the reported numbers.

Nothing is written to reports/. Modes:
- test: train 2013B, 2013J, 2014B; check on 2014J against reports/baseline_metrics.json.
- smoke: the test sample, checked against a fresh fit. Checks the code only.
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import duckdb
import mlflow
import pandas as pd
import sklearn
import skops.io as sio
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from early_warning.baseline import build_pipeline
from early_warning.download import OULAD_SHA256, sha256_of
from early_warning.drift import build_reference
from early_warning.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    WEEKS,
    load_week,
    model_input,
)
from early_warning.gate import check_baseline_matches
from early_warning.ingest import register_tables
from early_warning.metrics import evaluate
from early_warning.registry import default_tracking_uri, git_commit, log_and_register, model_name
from early_warning.scoring import TRUSTED_TYPES, load_bundle, score
from early_warning.views import create_views

MODES = {
    # mode: (train presentations, test presentations)
    "test": (("2013B", "2013J", "2014B"), ("2014J",)),
    "smoke": (("2013J",), ("2013J",)),
}
CHAMPION = "champion"
MODEL_FILE = "model.skops"  # the file name MLflow uses for a skops-saved scikit-learn model


def split(df: pd.DataFrame, presentations: tuple[str, ...]) -> pd.DataFrame:
    return df[df.code_presentation.isin(presentations)].copy()


def lr_main_metrics(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[Pipeline, dict]:
    """lr_main fitted as on Day 2, and its metrics on the test rows."""
    model = build_pipeline(False).fit(train_df, train_df.at_risk.astype(int))
    scored = test_df.copy()
    scored["lr_main"] = model.predict_proba(scored)[:, 1]
    return model, evaluate(scored, "lr_main")


def champion_version(client: MlflowClient, week: int):
    """The registered version with alias champion, or None if there is none."""
    try:
        return client.get_model_version_by_alias(model_name(week), CHAMPION)
    except MlflowException:
        return None


def ensure_champion(
    week: int,
    train_df: pd.DataFrame,
    expected: dict,
    test_df: pd.DataFrame,
    tracking_uri: str,
    artifact_dir: Path,
) -> int:
    """Register lr_main as the week's champion unless one exists. Returns the champion version.

    expected is {"lr_main": metrics} for this week; registration stops if lr_main differs.
    """
    existing = champion_version(MlflowClient(tracking_uri=tracking_uri), week)
    if existing is not None:
        return int(existing.version)
    model, metrics = lr_main_metrics(train_df, test_df)
    check_baseline_matches(
        {week: {"lr_main": metrics}}, {"metrics": {str(week): expected}}, ("lr_main",)
    )
    return log_and_register(
        week,
        model,
        train_df,
        params={
            "model": "logistic regression (lr_main, Day 2)",
            "train": ",".join(sorted(train_df.code_presentation.unique())),
        },
        metrics={
            "test_recall_top20": metrics["top_20"]["recall"],
            "test_precision_top20": metrics["top_20"]["precision"],
            "test_auc_pr": metrics["auc_pr"],
            "test_roc_auc": metrics["roc_auc"],
        },
        passed=None,
        tracking_uri=tracking_uri,
        artifact_dir=artifact_dir,
        alias=CHAMPION,
        tags={"role": "incumbent", "model": "lr_main"},
    )


def check_servable(path: Path, week: int) -> None:
    """Stop unless the file holds a logistic regression pipeline the service can load.

    The service explains scores with the coefficients, so another model (for example a gradient
    boosting champion) needs SHAP reasons in the service before it can be released.
    """
    refusal = (
        f"week {week}: the champion is not a logistic regression pipeline the service can load"
    )
    if set(sio.get_untrusted_types(file=path)) - set(TRUSTED_TYPES):
        raise ValueError(refusal)
    model = sio.load(path, trusted=TRUSTED_TYPES)
    if not (isinstance(model, Pipeline) and isinstance(model[-1], LogisticRegression)):
        raise ValueError(refusal)


def export_bundle(tracking_uri: str, out_dir: Path, trains: dict[int, pd.DataFrame]) -> dict:
    """Copy each week's champion model file into out_dir, write the drift reference from the
    training cohorts (trains, one frame per week) and write manifest.json."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    out_dir.mkdir(parents=True, exist_ok=True)
    weeks, references = {}, {}
    for week in WEEKS:
        version = champion_version(client, week)
        if version is None:
            raise ValueError(f"{model_name(week)} has no champion to release")
        with tempfile.TemporaryDirectory() as tmp:
            local = mlflow.artifacts.download_artifacts(
                artifact_uri=f"models:/{model_name(week)}@{CHAMPION}", dst_path=tmp
            )
            source = Path(local) / MODEL_FILE
            check_servable(source, week)
            target = out_dir / f"week-{week}.skops"
            shutil.copyfile(source, target)
        model = sio.load(target, trusted=TRUSTED_TYPES)
        train_risk = model.predict_proba(model_input(trains[week]))[:, 1]
        references[str(week)] = build_reference(trains[week], train_risk)
        run_tags = client.get_run(version.run_id).data.tags
        weeks[str(week)] = {
            "file": target.name,
            "sha256": sha256_of(target),
            "model_name": model_name(week),
            "version": int(version.version),
            "run_id": version.run_id,
            "role": run_tags.get("role", ""),
            "trained_at_commit": run_tags.get("git_commit", ""),
        }
    reference = out_dir / "reference.json"
    reference.write_text(json.dumps({"weeks": references}, indent=2) + "\n")
    manifest = {
        "released_at_commit": git_commit(),
        "oulad_sha256": OULAD_SHA256,
        "sklearn_version": sklearn.__version__,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "weeks": weeks,
        "reference": {"file": reference.name, "sha256": sha256_of(reference)},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def check_bundle(bundle_dir: Path, tests: dict[int, pd.DataFrame], expected: dict) -> None:
    """Load the bundle as the service does and stop if any test metric differs from expected."""
    bundle = load_bundle(bundle_dir)
    for week, test_df in tests.items():
        scored = test_df.copy()
        scored["lr_main"] = score(bundle, week, test_df)["risk"]
        if evaluate(scored, "lr_main") != expected[week]["lr_main"]:
            raise ValueError(f"week {week}: the released model does not reproduce lr_main")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--reports", type=Path, default=Path("reports"), help="read only")
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--mlflow-dir", type=Path, default=Path("mlruns"))
    parser.add_argument("--tracking-uri", help="default: SQLite file in --mlflow-dir")
    parser.add_argument("--bundle", type=Path, default=Path("build/model"))
    args = parser.parse_args(argv)

    train, test = MODES[args.mode]
    smoke = args.mode == "smoke"
    if smoke and args.mlflow_dir.resolve() == Path("mlruns").resolve():
        parser.error("smoke mode must not write to mlruns/: pass --mlflow-dir")
    if smoke and args.bundle.resolve().is_relative_to(Path("reports").resolve()):
        parser.error("smoke mode must not write to reports/: pass --bundle")

    con = duckdb.connect()
    register_tables(con, args.processed)
    create_views(con)
    frames = {week: load_week(con, week) for week in WEEKS}
    con.close()
    trains = {week: split(df, train) for week, df in frames.items()}
    tests = {week: split(df, test) for week, df in frames.items()}

    if smoke:  # no saved report for the sample: expect what a fresh lr_main fit gives
        expected = {w: {"lr_main": lr_main_metrics(trains[w], tests[w])[1]} for w in WEEKS}
    else:
        saved = json.loads((args.reports / "baseline_metrics.json").read_text())["metrics"]
        expected = {w: {"lr_main": saved[str(w)]["lr_main"]} for w in WEEKS}

    args.mlflow_dir.mkdir(parents=True, exist_ok=True)
    uri = args.tracking_uri or default_tracking_uri(args.mlflow_dir)
    for week in WEEKS:
        version = ensure_champion(
            week, trains[week], expected[week], tests[week], uri, args.mlflow_dir / "artifacts"
        )
        print(f"week {week}: champion is version {version}")
    export_bundle(uri, args.bundle, trains)
    check_bundle(args.bundle, tests, expected)
    print(f"Wrote {args.bundle}: the released models reproduce the expected test metrics")
    return 0
