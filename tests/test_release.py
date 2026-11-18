import json

import pytest
from mlflow.tracking import MlflowClient

from early_warning.download import sha256_of
from early_warning.features import WEEKS, load_week
from early_warning.gbm import build_gbm_pipeline
from early_warning.registry import default_tracking_uri, log_and_register, model_name
from early_warning.release import export_bundle, main
from early_warning.views import create_views


def release(parquet_dir, tmp_path):
    return main(
        [
            "--mode", "smoke",
            "--processed", str(parquet_dir),
            "--mlflow-dir", str(tmp_path / "ml"),
            "--bundle", str(tmp_path / "bundle"),
        ]
    )  # fmt: skip


def test_release_registers_lr_main_as_champion_and_writes_bundle(parquet_dir, tmp_path):
    assert release(parquet_dir, tmp_path) == 0
    manifest = json.loads((tmp_path / "bundle" / "manifest.json").read_text())
    assert sorted(manifest["weeks"]) == sorted(str(w) for w in WEEKS)
    for entry in manifest["weeks"].values():
        assert entry["role"] == "incumbent"
        assert sha256_of(tmp_path / "bundle" / entry["file"]) == entry["sha256"]
    client = MlflowClient(tracking_uri=default_tracking_uri(tmp_path / "ml"))
    version = client.get_model_version_by_alias(model_name(8), "champion")
    assert client.get_run(version.run_id).data.tags["model"] == "lr_main"


def test_second_release_adds_no_version(parquet_dir, tmp_path):
    release(parquet_dir, tmp_path)
    release(parquet_dir, tmp_path)
    client = MlflowClient(tracking_uri=default_tracking_uri(tmp_path / "ml"))
    for week in WEEKS:
        assert len(client.search_model_versions(f"name='{model_name(week)}'")) == 1


def test_gradient_boosting_champion_is_refused(con, tmp_path):
    create_views(con)
    df = load_week(con, 2)
    model = build_gbm_pipeline(max_iter=5, max_leaf_nodes=3, min_samples_leaf=5)
    model.fit(df, df.at_risk.astype(int))
    uri = default_tracking_uri(tmp_path)
    log_and_register(2, model, df, {}, {}, passed=True, tracking_uri=uri, artifact_dir=tmp_path)
    with pytest.raises(ValueError, match="not a logistic regression"):
        export_bundle(uri, tmp_path / "bundle")


def test_smoke_refuses_the_real_registry(parquet_dir):
    with pytest.raises(SystemExit):
        main(["--mode", "smoke", "--processed", str(parquet_dir)])
