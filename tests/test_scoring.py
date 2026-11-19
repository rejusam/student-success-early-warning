import math
import shutil

import numpy as np
import pytest

from early_warning.features import DEMOGRAPHIC_FEATURES, WEEKS, model_input
from early_warning.scoring import LIST_COLUMNS, load_bundle, predict_risk, risk_lists, score


def test_bundle_loads_every_week(bundle_dir):
    bundle = load_bundle(bundle_dir)
    assert sorted(bundle.models) == sorted(WEEKS)


def test_changed_model_file_is_refused(bundle_dir, tmp_path):
    copy = tmp_path / "bundle"
    shutil.copytree(bundle_dir, copy)
    with open(copy / "week-8.skops", "ab") as f:
        f.write(b"x")
    with pytest.raises(ValueError, match="SHA-256"):
        load_bundle(copy)


def test_changed_drift_reference_is_refused(bundle_dir, tmp_path):
    copy = tmp_path / "bundle"
    shutil.copytree(bundle_dir, copy)
    with open(copy / "reference.json", "a") as f:
        f.write(" ")
    with pytest.raises(ValueError, match="SHA-256"):
        load_bundle(copy)


def test_predict_risk_matches_score(bundle_dir, frames):
    bundle = load_bundle(bundle_dir)
    df = frames[4]
    assert np.allclose(predict_risk(bundle, 4, df), score(bundle, 4, df)["risk"])


def test_missing_week_is_refused(bundle_dir, tmp_path):
    copy = tmp_path / "bundle"
    shutil.copytree(bundle_dir, copy)
    (copy / "manifest.json").write_text(
        (copy / "manifest.json").read_text().replace('"12":', '"13":')
    )
    with pytest.raises(ValueError, match="week 12"):
        load_bundle(copy)


def test_score_matches_the_pipeline(bundle_dir, frames):
    bundle = load_bundle(bundle_dir)
    df = frames[8]
    expected = bundle.models[8].predict_proba(model_input(df))[:, 1]
    assert np.allclose(score(bundle, 8, df)["risk"], expected)


def test_risk_list_is_top_20_percent_of_each_module(bundle_dir, frames):
    bundle = load_bundle(bundle_dir)
    df = frames[12]
    lists = risk_lists(bundle, 12, df)
    assert list(lists.columns) == LIST_COLUMNS
    for module, n in df.groupby("code_module").size().items():
        part = lists[lists.code_module == module]
        assert len(part) == math.ceil(0.2 * n)
        assert list(part["rank"]) == list(range(1, len(part) + 1))
        assert part.risk.is_monotonic_decreasing


def test_risk_list_has_no_outcome_or_demographics(bundle_dir, frames):
    lists = risk_lists(load_bundle(bundle_dir), 4, frames[4])
    assert not set(lists.columns) & {"at_risk", "final_result", *DEMOGRAPHIC_FEATURES}
