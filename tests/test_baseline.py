import json

import pytest

from early_warning.baseline import build_pipeline, main, run
from early_warning.features import DEMOGRAPHIC_FEATURES, WEEKS, load_week
from early_warning.views import create_views


@pytest.fixture
def fcon(con):
    create_views(con)
    return con


def input_names(pipe):
    """Names of the columns the logistic regression actually sees."""
    return list(pipe[:-1].get_feature_names_out())


def test_pipeline_ignores_demographics_in_main(fcon):
    df = load_week(fcon, 4)
    names = input_names(build_pipeline(False).fit(df, df.at_risk.astype(int)))
    assert not [n for n in names if any(d in n for d in DEMOGRAPHIC_FEATURES)]
    assert not [n for n in names if "unregistration" in n or "final_result" in n]
    assert not [n for n in names if "at_risk" in n or "id_student" in n]


def test_pipeline_with_demographics_uses_them(fcon):
    df = load_week(fcon, 4)
    names = input_names(build_pipeline(True).fit(df, df.at_risk.astype(int)))
    for column in DEMOGRAPHIC_FEATURES:
        assert any(column in n for n in names)


def test_run_smoke_on_fixture(fcon):
    # The fixture has only 2013J, so the smoke run trains and scores the same presentation.
    out = run(fcon, ("2013J",), ("2013J",), allow_overlap=True)
    assert set(out["metrics"]) == set(WEEKS)
    for week in WEEKS:
        assert set(out["metrics"][week]) == {"lr_main", "lr_with_demographics", "clicks_rank"}
        assert out["cohort"][week]["2013J"]["n"] > 0
    assert set(out["coefficients"].columns) == {"week", "feature", "coefficient"}


def test_run_reports_registrations_and_exclusions(fcon):
    out = run(fcon, ("2013J",), ("2013J",), allow_overlap=True)
    assert out["registrations"] == {"2013J": 80}
    for week in WEEKS:
        counts = out["cohort"][week]["2013J"]
        # every registration is either in the cohort or excluded for one of the two reasons
        assert counts["n"] + counts["left_by_cutoff"] + counts["not_yet_registered"] == 80


def test_run_rejects_overlap_between_train_and_real_test(fcon):
    with pytest.raises(ValueError):
        run(fcon, ("2013B", "2014J"), ("2014J",))


def test_main_writes_only_to_out_dir(tmp_path, parquet_dir):
    assert main(["--processed", str(parquet_dir), "--out", str(tmp_path), "--mode", "smoke"]) == 0
    result = json.loads((tmp_path / "smoke_metrics.json").read_text())
    assert result["train"] == ["2013J"]
    assert result["test"] == ["2013J"]
    assert set(result["metrics"]) == {str(w) for w in WEEKS}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["smoke_metrics.json"]
