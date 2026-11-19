import math

import numpy as np
import pandas as pd
import pytest

from early_warning.drift import (
    EPS,
    band,
    bin_shares,
    build_reference,
    categorical_profile,
    compare,
    numeric_profile,
    psi,
)
from early_warning.features import NUMERIC_FEATURES


def cohort(n=1000, seed=0, shift=0.0, module="AAA"):
    """A made-up cohort with every model input, for drift tests only."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({name: rng.poisson(5, n).astype(float) for name in NUMERIC_FEATURES})
    df["clicks_total"] = rng.normal(100 + shift, 20, n)
    df.loc[: n // 10, "mean_score"] = np.nan  # some students have no score yet
    df["code_module"] = rng.choice([module, "BBB"], n)
    risk = pd.Series(rng.uniform(0, 1, n))
    return df, risk


def test_psi_is_zero_for_the_same_shares():
    assert psi([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]) == 0


def test_psi_matches_the_formula_and_stays_finite_for_empty_bins():
    assert psi([0.5, 0.5], [0.25, 0.75]) == pytest.approx(
        (0.25 - 0.5) * math.log(0.25 / 0.5) + (0.75 - 0.5) * math.log(0.75 / 0.5)
    )
    assert math.isfinite(psi([0.5, 0.5, 0.0], [0.5, 0.0, 0.5]))
    assert psi([0.5, 0.5, 0.0], [0.5, 0.0, 0.5]) == pytest.approx(
        2 * (0.5 - EPS) * math.log(0.5 / EPS)
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.0, "stable"), (0.0999, "stable"), (0.1, "moderate"), (0.2499, "moderate"),
     (0.25, "major"), (3.0, "major")],
)  # fmt: skip
def test_bands(value, expected):
    assert band(value) == expected


def test_numeric_bins_use_reference_deciles_and_a_missing_bin():
    ref = pd.Series([*range(1, 101), None], dtype="float64")
    profile = numeric_profile(ref)
    assert len(profile["edges"]) == 9
    assert len(profile["shares"]) == 11  # 10 bins + missing
    assert profile["shares"][-1] == pytest.approx(1 / 101)
    assert sum(profile["shares"]) == pytest.approx(1)
    assert bin_shares(profile, ref) == pytest.approx(profile["shares"])


def test_values_outside_the_reference_range_fall_in_the_end_bins():
    profile = numeric_profile(pd.Series(range(1, 101), dtype="float64"))
    shares = bin_shares(profile, pd.Series([-50.0, 500.0]))
    assert shares[0] == 0.5 and shares[-2] == 0.5


def test_constant_input_gets_one_edge_and_no_drift():
    values = pd.Series([0.0] * 50)
    profile = numeric_profile(values)
    assert profile["edges"] == [0.0]
    assert psi(profile["shares"], bin_shares(profile, values)) == 0


def test_all_missing_reference_has_only_the_missing_bin():
    profile = numeric_profile(pd.Series([None, None], dtype="float64"))
    assert profile["edges"] == []
    assert profile["shares"] == [0.0, 1.0]
    assert bin_shares(profile, pd.Series([1.0, None])) == [0.5, 0.5]


def test_unseen_module_goes_to_the_unseen_bin():
    profile = categorical_profile(pd.Series(["AAA", "BBB", "BBB"]))
    assert profile["categories"] == ["AAA", "BBB"]
    assert bin_shares(profile, pd.Series(["AAA", "ZZZ"])) == [0.5, 0.0, 0.5]


def test_same_cohort_is_ok():
    df, risk = cohort()
    result = compare(build_reference(df, risk), df, risk)
    assert result["status"] == "ok"
    assert result["score"]["psi"] == 0
    assert all(entry["psi"] == 0 for entry in result["inputs"].values())


def test_shifted_input_is_an_alert_and_named():
    ref_df, ref_risk = cohort(seed=0)
    cur_df, cur_risk = cohort(seed=1, shift=40)
    result = compare(build_reference(ref_df, ref_risk), cur_df, cur_risk)
    assert result["inputs"]["clicks_total"]["band"] == "major"
    assert result["status"] == "alert"
    assert result["largest_input"] == "clicks_total"
    assert result["n_reference"] == result["n_current"] == 1000


def test_unseen_module_in_current_cohort_is_counted_not_an_error():
    ref_df, ref_risk = cohort()
    cur_df, cur_risk = cohort(seed=1, module="ZZZ")
    result = compare(build_reference(ref_df, ref_risk), cur_df, cur_risk)
    assert result["inputs"]["code_module"]["band"] == "major"


def test_empty_cohort_is_refused():
    df, risk = cohort()
    with pytest.raises(ValueError, match="empty cohort"):
        compare(build_reference(df, risk), df.iloc[:0], risk.iloc[:0])


def test_reference_holds_no_student_rows():
    df, risk = cohort()
    df["id_student"] = range(len(df))
    reference = build_reference(df, risk)
    assert set(reference["inputs"]) == {*NUMERIC_FEATURES, "code_module"}
    assert "id_student" not in str(reference)
