import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from early_warning.fairness import (
    INTERSECTION,
    add_gaps,
    add_groups,
    audit_once,
    bootstrap,
    finish,
    group_table,
    intervals,
    reference_group,
    resample,
    review_flag,
)
from early_warning.features import DEMOGRAPHIC_FEATURES
from early_warning.metrics import flag_top_k


def small_frame() -> pd.DataFrame:
    """Two modules of 10 students. Top 20% = 2 students per module.

    Module A: highest scores go to students 1 (group x, at risk) and 2 (group y, not at risk).
    Module B: highest scores go to students 11 (group y, at risk) and 12 (group y, at risk).
    """
    rows = []
    for module, offset in (("A", 0), ("B", 10)):
        for i in range(10):
            rows.append(
                {
                    "code_module": module,
                    "code_presentation": "2014J",
                    "id_student": offset + i + 1,
                    "score": 1.0 - i / 10,
                    "at_risk": False,
                    "group": "x",
                }
            )
    df = pd.DataFrame(rows)
    df.loc[df.id_student.isin([2, 11, 12, 13, 14]), "group"] = "y"
    df.loc[df.id_student.isin([1, 5, 11, 12, 13]), "at_risk"] = True
    return df


def test_group_measures_match_hand_counts():
    df = small_frame()
    flagged = flag_top_k(df, "score", 0.2)
    table = group_table(df, "score", flagged, "group")
    # x: 15 students, at risk 1 and 5; flagged 1 -> recall 1/2, precision 1/1
    # y: 5 students, at risk 11, 12, 13; flagged 2, 11, 12 -> recall 2/3, precision 2/3
    assert table.loc["x", "n"] == 15 and table.loc["y", "n"] == 5
    assert table.loc["x", "recall"] == pytest.approx(0.5)
    assert table.loc["x", "precision"] == pytest.approx(1.0)
    assert table.loc["y", "recall"] == pytest.approx(2 / 3)
    assert table.loc["y", "precision"] == pytest.approx(2 / 3)
    assert table.loc["y", "flag_rate"] == pytest.approx(3 / 5)
    assert table.loc["x", "at_risk_rate"] == pytest.approx(2 / 15)
    assert table.loc["x", "calibration_gap"] == pytest.approx(
        df[df.group == "x"].score.mean() - 2 / 15
    )


def test_flags_come_from_module_list_not_per_group():
    # Group y has 5 students. A list built for y alone would flag 1 per module; the module
    # list flags 3 of them (students 2, 11 and 12).
    df = small_frame()
    table = group_table(df, "score", flag_top_k(df, "score", 0.2), "group")
    assert table.loc["y", "flag_rate"] * table.loc["y", "n"] == 3


def test_roc_auc_matches_sklearn():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "group": rng.choice(["a", "b"], 300),
            "at_risk": rng.random(300) < 0.4,
            "score": rng.integers(0, 20, 300) / 20,  # ties on purpose
        }
    )
    table = group_table(df, "score", pd.Series(False, index=df.index), "group")
    for name, part in df.groupby("group"):
        assert table.loc[name, "roc_auc"] == pytest.approx(roc_auc_score(part.at_risk, part.score))


def test_group_without_at_risk_students_gives_nan_not_error():
    df = small_frame()
    df.loc[df.group == "y", "at_risk"] = False
    table = group_table(df, "score", flag_top_k(df, "score", 0.2), "group")
    assert np.isnan(table.loc["y", "recall"]) and np.isnan(table.loc["y", "roc_auc"])


def test_reference_is_largest_group_and_gaps():
    df = small_frame()
    table = group_table(df, "score", flag_top_k(df, "score", 0.2), "group")
    assert reference_group(table) == "x"
    gaps = add_gaps(table, "x")
    assert gaps.loc["y", "recall_diff"] == pytest.approx(2 / 3 - 0.5)
    assert gaps.loc["y", "recall_ratio"] == pytest.approx((2 / 3) / 0.5)
    assert gaps.loc["x", "recall_ratio"] == 1


def test_add_groups_unknown_imd_and_intersection():
    df = pd.DataFrame(
        {
            "gender": ["F", "M", "F"],
            "age_band": ["0-35"] * 3,
            "region": ["Wales"] * 3,
            "highest_education": ["HE Qualification"] * 3,
            "imd_band": ["10-20", None, "90-100%"],
            "disability": ["Y", "N", "N"],
        }
    )
    out = add_groups(df)
    assert list(out.imd_band) == ["10-20", "Unknown", "90-100%"]
    assert list(out[INTERSECTION]) == [
        "disability Y, IMD 0-30%",
        "disability N, IMD Unknown",
        "disability N, IMD 70-100%",
    ]
    assert set(out["all"]) == {"All students"}
    assert df.imd_band.isna().sum() == 1  # input left as it was


def test_add_groups_rejects_unknown_imd_label():
    df = pd.DataFrame({c: ["a"] for c in DEMOGRAPHIC_FEATURES})
    with pytest.raises(ValueError, match="IMD"):
        add_groups(df)


def test_resample_keeps_module_sizes_and_unique_index():
    df = small_frame().iloc[:15]  # 10 in A, 5 in B
    out = resample(df, np.random.default_rng(0))
    assert out.groupby("code_module").size().to_dict() == {"A": 10, "B": 5}
    assert out.index.is_unique


def boot_frame() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = 200
    at_risk = rng.random(n) < 0.4
    return pd.DataFrame(
        {
            "code_module": np.repeat(["A", "B"], n // 2),
            "code_presentation": "2014J",
            "id_student": np.arange(n),
            "group": rng.choice(["a", "b"], n),
            "at_risk": at_risk,
            "m1": at_risk + rng.normal(0, 0.8, n),
            "m2": at_risk + rng.normal(0, 0.8, n),
        }
    )


def test_bootstrap_same_seed_same_result_and_range_holds_point():
    df = boot_frame()
    refs = {"group": "a"}
    first, overall = bootstrap(df, ["m1", "m2"], ["group"], refs, n_boot=30, seed=5)
    second, _ = bootstrap(df, ["m1", "m2"], ["group"], refs, n_boot=30, seed=5)
    pd.testing.assert_frame_equal(first, second)
    assert list(overall.columns) == ["replicate", "m1", "m2"]

    point, _ = audit_once(df, ["m1", "m2"], ["group"], refs)
    joined = point.merge(intervals(first), on=["model", "attribute", "group"])
    assert (joined.recall_lo <= joined.recall).all() and (joined.recall <= joined.recall_hi).all()


def test_bootstrap_does_not_depend_on_row_order():
    df = boot_frame()
    refs = {"group": "a"}
    first, overall = bootstrap(df, ["m1"], ["group"], refs, n_boot=10, seed=2)
    shuffled = df.sample(frac=1, random_state=7)
    second, overall_shuffled = bootstrap(shuffled, ["m1"], ["group"], refs, n_boot=10, seed=2)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(overall, overall_shuffled)


def flag_row(**changes) -> dict:
    row = {
        "small": False,
        "recall_ratio": 1.0,
        "recall_diff_lo": -0.05,
        "recall_diff_hi": 0.05,
        "calibration_gap": 0.0,
        "calibration_gap_lo": -0.02,
        "calibration_gap_hi": 0.02,
    }
    return {**row, **changes}


def test_review_flag_rules():
    assert review_flag(flag_row()) == ""
    recall_gap = {"recall_ratio": 0.5, "recall_diff_lo": -0.3, "recall_diff_hi": -0.1}
    assert review_flag(flag_row(**recall_gap)) == "recall gap"
    # same ratio, but the range includes 0: no flag
    assert review_flag(flag_row(**{**recall_gap, "recall_diff_hi": 0.05})) == ""
    # ratio within 0.8-1.25 even though the range excludes 0: no flag
    assert review_flag(flag_row(recall_ratio=0.9, recall_diff_lo=-0.1, recall_diff_hi=-0.01)) == ""
    calibration = {"calibration_gap": 0.08, "calibration_gap_lo": 0.06, "calibration_gap_hi": 0.1}
    assert review_flag(flag_row(**calibration)) == "calibration"
    assert review_flag(flag_row(**recall_gap, **calibration)) == "recall gap; calibration"
    # small groups are never flagged
    assert review_flag(flag_row(small=True, **recall_gap, **calibration)) == ""
    # NaN ends (no at-risk students in some resamples) never flag
    assert review_flag(flag_row(recall_ratio=0.5, recall_diff_lo=np.nan, recall_diff_hi=-0.1)) == ""


def test_finish_labels_small_groups():
    df = boot_frame()
    refs = {"group": "a"}
    point, _ = audit_once(df, ["m1"], ["group"], refs)
    reps, _ = bootstrap(df, ["m1"], ["group"], refs, n_boot=10, seed=0)
    out = finish(point, intervals(reps), min_n=1000, min_at_risk=1)
    assert out.small.all() and (out.flag == "").all()
