import pandas as pd

from early_warning.metrics import evaluate, flag_top_k, top_k_metrics


def frame():
    """Two module presentations. AAA: 5 students, 3 at risk. BBB: 3 students, 1 at risk."""
    return pd.DataFrame(
        {
            "code_module": ["AAA"] * 5 + ["BBB"] * 3,
            "code_presentation": ["2014J"] * 8,
            "id_student": [1, 2, 3, 4, 5, 6, 7, 8],
            "score": [0.9, 0.8, 0.7, 0.1, 0.1, 0.5, 0.5, 0.2],
            "at_risk": [True, False, True, False, True, False, True, False],
        }
    )


def test_k_rounds_up_per_group():
    df = frame()
    # AAA: ceil(0.2 * 5) = 1 -> id 1. BBB: ceil(0.2 * 3) = 1 -> ids 6 and 7 tie, lower id wins.
    assert set(df[flag_top_k(df, "score", 0.2)].id_student) == {1, 6}


def test_pooled_recall_precision():
    # Flagged: 1 (at risk) and 6 (not). 4 at risk in total.
    assert top_k_metrics(frame(), "score", 0.2) == {
        "recall": 0.25,
        "precision": 0.5,
        "n_flagged": 2,
    }


def test_ties_broken_by_id():
    df = frame()
    df["score"] = 0.0
    assert set(df[flag_top_k(df, "score", 0.2)].id_student) == {1, 6}


def test_flags_follow_original_row_order():
    df = frame().sample(frac=1, random_state=0)
    flagged = df[flag_top_k(df, "score", 0.2)]
    assert set(flagged.id_student) == {1, 6}


def test_group_without_positives():
    df = frame()
    df.loc[df.code_module == "BBB", "at_risk"] = False
    assert top_k_metrics(df, "score", 0.2)["recall"] == round(1 / 3, 4)


def test_no_positives_at_all():
    df = frame()
    df["at_risk"] = False
    assert top_k_metrics(df, "score", 0.2)["recall"] is None


def test_evaluate_keys():
    result = evaluate(frame(), "score")
    assert set(result) == {"n", "base_rate", "auc_pr", "roc_auc", "top_10", "top_20", "top_30"}
    assert result["n"] == 8
    assert result["base_rate"] == 0.5


def test_per_module_auc_skips_single_class_groups():
    from early_warning.metrics import per_module_auc

    df = frame()
    df.loc[df.code_module == "BBB", "at_risk"] = False  # BBB now has no one at risk
    result = per_module_auc(df, "score")
    assert result["n_modules"] == 1
    # AAA scores 0.9, 0.8, 0.7, 0.1, 0.1 with at_risk T, F, T, F, T. Positive-negative pairs
    # ranked right: (1,2) (1,4) (3,4) = 3, tie (5,4) = 0.5, wrong (5,2) = 0. 3.5 / 6.
    assert result["roc_auc_module_mean"] == round(3.5 / 6, 4)


def test_calibration_metrics_perfect_and_off():
    from early_warning.metrics import calibration_metrics

    y = pd.Series([0, 0, 1, 1] * 25)
    perfect = calibration_metrics(y, pd.Series([0.5] * 100), n_bins=1)
    assert perfect["ece"] == 0.0
    assert perfect["brier"] == 0.25
    off = calibration_metrics(y, pd.Series([0.9] * 100), n_bins=1)
    assert off["ece"] == 0.4
    assert off["bins"] == [{"n": 100, "mean_prob": 0.9, "rate": 0.5}]


def test_calibration_bins_have_equal_size():
    from early_warning.metrics import calibration_table

    prob = pd.Series([i / 100 for i in range(100)])
    table = calibration_table(pd.Series([0] * 100), prob, n_bins=10)
    assert list(table.n) == [10] * 10
    assert table.mean_prob.is_monotonic_increasing


def test_calibration_bins_sort_by_probability_whatever_the_row_order():
    from early_warning.metrics import calibration_table

    table = calibration_table(pd.Series([1, 0, 1, 0]), pd.Series([0.9, 0.1, 0.8, 0.2]), n_bins=2)
    assert list(table.mean_prob.round(4)) == [0.15, 0.85]
    assert list(table.rate) == [0.0, 1.0]
