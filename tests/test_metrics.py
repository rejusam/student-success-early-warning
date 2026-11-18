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
