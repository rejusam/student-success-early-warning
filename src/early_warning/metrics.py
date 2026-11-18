"""Metrics for a ranked list of students.

An advisor gets one list per module presentation, so the top k% is taken within each module
presentation, then the counts are added up over all of them. Higher score = higher risk.
"""

import math

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

GROUP = ["code_module", "code_presentation"]
K_FRACTIONS = (0.1, 0.2, 0.3)


def flag_top_k(df: pd.DataFrame, score_col: str, frac: float) -> pd.Series:
    """True for the top ceil(frac * n) students of each module presentation.

    Ties in score are broken by id_student (lower id first), so the result is repeatable.
    """
    ranked = df.sort_values([score_col, "id_student"], ascending=[False, True])
    position = ranked.groupby(GROUP).cumcount()  # 0 for the highest score in each group
    group_size = ranked.groupby(GROUP)["id_student"].transform("size")
    k = (frac * group_size).apply(math.ceil)
    return (position < k).reindex(df.index)


def top_k_metrics(df: pd.DataFrame, score_col: str, frac: float) -> dict:
    """Recall and precision of the top-k lists, pooled over module presentations."""
    flagged = flag_top_k(df, score_col, frac)
    true_positives = int((flagged & df["at_risk"]).sum())
    positives = int(df["at_risk"].sum())
    n_flagged = int(flagged.sum())
    return {
        "recall": round(true_positives / positives, 4) if positives else None,
        "precision": round(true_positives / n_flagged, 4),
        "n_flagged": n_flagged,
    }


def evaluate(df: pd.DataFrame, score_col: str) -> dict:
    """All Day 2 metrics for one ranking of one cohort."""
    result = {
        "n": len(df),
        "base_rate": round(float(df["at_risk"].mean()), 4),
        "auc_pr": round(float(average_precision_score(df["at_risk"], df[score_col])), 4),
        "roc_auc": round(float(roc_auc_score(df["at_risk"], df[score_col])), 4),
    }
    for frac in K_FRACTIONS:
        result[f"top_{round(frac * 100)}"] = top_k_metrics(df, score_col, frac)
    return result
