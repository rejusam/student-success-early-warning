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


def per_module_auc(df: pd.DataFrame, score_col: str) -> dict:
    """Mean AUC-PR and ROC-AUC over module presentations, each ranked on its own.

    evaluate() pools all modules into one ranking. Advisors get one list per module, so this
    is the closer match to how the scores are used.
    """
    auc_pr, roc = [], []
    for _, group in df.groupby(GROUP):
        if group["at_risk"].nunique() < 2:
            continue  # AUC is undefined when everyone, or no one, is at risk
        auc_pr.append(average_precision_score(group["at_risk"], group[score_col]))
        roc.append(roc_auc_score(group["at_risk"], group[score_col]))
    return {
        "auc_pr_module_mean": round(sum(auc_pr) / len(auc_pr), 4) if auc_pr else None,
        "roc_auc_module_mean": round(sum(roc) / len(roc), 4) if roc else None,
        "n_modules": len(auc_pr),
    }


def calibration_table(y: pd.Series, prob: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    """Students sorted by probability and cut into n_bins groups of (nearly) equal size.

    Each row: the mean predicted probability and the observed at-risk rate in that group.
    """
    rank = prob.rank(method="first").to_numpy() - 1  # 0 for the lowest probability
    frame = pd.DataFrame(
        {
            "prob": prob.to_numpy(),
            "y": y.astype(float).to_numpy(),
            "bin": (rank * n_bins // len(prob)).astype(int),
        }
    )
    table = frame.groupby("bin").agg(
        n=("y", "size"), mean_prob=("prob", "mean"), rate=("y", "mean")
    )
    return table.reset_index()


def calibration_metrics(y: pd.Series, prob: pd.Series, n_bins: int = 10) -> dict:
    """Brier score and expected calibration error (ECE) of predicted probabilities.

    ECE = sum over bins of (bin share of students) * |mean predicted - observed rate|.
    """
    y_float = y.astype(float).to_numpy()
    brier = float(((prob.to_numpy() - y_float) ** 2).mean())
    table = calibration_table(y, prob, n_bins)
    ece = float((table.n / table.n.sum() * (table.mean_prob - table.rate).abs()).sum())
    return {
        "brier": round(brier, 4),
        "ece": round(ece, 4),
        "bins": [
            {"n": int(r.n), "mean_prob": round(r.mean_prob, 4), "rate": round(r.rate, 4)}
            for r in table.itertuples()
        ],
    }
