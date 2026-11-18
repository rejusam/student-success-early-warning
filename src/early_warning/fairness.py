"""Fairness audit by student group: measures, gaps to a reference group, review flags and
bootstrap ranges.

The top-20% lists are built once per module presentation over all its students, as an advisor
gets them. Each student's flag is then counted in their group, so the audit measures the list
as it is used. Groups never get a list of their own.
"""

import numpy as np
import pandas as pd

from early_warning.features import DEMOGRAPHIC_FEATURES
from early_warning.metrics import GROUP, flag_top_k

FRAC = 0.2  # the headline list: top 20% of each module presentation
ATTRIBUTES = list(DEMOGRAPHIC_FEATURES)
INTERSECTION = "disability_x_imd"
ALL = "all"  # one group holding every student, to compare each group with the whole cohort
UNKNOWN = "Unknown"

# Groups smaller than this are reported with their ranges but never flagged.
MIN_N = 100
MIN_AT_RISK = 30
# Review flags, fixed before the audit ran (docs/design/day4-fairness-audit.md, decision 8).
RATIO_LOW, RATIO_HIGH = 0.8, 1.25
CALIBRATION_LIMIT = 0.05

# IMD band in three wider bands, for the intersection with disability. The source writes
# "10-20" without a "%".
IMD_GROUPS = {
    "0-10%": "0-30%",
    "10-20": "0-30%",
    "20-30%": "0-30%",
    "30-40%": "30-70%",
    "40-50%": "30-70%",
    "50-60%": "30-70%",
    "60-70%": "30-70%",
    "70-80%": "70-100%",
    "80-90%": "70-100%",
    "90-100%": "70-100%",
    UNKNOWN: UNKNOWN,
}

# Measures that get a bootstrap range.
MEASURES = [
    "flag_rate",
    "recall",
    "recall_diff",
    "recall_ratio",
    "precision",
    "mean_risk",
    "calibration_gap",
    "roc_auc",
]


def add_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of df with missing demographic values as "Unknown", the intersection column, and
    an "all" column with one group for the whole cohort."""
    out = df.copy()
    for column in ATTRIBUTES:
        out[column] = out[column].astype(object).where(out[column].notna(), UNKNOWN).astype(str)
    unmapped = set(out["imd_band"]) - set(IMD_GROUPS)
    if unmapped:
        raise ValueError(f"IMD bands with no wider band: {sorted(unmapped)}")
    imd_group = out["imd_band"].map(IMD_GROUPS)
    out[INTERSECTION] = "disability " + out["disability"] + ", IMD " + imd_group
    out[ALL] = "All students"
    return out


def group_table(df: pd.DataFrame, score_col: str, flagged: pd.Series, attribute: str):
    """One row per group of `attribute`: sizes and measures of the given list.

    flagged must come from the whole module list (flag_top_k on all students), not per group.
    recall, precision and roc_auc are NaN when a group has no students to divide by.
    """
    frame = pd.DataFrame(
        {
            "group": df[attribute].to_numpy(),
            "y": df["at_risk"].astype(bool).to_numpy(),
            "flag": flagged.astype(bool).to_numpy(),
            "risk": df[score_col].to_numpy(dtype=float),
        }
    )
    frame["hit"] = frame["y"] & frame["flag"]
    # ROC-AUC from ranks: (sum of the at-risk students' ranks - n1(n1+1)/2) / (n1 * n0)
    frame["rank"] = frame.groupby("group")["risk"].rank()
    frame["rank_at_risk"] = frame["rank"].where(frame["y"], 0.0)

    by_group = frame.groupby("group")
    table = pd.DataFrame(
        {
            "n": by_group.size(),
            "n_at_risk": by_group["y"].sum(),
            "n_flagged": by_group["flag"].sum(),
            "hits": by_group["hit"].sum(),
            "mean_risk": by_group["risk"].mean(),
            "rank_sum": by_group["rank_at_risk"].sum(),
        }
    )
    n1 = table["n_at_risk"].astype(float)
    n0 = table["n"] - n1
    table["at_risk_rate"] = n1 / table["n"]
    table["flag_rate"] = table["n_flagged"] / table["n"]
    table["recall"] = table["hits"] / n1.where(n1 > 0)
    table["precision"] = table["hits"] / table["n_flagged"].where(table["n_flagged"] > 0)
    table["calibration_gap"] = table["mean_risk"] - table["at_risk_rate"]
    both_classes = (n1 > 0) & (n0 > 0)
    table["roc_auc"] = ((table["rank_sum"] - n1 * (n1 + 1) / 2) / (n1 * n0)).where(both_classes)
    table.index.name = "group"
    return table.drop(columns=["rank_sum", "hits", "n_flagged"])


def reference_group(table: pd.DataFrame) -> str:
    """The largest group (first in sorted order if two are the same size)."""
    return str(table["n"].sort_index().idxmax())


def add_gaps(table: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Recall difference (group - reference) and ratio (group / reference)."""
    out = table.copy()
    ref = out.loc[reference, "recall"]
    out["recall_diff"] = out["recall"] - ref
    out["recall_ratio"] = out["recall"] / ref if ref > 0 else np.nan
    return out


def audit_once(
    df: pd.DataFrame, score_cols: list[str], attributes: list[str], references: dict[str, str]
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Group tables for every model and attribute, and each model's overall recall.

    Returns a long frame (model, attribute, group, measures) and {model: recall}.
    """
    tables, overall = [], {}
    positives = df["at_risk"].astype(bool)
    for model in score_cols:
        flagged = flag_top_k(df, model, FRAC)
        overall[model] = float((flagged & positives).sum() / positives.sum())
        for attribute in attributes:
            table = add_gaps(group_table(df, model, flagged, attribute), references[attribute])
            tables.append(table.reset_index().assign(model=model, attribute=attribute))
    return pd.concat(tables, ignore_index=True), overall


def resample(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Draw students with replacement within each module presentation, keeping its size.

    The index is reset, because a student drawn twice would otherwise appear twice under the
    same label, and flag_top_k aligns on the index.
    """
    parts = [
        group.iloc[rng.integers(0, len(group), len(group))]
        for _, group in df.groupby(GROUP, sort=True)
    ]
    return pd.concat(parts, ignore_index=True)


def bootstrap(
    df: pd.DataFrame,
    score_cols: list[str],
    attributes: list[str],
    references: dict[str, str],
    n_boot: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repeat audit_once on n_boot resamples. All models see the same resamples (paired).

    Returns the stacked group tables and the overall recalls, each with a "replicate" column.
    The reference groups stay those of the full data. Rows are sorted first, because resample
    draws by position and the feature query returns rows in no fixed order. Without the sort,
    the same seed would draw different students on each run.
    """
    df = df.sort_values([*GROUP, "id_student"]).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    tables, overall = [], []
    for replicate in range(n_boot):
        table, recall = audit_once(resample(df, rng), score_cols, attributes, references)
        tables.append(table.assign(replicate=replicate))
        overall.append({"replicate": replicate, **recall})
    return pd.concat(tables, ignore_index=True), pd.DataFrame(overall)


def intervals(replicates: pd.DataFrame, measures: list[str] = MEASURES) -> pd.DataFrame:
    """2.5th and 97.5th percentiles per model, attribute and group. NaN values are skipped."""
    by_group = replicates.groupby(["model", "attribute", "group"])[measures]
    low = by_group.quantile(0.025).add_suffix("_lo")
    high = by_group.quantile(0.975).add_suffix("_hi")
    return low.join(high).reset_index()


def is_small(n: int, n_at_risk: int, min_n: int = MIN_N, min_at_risk: int = MIN_AT_RISK) -> bool:
    return n < min_n or n_at_risk < min_at_risk


def excludes_zero(low: float, high: float) -> bool:
    """True if the whole range is above or below 0. False when either end is NaN."""
    if np.isnan(low) or np.isnan(high):
        return False
    return bool(low > 0 or high < 0)


def review_flag(row) -> str:
    """Why a group needs a person to look at it, or "" if it does not.

    row needs: small, recall_ratio, recall_diff_lo, recall_diff_hi, calibration_gap,
    calibration_gap_lo, calibration_gap_hi.
    """
    if row["small"]:
        return ""
    reasons = []
    ratio_off = row["recall_ratio"] < RATIO_LOW or row["recall_ratio"] > RATIO_HIGH
    if ratio_off and excludes_zero(row["recall_diff_lo"], row["recall_diff_hi"]):
        reasons.append("recall gap")
    calibration_off = abs(row["calibration_gap"]) > CALIBRATION_LIMIT
    if calibration_off and excludes_zero(row["calibration_gap_lo"], row["calibration_gap_hi"]):
        reasons.append("calibration")
    return "; ".join(reasons)


def finish(
    point: pd.DataFrame, ranges: pd.DataFrame, min_n: int = MIN_N, min_at_risk: int = MIN_AT_RISK
) -> pd.DataFrame:
    """Join point values and ranges, then add the small label and the review flag."""
    out = point.merge(ranges, on=["model", "attribute", "group"], how="left")
    out["small"] = [
        is_small(n, k, min_n, min_at_risk) for n, k in zip(out["n"], out["n_at_risk"], strict=True)
    ]
    out["flag"] = out.apply(review_flag, axis=1)
    return out
