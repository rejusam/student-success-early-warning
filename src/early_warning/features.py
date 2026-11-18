"""Build the features for one prediction week from sql/features/student_week.sql.

Week k means the prediction is made at the start of week k, using data up to the end of
week k - 1, so the last usable day is 7 * k - 1. The views from views.py must exist first.
"""

from pathlib import Path

import duckdb
import pandas as pd

SQL_PATH = Path(__file__).resolve().parents[2] / "sql" / "features" / "student_week.sql"

WEEKS = (2, 4, 8, 12)

NUMERIC_FEATURES = [
    "clicks_total",
    "clicks_last_week",
    "active_days_total",
    "weeks_active",
    "clicks_before_start",
    "n_due",
    "n_submitted",
    "n_missed",
    "n_late",
    "mean_score",
    "num_of_prev_attempts",
    "studied_credits",
    "date_registration",
]
CATEGORICAL_FEATURES = ["code_module"]
# Used only by the comparison model and the fairness audit, not by the main model.
DEMOGRAPHIC_FEATURES = [
    "gender",
    "age_band",
    "region",
    "highest_education",
    "imd_band",
    "disability",
]


def cutoff_day(week: int) -> int:
    """Last day of data that may be used for a prediction at the start of `week`."""
    return 7 * week - 1


def load_week(con: duckdb.DuckDBPyConnection, week: int) -> pd.DataFrame:
    """One row per student in the week's cohort: keys, features and the at_risk target."""
    params = {"cutoff": cutoff_day(week), "last_week": week - 1}
    return con.execute(SQL_PATH.read_text(), params).df()


def excluded_counts(con: duckdb.DuckDBPyConnection, week: int) -> dict[str, dict[str, int]]:
    """Per presentation, how many registrations the week's cohort rule leaves out, and why.

    A registration that unregistered by the cutoff counts as left_by_cutoff even if it also
    registered late, so the two counts do not overlap.
    """
    rows = con.execute(
        """
        SELECT
            code_presentation,
            count(*) FILTER (WHERE date_unregistration <= $cutoff) AS left_by_cutoff,
            count(*) FILTER (
                WHERE date_registration > $cutoff
                  AND NOT coalesce(date_unregistration <= $cutoff, false)
            ) AS not_yet_registered
        FROM student_module
        GROUP BY 1
        """,
        {"cutoff": cutoff_day(week)},
    ).fetchall()
    return {p: {"left_by_cutoff": left, "not_yet_registered": late} for p, left, late in rows}


def model_input(df: pd.DataFrame) -> pd.DataFrame:
    """The columns the model reads, numeric ones as float (the serving input format)."""
    out = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    out[NUMERIC_FEATURES] = out[NUMERIC_FEATURES].astype("float64")
    return out
