"""Reasons for each student's risk score, from the logistic regression itself (used in serving).

The model's log-odds of risk are the intercept plus, for every model input, coefficient times
the input's scaled value. The inputs are scaled to mean 0 on the training data, so each term is
that input's push on the risk compared with an average training student. The pushes plus the
intercept equal the log-odds exactly; for this model this is the same as SHAP's linear explainer.

Inputs are added up into reasons an advisor can read:
- a value and its missing-value indicator (mean_score, date_registration) are one reason;
- the module's one-hot columns are one group, never shown, because an advisor cannot act on it;
- n_missed is always n_due - n_submitted, so the model cannot tell the three apart. Only their
  total push means anything, so they are one reason, "submissions";
- the five course-site activity counts move together, so the model can give one of them a
  small positive weight while the others carry a negative one (at week 8, clicks_total
  +0.099 against weeks_active -0.4179, reports/baseline_coefficients.csv). Alone, that would
  read as "many clicks raise the risk". Together they are one reason, "activity".
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from early_warning.labels import describe

GROUPS = {
    "submissions": ("n_due", "n_submitted", "n_missed"),
    "activity": (
        "clicks_total",
        "clicks_last_week",
        "active_days_total",
        "weeks_active",
        "clicks_before_start",
    ),
}
GROUP_OF = {feature: group for group, features in GROUPS.items() for feature in features}


def reason_of(name: str) -> str:
    """The reason a model input belongs to, e.g. missingindicator_mean_score -> mean_score."""
    name = name.removeprefix("missingindicator_")
    if name.startswith("code_module_"):
        return "code_module"
    return GROUP_OF.get(name, name)


def pushes(pipeline: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    """One row per student, one column per reason: its push on the log-odds of risk."""
    inputs = pipeline[:-1].transform(df)
    if hasattr(inputs, "toarray"):  # the one-hot step can return a sparse matrix
        inputs = inputs.toarray()
    names = pipeline[:-1].get_feature_names_out()
    terms = pd.DataFrame(np.asarray(inputs) * pipeline[-1].coef_[0], columns=names, index=df.index)
    return terms.T.groupby(reason_of, sort=False).sum().T


def reason_text(reason: str, row: pd.Series) -> str:
    """Plain words for one reason, with the student's own values (before any imputing)."""
    if reason == "submissions":
        return f"submitted {row.n_submitted:.0f} of {row.n_due:.0f} assessments due so far"
    if reason == "activity":
        return (
            f"course-site activity so far: clicks {row.clicks_total:.0f}, active days "
            f"{row.active_days_total:.0f}, clicks last week {row.clicks_last_week:.0f}"
        )
    return describe(reason, row[reason])


def top_reasons(pipeline: Pipeline, df: pd.DataFrame, n: int = 3) -> pd.Series:
    """For each student, up to n reasons that push the risk up, largest push first."""
    values = pushes(pipeline, df).drop(columns="code_module")
    reasons = []
    for index, row in values.iterrows():
        ups = row[row > 0].sort_values(ascending=False).head(n)
        reasons.append([reason_text(reason, df.loc[index]) for reason in ups.index])
    return pd.Series(reasons, index=df.index)
