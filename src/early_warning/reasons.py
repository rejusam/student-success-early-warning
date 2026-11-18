"""Reasons for each student's risk score, from SHAP values of the gradient boosting model.

A SHAP value is one feature's push on the model's log-odds of risk for one student. The base
value plus all of a student's SHAP values equals the model's raw score for that student.
A student's reasons are the features that push the risk up the most.
"""

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from early_warning.labels import LABELS, describe


def shap_frame(pipeline: Pipeline, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """SHAP values per student and model input, the model inputs, and the base value."""
    inputs = pipeline[:-1].transform(df)
    names = pipeline[:-1].get_feature_names_out()
    explainer = shap.TreeExplainer(pipeline[-1])
    values = explainer.shap_values(inputs)
    base = float(np.ravel(explainer.expected_value)[0])
    return (
        pd.DataFrame(values, columns=names, index=df.index),
        pd.DataFrame(inputs, columns=names, index=df.index),
        base,
    )


def top_reasons(shap_values: pd.DataFrame, inputs: pd.DataFrame, n: int = 3) -> pd.Series:
    """For each student, up to n plain-language reasons, joined by '; '.

    Only features that push the risk up are used, largest push first. Module columns are skipped.
    """
    usable = [c for c in shap_values.columns if c in LABELS]
    reasons = []
    for index, row in shap_values[usable].iterrows():
        ups = row[row > 0].sort_values(ascending=False).head(n)
        reasons.append("; ".join(describe(f, inputs.at[index, f]) for f in ups.index))
    return pd.Series(reasons, index=shap_values.index)


def importance(shap_values: pd.DataFrame) -> pd.Series:
    """Mean absolute SHAP value per model input, largest first (the global view)."""
    return shap_values.abs().mean().sort_values(ascending=False)
