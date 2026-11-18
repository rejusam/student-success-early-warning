"""Calibrate a fitted model's probabilities with Platt scaling.

The calibration data comes from leave-one-presentation-out predictions: fit on all training
presentations but one, predict the one left out, and repeat. That matches how the model is
used, on a presentation it has never seen. Platt scaling fits a logistic curve to the model's
score. The curve only goes up, so the order of students, and so every top-k list, is the same
as the uncalibrated model's.
"""

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import Pipeline


def calibration_splits(train_df: pd.DataFrame) -> list:
    """Leave one presentation out. With a single presentation (the test sample), 3 folds."""
    y = train_df.at_risk.astype(int)
    if train_df.code_presentation.nunique() >= 2:
        return list(LeaveOneGroupOut().split(train_df, y, groups=train_df.code_presentation))
    return list(StratifiedKFold(n_splits=3, shuffle=True, random_state=0).split(train_df, y))


def calibrate(pipeline: Pipeline, train_df: pd.DataFrame) -> CalibratedClassifierCV:
    """Fit the pipeline on all of train_df, with a Platt curve learned from held-out predictions.

    With ensemble=False, scikit-learn uses the held-out predictions only to fit the curve,
    then refits one copy of the pipeline on all of train_df. That copy scores new students.
    """
    calibrated = CalibratedClassifierCV(
        pipeline, method="sigmoid", cv=calibration_splits(train_df), ensemble=False
    )
    return calibrated.fit(train_df, train_df.at_risk.astype(int))


def fitted_pipeline(calibrated: CalibratedClassifierCV) -> Pipeline:
    """The uncalibrated pipeline inside a calibrated model (used for SHAP reasons)."""
    return calibrated.calibrated_classifiers_[0].estimator
