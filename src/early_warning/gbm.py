"""Gradient boosting model per prediction week, and the tuning step on a validation presentation.

The model is scikit-learn's HistGradientBoostingClassifier. It takes the same features as the
Day 2 lr_main model (no demographic fields). Missing values (mean_score when nothing has been
submitted, a missing date_registration) are handled by the model itself, so there is no
imputing. code_module is one-hot encoded, so SHAP values add up exactly (see reasons.py).
"""

import itertools

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from early_warning.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

LEARNING_RATE = 0.05
MAX_TREES = 500
SEED = 0

# Every combination is tried. The rest of the settings stay at the scikit-learn defaults.
GRID = {
    "max_leaf_nodes": [15, 31, 63],
    "min_samples_leaf": [20, 100],
    "l2_regularization": [0.0, 1.0],
}


def to_float(frame: pd.DataFrame) -> pd.DataFrame:
    """Nullable integer columns (for example date_registration) to float, with NaN for missing."""
    return frame.astype("float64")


def build_gbm_pipeline(max_iter: int = MAX_TREES, **params) -> Pipeline:
    """Numeric columns as they are, code_module one-hot, then gradient boosting."""
    columns = ColumnTransformer(
        [
            (
                "num",
                FunctionTransformer(to_float, feature_names_out="one-to-one"),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    model = HistGradientBoostingClassifier(
        learning_rate=LEARNING_RATE,
        max_iter=max_iter,
        early_stopping=False,  # the number of trees is chosen on the validation presentation
        random_state=SEED,
        **params,
    )
    return Pipeline([("columns", columns), ("model", model)])


def grid_settings(grid: dict[str, list]) -> list[dict]:
    """All combinations of the grid, as a list of keyword dicts."""
    names = list(grid)
    return [dict(zip(names, values, strict=True)) for values in itertools.product(*grid.values())]


def tune_week(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    grid: dict[str, list] = GRID,
    max_trees: int = MAX_TREES,
) -> dict:
    """Try every setting in the grid. Return all results and the one with the lowest log loss.

    For each setting, one model with max_trees trees is fitted, and the validation log loss is
    read after every tree (staged_predict_proba). So the best number of trees comes for free.
    """
    y_train = train_df.at_risk.astype(int)
    y_valid = valid_df.at_risk.astype(int)
    results = []
    for params in grid_settings(grid):
        pipe = build_gbm_pipeline(max_iter=max_trees, **params).fit(train_df, y_train)
        x_valid = pipe[:-1].transform(valid_df)
        losses = [
            log_loss(y_valid, proba[:, 1]) for proba in pipe[-1].staged_predict_proba(x_valid)
        ]
        best = int(np.argmin(losses))
        results.append({**params, "n_trees": best + 1, "valid_log_loss": round(losses[best], 5)})
    chosen = min(results, key=lambda r: r["valid_log_loss"])
    return {"results": results, "chosen": chosen}


def settings_from(chosen: dict) -> dict:
    """Keyword arguments for build_gbm_pipeline from a chosen row of tune_week."""
    params = {name: chosen[name] for name in GRID}
    return {"max_iter": chosen["n_trees"], **params}
