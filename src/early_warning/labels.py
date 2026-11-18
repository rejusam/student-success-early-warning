"""Plain-language text for each model input, shared by the SHAP reasons (Day 3) and the
logistic regression reasons used in serving (Day 5). No heavy imports, so the service can use it.
"""

import pandas as pd

# Plain words for each feature; the student's value is shown after a colon.
# code_module is left out on purpose: an advisor cannot act on which module it is.
LABELS = {
    "clicks_total": "clicks on the course site so far",
    "clicks_last_week": "clicks last week",
    "active_days_total": "days active so far",
    "weeks_active": "weeks active so far",
    "clicks_before_start": "clicks before the module started",
    "n_due": "assessments due so far",
    "n_submitted": "assessments submitted so far",
    "n_missed": "assessments due but not submitted",
    "n_late": "assessments submitted late",
    "mean_score": "average assessment score",
    "num_of_prev_attempts": "earlier attempts at this module",
    "studied_credits": "credits being studied",
    "date_registration": "registration day (0 = module start)",
}
MISSING_LABELS = {
    "mean_score": "no assessment score yet",
    "date_registration": "registration date not recorded",
}


def describe(feature: str, value: float) -> str:
    if pd.isna(value):
        return MISSING_LABELS.get(feature, f"{feature} missing")
    return f"{LABELS[feature]}: {value:.0f}"
