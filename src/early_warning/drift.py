"""Drift checks: does a new cohort look like the cohort the model was trained on?

The Population Stability Index (PSI) compares the share of students in each bin of an input
(or of the risk score) between a reference cohort and a current cohort:

    PSI = sum over bins of (current share - reference share) * ln(current share / reference share)

Bins come from the reference only: numeric inputs are cut at the reference deciles (repeated
edges dropped, so a mostly-zero input gets fewer bins), with a last bin for missing values;
the module gets one bin per module seen in training and a last bin for any other. The
reference is saved with the released model (release.py), so a later cohort can be checked
without the training data. A profile holds bin edges and shares of a whole cohort, never a
student row.

Bands (the usual ones in credit-risk monitoring): PSI below 0.1 stable, 0.1 to 0.25
moderate, 0.25 or more major.
"""

import numpy as np
import pandas as pd

from early_warning.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

EPS = 0.0001  # stands in for an empty bin's share, so the log is defined
MODERATE, MAJOR = 0.1, 0.25
DECILES = [i / 10 for i in range(1, 10)]
STATUS = {"stable": "ok", "moderate": "watch", "major": "alert"}


def numeric_profile(values: pd.Series) -> dict:
    """Decile edges of the non-missing values, and the share of values in each bin."""
    present = values.dropna().to_numpy(dtype="float64")
    edges = np.unique(np.quantile(present, DECILES)).tolist() if len(present) else []
    profile = {"edges": edges}
    profile["shares"] = bin_shares(profile, values)
    profile["mean"] = round(float(present.mean()), 4) if len(present) else None
    profile["missing_share"] = round(profile["shares"][-1], 4)
    return profile


def categorical_profile(values: pd.Series) -> dict:
    """One bin per category in the values, sorted, and a last bin for unseen categories."""
    profile = {"categories": sorted(values.dropna().unique().tolist())}
    profile["shares"] = bin_shares(profile, values)
    return profile


def bin_shares(profile: dict, values: pd.Series) -> list[float]:
    """Share of the values in each bin of the profile. The last bin is missing (numeric) or
    unseen (categorical)."""
    if "categories" in profile:
        positions = {category: i for i, category in enumerate(profile["categories"])}
        index = values.map(positions).fillna(len(positions)).to_numpy(dtype=int)
        n_bins = len(positions) + 1
    else:
        edges = profile["edges"]
        x = values.to_numpy(dtype="float64")
        # bin i holds edges[i-1] < x <= edges[i]; values above the last edge go to bin len(edges)
        index = np.searchsorted(edges, x, side="left")
        index[np.isnan(x)] = len(edges) + 1
        n_bins = len(edges) + 2
    counts = np.bincount(index, minlength=n_bins)
    return (counts / counts.sum()).tolist()


def psi(reference: list[float], current: list[float]) -> float:
    """Population Stability Index of two lists of bin shares."""
    ref = np.where(np.asarray(reference) == 0, EPS, reference)
    cur = np.where(np.asarray(current) == 0, EPS, current)
    return float(((cur - ref) * np.log(cur / ref)).sum())


def band(value: float) -> str:
    if value >= MAJOR:
        return "major"
    return "moderate" if value >= MODERATE else "stable"


def build_reference(df: pd.DataFrame, risk: pd.Series) -> dict:
    """The profile of every model input and of the risk score for one cohort."""
    inputs = {name: numeric_profile(df[name]) for name in NUMERIC_FEATURES}
    inputs |= {name: categorical_profile(df[name]) for name in CATEGORICAL_FEATURES}
    return {"n": len(df), "inputs": inputs, "score": numeric_profile(pd.Series(risk))}


def _entry(profile: dict, values: pd.Series) -> dict:
    shares = bin_shares(profile, values)
    value = psi(profile["shares"], shares)
    entry = {"psi": round(value, 4), "band": band(value)}
    if "edges" in profile:
        present = values.dropna()
        entry |= {
            "reference_mean": profile["mean"],
            "current_mean": round(float(present.mean()), 4) if len(present) else None,
            "reference_missing": profile["missing_share"],
            "current_missing": round(shares[-1], 4),
        }
    return entry


def compare(reference: dict, df: pd.DataFrame, risk: pd.Series) -> dict:
    """PSI of each input and of the score against the reference, and an overall status:
    alert if any PSI is major, watch if any is moderate, else ok."""
    if len(df) == 0:
        raise ValueError("cannot check drift for an empty cohort")
    inputs = {name: _entry(profile, df[name]) for name, profile in reference["inputs"].items()}
    score = _entry(reference["score"], pd.Series(risk))
    worst = max([score["psi"], *(e["psi"] for e in inputs.values())])
    return {
        "n_reference": reference["n"],
        "n_current": len(df),
        "status": STATUS[band(worst)],
        "score": score,
        "largest_input": max(inputs, key=lambda name: inputs[name]["psi"]),
        "inputs": inputs,
    }
