"""Load the released model bundle and score students with it. Used by the API and the batch job.

A bundle is a folder made by release.py: one week-<k>.skops file per prediction week and a
manifest.json with each file's SHA-256. Every file is checked against the manifest before it is
loaded, and skops refuses any type that is not trusted by default or listed below.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import skops.io as sio
from sklearn.pipeline import Pipeline

from early_warning.download import sha256_of
from early_warning.explain import top_reasons
from early_warning.features import WEEKS, model_input
from early_warning.metrics import GROUP, flag_top_k

# The only type in a fitted lr_main pipeline that skops does not trust by default.
TRUSTED_TYPES = ["numpy.dtype"]
LIST_FRACTION = 0.2  # each module's list is its top 20% by risk, as in the metrics
LIST_COLUMNS = [*GROUP, "week", "rank", "id_student", "risk", "reasons"]


@dataclass(frozen=True)
class Bundle:
    models: dict[int, Pipeline]
    manifest: dict


def load_bundle(directory: Path) -> Bundle:
    """Read manifest.json, check each week's file against its SHA-256, and load the models."""
    manifest = json.loads((directory / "manifest.json").read_text())
    models = {}
    for week in WEEKS:
        entry = manifest["weeks"].get(str(week))
        if entry is None:
            raise ValueError(f"the bundle has no model for week {week}")
        path = directory / entry["file"]
        if sha256_of(path) != entry["sha256"]:
            raise ValueError(f"{path.name} does not match its SHA-256 in manifest.json")
        models[week] = sio.load(path, trusted=TRUSTED_TYPES)
    return Bundle(models=models, manifest=manifest)


def score(bundle: Bundle, week: int, df: pd.DataFrame) -> pd.DataFrame:
    """Risk (probability of Fail or Withdrawn) and a list of reasons for each row of df."""
    model = bundle.models[week]
    inputs = model_input(df)
    return pd.DataFrame(
        {"risk": model.predict_proba(inputs)[:, 1], "reasons": top_reasons(model, inputs)},
        index=df.index,
    )


def risk_lists(bundle: Bundle, week: int, df: pd.DataFrame) -> pd.DataFrame:
    """The top 20% of each module presentation by risk, ranked, with reasons.

    Only ids, risk and reasons are kept: no outcome and no demographic column.
    """
    scored = df[[*GROUP, "id_student"]].copy()
    scored[["risk", "reasons"]] = score(bundle, week, df)
    listed = scored[flag_top_k(scored, "risk", LIST_FRACTION)]
    listed = listed.sort_values([*GROUP, "risk", "id_student"], ascending=[True, True, False, True])
    listed["rank"] = listed.groupby(GROUP).cumcount() + 1
    listed["week"] = week
    listed["risk"] = listed["risk"].round(4)
    listed["reasons"] = listed["reasons"].map("; ".join)
    return listed[LIST_COLUMNS].reset_index(drop=True)
