"""Day 2 baseline: logistic regression per prediction week, compared with a clicks ranking.

Three rankings are scored each week:
- lr_main: logistic regression on engagement, assessment and study-record features
- lr_with_demographics: the same plus the six demographic fields (comparison only)
- clicks_rank: no model, students with the fewest clicks so far come first

Train and test sets are whole module presentations, so the test set is a later run of each
module that the model has not seen.
"""

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from early_warning.features import (
    CATEGORICAL_FEATURES,
    DEMOGRAPHIC_FEATURES,
    NUMERIC_FEATURES,
    WEEKS,
    excluded_counts,
    load_week,
)
from early_warning.ingest import register_tables
from early_warning.metrics import evaluate
from early_warning.views import create_views

MODES = {
    # mode: (train presentations, test presentations, output file, note)
    "dev": (
        ("2013B", "2013J"),
        ("2014B",),
        "dev_metrics.json",
        "Development check on training presentations only. Not the test result.",
    ),
    "test": (
        ("2013B", "2013J", "2014B"),
        ("2014J",),
        "baseline_metrics.json",
        "Single run on the held-out 2014J presentation.",
    ),
    "smoke": (
        ("2013J",),
        ("2013J",),
        "smoke_metrics.json",
        "Test-sample run that trains and scores the same presentation. Checks the code only.",
    ),
}


def build_pipeline(with_demographics: bool) -> Pipeline:
    """Impute, scale and one-hot encode, then fit a logistic regression with default settings."""
    categorical = CATEGORICAL_FEATURES + (DEMOGRAPHIC_FEATURES if with_demographics else [])
    numeric_steps = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True), StandardScaler()
    )
    categorical_steps = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"),
        OneHotEncoder(handle_unknown="ignore"),
    )
    columns = ColumnTransformer(
        [("num", numeric_steps, NUMERIC_FEATURES), ("cat", categorical_steps, categorical)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([("columns", columns), ("model", LogisticRegression(max_iter=1000))])


def run(
    con: duckdb.DuckDBPyConnection,
    train: tuple[str, ...],
    test: tuple[str, ...],
    allow_overlap: bool = False,
) -> dict:
    """Fit and score every week. Returns metrics, cohort sizes and main-model coefficients."""
    if set(train) & set(test) and not allow_overlap:
        raise ValueError(f"train {train} and test {test} share a presentation")
    metrics, cohort, coefficients = {}, {}, []
    for week in WEEKS:
        df = load_week(con, week)
        excluded = excluded_counts(con, week)
        cohort[week] = {
            presentation: {
                "n": len(group),
                "base_rate": round(float(group.at_risk.mean()), 4),
                **excluded[presentation],
            }
            for presentation, group in df.groupby("code_presentation")
        }
        train_df = df[df.code_presentation.isin(train)]
        test_df = df[df.code_presentation.isin(test)].copy()
        y_train = train_df.at_risk.astype(int)

        main_model = build_pipeline(False).fit(train_df, y_train)
        demo_model = build_pipeline(True).fit(train_df, y_train)
        test_df["lr_main"] = main_model.predict_proba(test_df)[:, 1]
        test_df["lr_with_demographics"] = demo_model.predict_proba(test_df)[:, 1]
        test_df["clicks_rank"] = -test_df.clicks_total  # fewest clicks = highest risk

        metrics[week] = {
            name: evaluate(test_df, name)
            for name in ("lr_main", "lr_with_demographics", "clicks_rank")
        }
        names = main_model[:-1].get_feature_names_out()
        for name, value in zip(names, main_model[-1].coef_[0], strict=True):
            coefficients.append({"week": week, "feature": name, "coefficient": round(value, 4)})
    registrations = dict(
        con.execute(
            "SELECT code_presentation, count(*) FROM student_module GROUP BY 1 ORDER BY 1"
        ).fetchall()
    )
    return {
        "metrics": metrics,
        "cohort": cohort,
        "registrations": registrations,
        "coefficients": pd.DataFrame(coefficients),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("reports"))
    parser.add_argument("--mode", choices=MODES, required=True)
    args = parser.parse_args(argv)

    train, test, filename, note = MODES[args.mode]
    con = duckdb.connect()
    register_tables(con, args.processed)
    create_views(con)
    result = run(con, train, test, allow_overlap=args.mode == "smoke")

    header = {"note": note, "train": list(train), "test": list(test), "weeks": list(WEEKS)}
    args.out.mkdir(parents=True, exist_ok=True)
    metrics = {**header, "metrics": {str(w): m for w, m in result["metrics"].items()}}
    (args.out / filename).write_text(json.dumps(metrics, indent=2) + "\n")

    if args.mode == "test":
        cohort = {
            **header,
            # one row per student per module presentation; a student can take several modules
            "registrations_per_presentation": result["registrations"],
            "cohort_by_week": {str(w): c for w, c in result["cohort"].items()},
        }
        (args.out / "cohort_by_week.json").write_text(json.dumps(cohort, indent=2) + "\n")
        result["coefficients"].to_csv(args.out / "baseline_coefficients.csv", index=False)
    con.close()
    print(f"Wrote {args.out / filename}")
    return 0
