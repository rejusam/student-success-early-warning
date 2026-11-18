# student-success-early-warning

Goal: an early-warning model for students at risk of failing or withdrawing, built to run the way
a university data science team would run it in production. Planned parts: versioned data pipeline,
tested training job, evaluation gate, model registry, fairness audit, cloud-deployed scoring API,
scheduled retraining and drift monitoring. See Status for what exists today.

Data: Open University Learning Analytics Dataset (OULAD), CC BY 4.0
(Kuzilek, Hlosta & Zdrahal, 2015, https://doi.org/10.24432/C5KK69). Details, quirks and checks:
[`docs/data.md`](docs/data.md).

## Status
Done so far: data foundation (download with checksum, typed CSV-to-Parquet ingest, SQL data
checks, day-keyed SQL views), a logistic regression baseline scored at weeks 2, 4, 8 and 12,
and a gradient boosting challenger with calibration, per-student reasons, an evaluation gate
and an MLflow model registry. The challenger did not pass the gate (see below).
The fairness audit, deployment and monitoring are not built yet.

## Baseline (logistic regression)
Design: [`docs/design/day2-weekly-baseline.md`](docs/design/day2-weekly-baseline.md).

At the start of week k of a module, each student still enrolled gets a risk score for ending
with Fail or Withdrawn. Only data dated on or before day 7k - 1 is used, and a test checks this
by deleting all later clicks and submissions and confirming the features do not change. Banked
results (carried over from an earlier presentation) are kept, because they are known from the
start. Students who had already unregistered by that day are left out, because an advisor can
no longer reach them, and so are students who had not registered yet.

The models are trained on the 2013B, 2013J and 2014B presentations and tested once on 2014J.
Each module presentation gets its own ranked list. The headline is recall at the top 20%: if
advisors contact the 20% of students the list ranks highest in each module, what share of the
students who went on to fail or withdraw did they reach? Precision is the share of those
contacts who were really at risk. Three rankings are compared:

- `lr_main`: logistic regression on clicks, assessment submissions and scores, prior attempts,
  credits, registration day and module. No demographic fields.
- `lr_with_demographics`: the same plus gender, age band, region, highest education, IMD band
  and disability. Kept only as a comparison, to show what leaving demographics out costs.
- `clicks_rank`: no model. Students with the fewest clicks so far come first.

Test results on 2014J (`reports/baseline_metrics.json`):

| Week | Students | At risk | `lr_main` recall / precision | `lr_with_demographics` | `clicks_rank` |
|---|---|---|---|---|---|
| 2 | 9,333 | 0.4165 | 0.3092 / 0.6428 | 0.3229 / 0.6711 | 0.3139 / 0.6524 |
| 4 | 9,214 | 0.4079 | 0.3550 / 0.7226 | 0.3592 / 0.7313 | 0.3302 / 0.6723 |
| 8 | 8,879 | 0.3854 | 0.4135 / 0.7958 | 0.4170 / 0.8026 | 0.3568 / 0.6867 |
| 12 | 8,571 | 0.3633 | 0.4496 / 0.8154 | 0.4518 / 0.8195 | 0.3754 / 0.6808 |

What this shows:
- At week 2 the logistic regression does not beat the clicks ranking (recall 0.3092 against
  0.3139). That early, a simple engagement list is as useful. From week 4 the model is ahead,
  and the gap grows as assessment results come in.
- Adding demographics raises recall and precision a little at every week (compare the two
  model columns). The gain is largest at week 2 and small from week 4. The main model leaves
  demographics out. Whether it treats groups fairly is checked separately (fairness audit, not
  built yet).
- Between 0.3633 and 0.4165 of the 2014J cohort ended with Fail or Withdrawn (the "At risk"
  column), so a list of 20% of the cohort cannot reach most of them. Precision is the more
  telling number for an advisor.
- AUC-PR, ROC-AUC and results at 10% and 30% are in the same file. Coefficients of the main
  model, per standardised feature, are in `reports/baseline_coefficients.csv`.

Limits:
- OULAD does not record when a score was released to staff. A score is used from the day the
  work was submitted, including work handed in before its due date. That is earlier than an
  advisor would have seen the mark, and the effect is largest at week 2, when the first
  assignments are being handed in but are not yet due. The early-week results may therefore be
  a little optimistic.
- One test presentation (2014J). There is no estimate of how much the results vary between runs.
- A development check (train 2013B and 2013J, score 2014B, `reports/dev_metrics.json`) was run
  first to catch pipeline errors. Nothing was changed after it.

## Gradient boosting, calibration and the evaluation gate
Design: [`docs/design/day3-gbm-gate-registry.md`](docs/design/day3-gbm-gate-registry.md).

The challenger is scikit-learn's `HistGradientBoostingClassifier`, one model per week, on the
same features and cohort as `lr_main` (no demographic fields). LightGBM and XGBoost were tried
first. Both need a system OpenMP library on macOS that `uv sync` cannot install, and the
scikit-learn model is the same histogram-based method with no extra system library.

Steps, in order:
1. **Tuning on 2014B, not 2014J.** Fit on 2013B and 2013J, try 12 settings (tree size,
   minimum leaf size, L2 penalty) and pick the one and the number of trees with the lowest
   log loss on 2014B. Chosen settings: `reports/gbm_tuning.json`. Log loss is used because
   recall@top-20% is a step function that moves with a few students, so tuning on it would
   fit noise. The test run reads the settings from that file and cannot tune.
2. **Refit and calibrate.** Refit on 2013B, 2013J and 2014B. Probabilities are calibrated with
   Platt scaling, learned from leave-one-presentation-out predictions (train on two
   presentations, predict the third). Platt scaling only rescales the score, so every top-k
   list is the same before and after calibration.
3. **Score 2014J once and apply the gate.** The gate agreed on Day 2: recall@top-20% must be
   strictly higher than both `lr_main` and `clicks_rank` at weeks 4, 8 and 12. The script
   refits `lr_main` and `clicks_rank` in the same run and stops if they differ from
   `reports/baseline_metrics.json` (they matched).
4. **Register.** Each weekly model is logged to MLflow (local SQLite store in `mlruns/`, not in
   git) with its settings, test metrics, git commit and data checksum, and registered as
   `early-warning-week-<k>`. Alias `champion` if the gate passes, `challenger` if not. A failed
   gate also makes the script exit with code 1, so a scheduled retraining job stops there.

Validation on 2014B (`reports/gbm_validation.json`), recall@top-20%, before the test run:

| Week | Gradient boosting | `lr_main` | `clicks_rank` |
|---|---|---|---|
| 2 | 0.3062 | 0.3013 | 0.2959 |
| 4 | 0.3451 | 0.3451 | 0.3124 |
| 8 | 0.3857 | 0.3896 | 0.3364 |
| 12 | 0.4270 | 0.4263 | 0.3614 |

Test on 2014J (`reports/gate.json`, week 2 from `reports/gbm_metrics.json`), recall@top-20%:

| Week | Gradient boosting | `lr_main` | `clicks_rank` | Gate |
|---|---|---|---|---|
| 2 | 0.3164 | 0.3092 | 0.3139 | not gated |
| 4 | 0.3619 | 0.3550 | 0.3302 | pass |
| 8 | 0.4155 | 0.4135 | 0.3568 | pass |
| 12 | 0.4454 | 0.4496 | 0.3754 | fail |

**The gate failed at week 12**, so the gradient boosting models are registered as
`challenger` and `lr_main` stays the model to use. Nothing was changed after the test run.
On this data the two models are close: the gradient boosting model is a little ahead at
weeks 2 to 8 and a little behind at week 12. The validation run was just as close, in a
different pattern (ahead at weeks 2 and 12, level at week 4, behind at week 8). Both are well ahead of the clicks ranking from week 4. A model that is not
clearly better does not justify losing the plain coefficients of the logistic regression.

Calibration on 2014J (`reports/gbm_calibration.json`). Brier score is the mean squared error
of the probabilities. ECE is the average gap between predicted and observed risk over ten
equal-size groups of students. Lower is better for both.

| Week | ECE, boosting before / after Platt | ECE, `lr_main` | Brier, boosting after / `lr_main` |
|---|---|---|---|
| 2 | 0.0558 / 0.0539 | 0.0656 | 0.216 / 0.2239 |
| 4 | 0.0592 / 0.0551 | 0.0686 | 0.2134 / 0.2197 |
| 8 | 0.051 / 0.0517 | 0.0342 | 0.1865 / 0.179 |
| 12 | 0.0629 / 0.0516 | 0.0417 | 0.1747 / 0.1721 |

Platt scaling lowered the boosting model's calibration error at weeks 2, 4 and 12, and left
it about the same at week 8. `lr_main` is not recalibrated. Its error is lower than the
calibrated boosting model's at weeks 8 and 12 and higher at weeks 2 and 4. Both models are
off by about 3 to 7 percentage points on average, so a risk of "0.6" should be read as a
rough level, not an exact chance.

AUC. Day 2 reported AUC-PR over all modules pooled into one ranking, while the lists are per
module. `reports/gbm_metrics.json` now has both: `metrics` (pooled, as before) and
`per_module` (mean over module presentations). For example at week 12, AUC-PR pooled is
0.7233 for gradient boosting and 0.7305 for `lr_main`. The per-module mean is 0.7441 and 0.7412.
Pooled numbers mix in the differences in base rate between modules, so the per-module mean is
the one that matches how advisors use the lists.

**Reasons per student.** For each student, SHAP values give each feature's push on the
gradient boosting model's risk score, and they add up exactly to the score (a test checks
this). A student's reasons are the three features that push the risk up most, in plain words
with the student's value. The module is never given as a reason. Example, the top 10 students
of module AAA at week 8 of 2014J: `reports/example_risk_list.csv`. Mean absolute SHAP value per
feature and week: `reports/gbm_shap_importance.csv`. At week 2 the largest are clicks last
week and total clicks. From week 4 the average assessment score is the largest, followed by
assessments due but not submitted.

Limits:
- One validation presentation and one test presentation, and no confidence intervals. The
  week 12 gap that fails the gate is 0.0042 in recall. With differences this small, another
  presentation could reverse the result.
- MLflow runs locally. A shared tracking server comes with deployment.
- The same marking-lag caveat as the baseline applies.
- 2014J was scored twice by this model. The first run had a bug in the calibration bins (they
  were not sorted by probability), which made the ECE values wrong. After the fix, the run was
  repeated with the same settings. Every other output, including the gate result, came out
  identical, and only the ECE values and bin tables changed.

## Quickstart
Needs [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked
uv run python scripts/download_oulad.py   # data/raw/, checksum verified
uv run python scripts/ingest.py           # data/processed/*.parquet
uv run python scripts/run_checks.py       # reports/data_checks.json, reports/data_profile.json
uv run python scripts/train_baseline.py --mode test   # reports/baseline_metrics.json and others
uv run python scripts/train_gbm.py --mode tune        # settings chosen on 2014B
uv run python scripts/train_gbm.py --mode test        # 2014J once, gate, MLflow registry
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db   # browse runs and models
uv run pytest                             # runs on a small sample in tests/fixtures/
```

Optional: `uv run pre-commit install` runs ruff and a gitleaks secret scan before each commit.

## Layout
| Path | Contents |
|---|---|
| `src/early_warning/` | download, ingest, checks, views, features, metrics, baseline, gradient boosting, calibration, reasons, gate and registry code |
| `sql/checks/` | one SQL file per data rule; each returns the rows that break it |
| `sql/views/` | `student_module`, `vle_weekly`, `assessment_submissions` |
| `sql/features/` | `student_week.sql`: one row per student at a prediction week |
| `scripts/` | command-line entry points |
| `reports/` | outputs of the scripts; every number quoted in the docs comes from here |
| `docs/design/` | design notes |
