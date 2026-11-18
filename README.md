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
checks, day-keyed SQL views) and a logistic regression baseline scored at weeks 2, 4, 8 and 12.
Gradient boosting, the fairness audit, deployment and monitoring are not built yet.

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

## Quickstart
Needs [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked
uv run python scripts/download_oulad.py   # data/raw/, checksum verified
uv run python scripts/ingest.py           # data/processed/*.parquet
uv run python scripts/run_checks.py       # reports/data_checks.json, reports/data_profile.json
uv run python scripts/train_baseline.py --mode test   # reports/baseline_metrics.json and others
uv run pytest                             # runs on a small sample in tests/fixtures/
```

Optional: `uv run pre-commit install` runs ruff and a gitleaks secret scan before each commit.

## Layout
| Path | Contents |
|---|---|
| `src/early_warning/` | download, ingest, checks, views, features, metrics and baseline code |
| `sql/checks/` | one SQL file per data rule; each returns the rows that break it |
| `sql/views/` | `student_module`, `vle_weekly`, `assessment_submissions` |
| `sql/features/` | `student_week.sql`: one row per student at a prediction week |
| `scripts/` | command-line entry points |
| `reports/` | outputs of the scripts; every number quoted in the docs comes from here |
| `docs/design/` | design notes |
