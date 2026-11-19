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
and an MLflow model registry. The challenger did not pass the gate (see below). A fairness
audit by student group and a model card ([`docs/model_card.md`](docs/model_card.md)) cover the
model in use. The model in use is served as a weekly batch job and an API from one container
image, with AWS infrastructure as code, keyless deployment from GitHub Actions and a retraining
workflow that runs the gate and the fairness audit (see "Serving and deployment"). Nothing is
deployed on AWS yet, so nothing is live. Drift and performance monitoring runs next to every
weekly list and as a report over the test presentation (see "Monitoring"). How this project
differs from the 2025 notebook it replaces: [`docs/v1-to-v2.md`](docs/v1-to-v2.md).

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
  demographics out. Whether it treats groups fairly is checked separately (see Fairness audit
  and model card).
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
| 4 | 0.059 / 0.0549 | 0.0686 | 0.2134 / 0.2197 |
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
- The Day 5 reproducibility run changed the week-4 ECE (0.0592 to 0.059 before Platt scaling,
  0.0551 to 0.0549 after) and some bin rates in `gbm_calibration.json` and
  `gbm_validation.json`. At week 4 some students have exactly the same probability, and the
  bins split such ties by row order. The feature query got a fixed row order on Day 4, after
  these two files were written on Day 3. All other reports, the gate and the models came out
  identical.

## Fairness audit and model card
Design: [`docs/design/day4-fairness-audit.md`](docs/design/day4-fairness-audit.md).
Model card: [`docs/model_card.md`](docs/model_card.md).

The audit asks whether the list reaches at-risk students equally across groups. It covers six
attributes (gender, age band, region, highest education, IMD band, disability) and one
combination (disability by IMD band), at every week, for `lr_main` (the model in use), the
gradient boosting challenger (loaded from the MLflow registry, not refitted) and
`lr_with_demographics`. The script stops if any model's 2014J metrics differ from the saved
Day 2 and Day 3 reports. Nothing about any model was changed because of the audit.

How it works:
- Each module's top-20% list is built once over all its students, as an advisor gets it; each
  student's flag is then counted in their group.
- Per group: recall (the headline, "equal opportunity"), precision, calibration gap (mean
  predicted minus observed risk) and ROC-AUC, with the group's size and at-risk rate.
- 95% ranges from 1,000 bootstrap resamples of students within each module presentation, with
  the lists rebuilt in each resample. The same resamples are used for every model, so model
  differences are paired.
- A group is flagged for review if its recall is below 0.8 or above 1.25 times the largest
  group's and the range of the difference excludes 0, or if its calibration gap is over 0.05
  with a range that excludes 0. Groups under 100 students or 30 at-risk students are labelled
  small and not flagged. The rules were fixed before the first run.

Main findings for `lr_main` at week 12 (`reports/fairness_groups.csv`):
- At-risk women are reached less often than at-risk men: recall 0.4132 against 0.4742. The
  difference's range excludes 0, but the ratio (0.8715) is inside the 0.8 rule. The same gap
  appears at every week and in the gradient boosting model.
- Recall is lowest where risk is rare: IMD band 90-100% (0.3618) and unknown IMD band
  (0.3187), both flagged. At-risk disabled students are reached more often (0.5233 against
  0.4398).
- Many calibration flags at week 2 reflect the whole cohort being predicted too high
  (gap 0.0621 for all students). Some groups sit well beyond that, and disabled students in
  the most deprived IMD bands are predicted too low that week (see the model card).
- Adding demographics as inputs (`lr_with_demographics`) widens the IMD and disability gaps for
  about the same overall recall (it narrows the gender gap a little), which supports leaving
  them out.

The week-12 gate failure is also put in context: a paired bootstrap of gradient boosting minus
`lr_main` recall gives -0.0042 with a 95% range of -0.0097 to 0.0042
(`reports/model_gap_bootstrap.json`), so at week 12 the two models cannot be told apart on this
data. The gate result stands.

Outputs: `reports/fairness_report.md` (all tables), `reports/fairness_groups.csv`,
`reports/fairness_intersectional.csv`, `reports/fairness_summary.json` (settings, flag counts,
flagged groups) and `reports/model_gap_bootstrap.json`.

Limits:
- One test presentation, so the ranges cover the sampling of students, not change between years.
- Many groups, weeks and models are compared, so some flags will appear by chance. Patterns
  that repeat matter more than single flags.
- The audit was run five times. The second run added an "All students" row to every table.
  The third fixed the bootstrap, which drew different students on each run because the feature
  query returned rows in no fixed order (the query now has a fixed order). Later runs changed
  only the summary and the Markdown layout. Point values were the same in every run, and the
  flag rules never changed. Details in the design note.

## Serving and deployment
Design: [`docs/design/day5-serving-deploy.md`](docs/design/day5-serving-deploy.md).
Setup and teardown: [`docs/deploy.md`](docs/deploy.md).

```
retrain.yml (GitHub Actions, manual)
  data checks -> baseline -> gradient boosting + gate -> fairness audit -> release
  -> monitoring report (drift + outcomes on 2014J) -> reports/ must come out unchanged
  -> image: build, scan, push to ECR -> Terraform: Lambda + HTTP API -> call the live API
  -> score.yml: risk lists + drift.json from the live image -> private S3 bucket
```

**Release.** `scripts/release_model.py` registers `lr_main` in MLflow as `champion` for each
week (fitted exactly as on Day 2; the gradient boosting models stay `challenger`), then copies
the champion files into a bundle with a manifest of their SHA-256 hashes. It loads the bundle
the way the service does, rescores 2014J and stops unless every number matches
`reports/baseline_metrics.json`, so serving changes no model and no number. It also
refuses a champion that is not a logistic regression, because the service explains scores with
coefficients.

**Reasons.** Each score comes with up to three reasons from the model's own coefficients:
an input's push on the log-odds compared with an average training student (exact: the pushes
plus the intercept give the model's output, and a test checks this). Inputs that move together
are reported as one reason, because on their own their coefficients can point the wrong way:
at week 8 total clicks has a small positive coefficient (0.099) while weeks active has a
negative one (-0.4179) (`reports/baseline_coefficients.csv`), which would read as "more clicks
raise the risk". Assessments due, submitted and missed are also one reason, since missed is
always due minus submitted. The module is never a reason. Reasons read like these:
- `submitted <n> of <m> assessments due so far`
- `no assessment score yet`
- `course-site activity so far: clicks <n>, active days <n>, clicks last week <n>`

**Batch job.** `scripts/score_batch.py --week 12 --presentation 2014J` builds the cohort with
the training feature query (data up to the cutoff day only) and writes one CSV per module: the
top 20% by risk, with rank, student id, risk and reasons, and a manifest with the model version
and counts. No outcome or demographic column is written.

**API.** FastAPI (`src/early_warning/api.py`). `GET /health` gives the model versions;
`POST /v1/score` takes a week and 1 to 500 students and returns risk and reasons. Pydantic
rejects unknown fields (so demographic fields cannot reach the model), unknown modules, negative
counts and counts that cannot occur together (for example missed not equal to due minus
submitted, or a registration after the cutoff day). A test sends every row of the test-sample
cohort, at every week, through the API, so the checks cannot reject rows the feature query
makes, and the API's risks must equal the model's own. Example request:
`docs/examples/score_request.json`.

**Image.** Two-stage `Dockerfile`: locked runtime dependencies only (no MLflow, SHAP or dev
tools), base images pinned by digest, a non-root user, read-only at run time. The AWS Lambda
Web Adapter lets the same image run uvicorn locally and on Lambda. CI builds it with a bundle
made from the test sample, calls both routes and the batch job, and scans it with Trivy.

**AWS.** Terraform in two stacks. `infra/foundation` (applied once by a person) holds what is
long-lived or sensitive: the GitHub OIDC provider, three roles, the ECR repository, the state
and risk-list buckets (private, encrypted, HTTPS only, lists deleted after 90 days) and a budget
alarm. `infra/service` (applied by the deploy job) holds the Lambda function and the rate-limited
HTTP API. GitHub Actions signs in with OIDC, so there are no AWS keys anywhere; each job's role
can do only that job, and each role trusts one GitHub environment (`production` for deploys,
`scoring` for the lists). CI runs `terraform validate` and a Trivy misconfiguration scan;
findings accepted on purpose are listed with reasons in `.trivyignore`.

Limits:
- Not deployed yet: it needs a new AWS account (`docs/deploy.md`). The deploy role's action
  list has not been run against a live account.
- The MLflow store in CI is a workflow artifact, not a shared server.
- Each retraining run rescores 2014J with the fixed Day 2 to 4 settings. That checks that the
  results reproduce; nothing is chosen from it, and the run fails if any report changes.

## Monitoring
Design: [`docs/design/day6-monitoring-freeze.md`](docs/design/day6-monitoring-freeze.md).

Two questions, asked at different times:
1. **Each week, before any outcome is known: do this cohort's inputs and risk scores look like
   the training cohort's?** The release step saves, with the model, the training cohort's
   profile at each week: decile bins of every input and of the model's own scores, with a bin
   for missing values, and the share of modules. The batch job bins the new cohort the same
   way and writes `drift.json` next to the lists: the Population Stability Index (PSI) of each
   input and of the score. Bands: below 0.1 stable, 0.1 to 0.25 moderate, 0.25 or more major.
   Status: `alert` if any PSI is major, `watch` if any is moderate. Drift never stops the job;
   `score.yml` puts the status in the run summary and, on `alert`, raises a warning for a
   person to review.
   `drift.json` holds bin shares and means of the whole cohort, never a student row.
2. **After the presentation ends: did the lists perform as expected?** Recall and precision of
   the top-20% lists, at-risk rate and calibration, compared with the development check (the
   same pipeline trained on 2013B and 2013J and scored on 2014B, `reports/dev_metrics.json`).
   Status `alert` if recall falls below 0.9 times the expected value or if mean predicted
   risk is more than 0.05 away from the observed rate.

The thresholds and the action for each status were fixed before the first run.
`scripts/monitor.py --mode test` runs both checks on 2014J, the presentation the released
model has not seen, and the retraining workflow reruns it, so the report must reproduce.

Results for 2014J (`reports/monitoring.json`, tables in `reports/monitoring_report.md`):

| Week | Score PSI | Largest input PSI | Drift | Recall@20% (expected) | At-risk rate (expected) | Mean risk minus observed rate | Outcomes |
|---|---|---|---|---|---|---|---|
| 2 | 0.0108 | `n_due` 0.4534 | alert | 0.3092 (0.3013) | 0.4165 (0.4961) | 0.0621 | alert |
| 4 | 0.0931 | `n_due` 0.4049 | alert | 0.355 (0.3451) | 0.4079 (0.4843) | 0.0335 | ok |
| 8 | 0.0462 | `n_due` 2.5427 | alert | 0.4135 (0.3896) | 0.3854 (0.4598) | 0.0105 | ok |
| 12 | 0.0184 | `n_due` 0.8704 | alert | 0.4496 (0.4263) | 0.3633 (0.4425) | 0.0391 | ok |

What this shows:
- **Drift raised an alert every week, and following it up finds a calendar change, not a
  change in students.** The inputs that moved most are assessments due by the cutoff day
  (`n_due`, major every week) and assessments submitted (`n_submitted`, major every week,
  0.3075 to 0.718). `n_due` is set by each module's assessment calendar, and the report's
  per-module table shows two calendars changed: by week 12, module BBB had 3.55 assessments
  due on average in training and 2.0 in 2014J, and DDD 3.45 and 3.0. Total clicks stayed
  stable (0.0075 to 0.0213). The average score was major at week 4 (0.2741) and moderate at
  the other weeks, and the module mix was moderate every week (0.1966 to 0.2008). Three more
  inputs were moderate in one week each: assessments missed at week 8 (0.1476), late
  submissions at week 12 (0.176) and clicks in the last week at week 12 (0.1542). The full
  table is in the report.
- **The scores hardly moved** (score PSI 0.0108 to 0.0931, all stable), and the lists reached
  more of the at-risk students than expected at every week. Because each module's list is its
  top 20%, a shift in inputs changes who is ranked first, not how many are listed.
- **Fewer students ended at risk in 2014J than in 2014B** (0.3633 against 0.4425 at week 12),
  so precision is lower than expected at every week (0.8154 against 0.9404 at week 12) even
  though recall is higher. Precision depends on how many at-risk students there are to find.
- **The outcome alert is at week 2:** the model's mean risk was 0.0621 above the observed
  rate, the same offset the fairness audit found for all students that week. By the rules this
  asks for a retraining review; the model card's first recommendation (recalibrate on the
  most recent presentation) is the matching fix. Nothing was changed here, because 2014J is
  the test presentation.

Limits:
- One monitored presentation, so there is no record yet of how often each status fires in
  normal years. The bands are the conventional ones, not tuned to this data.
- `n_due` and `n_submitted` follow the assessment calendar, so as set up they raise an alert
  whenever a module changes its calendar. A next version would check them within each module
  against that module's own calendar. The rules were not changed after the first run.
- PSI is computed over all modules together, so a change in the module mix also moves the
  inputs that differ by module.
- Expected values come from one development presentation (2014B), with no range.
- Outcomes are known only after a presentation ends, so during a teaching period only the
  drift check is available.

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
uv run python scripts/audit_fairness.py --mode test   # fairness audit, about 6 minutes
uv run python scripts/release_model.py --mode test  # lr_main as champion, bundle in build/model
uv run python scripts/monitor.py --mode test       # reports/monitoring.json, drift + outcomes
uv run python scripts/score_batch.py --week 12 --presentation 2014J   # lists + drift.json in build/risk-lists
MODEL_DIR=build/model uv run uvicorn early_warning.api:create_app --factory --port 8080
docker build -t early-warning . && docker run --rm -p 8080:8080 early-warning   # same API
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db   # browse runs and models
uv run pytest                             # runs on a small sample in tests/fixtures/
```

Optional: `uv run pre-commit install` runs ruff and a gitleaks secret scan before each commit.

## Layout
| Path | Contents |
|---|---|
| `src/early_warning/` | download, ingest, checks, views, features, metrics, baseline, gradient boosting, calibration, reasons, gate, registry, fairness audit, release, scoring, API, batch and monitoring code |
| `sql/checks/` | one SQL file per data rule; each returns the rows that break it |
| `sql/views/` | `student_module`, `vle_weekly`, `assessment_submissions` |
| `sql/features/` | `student_week.sql`: one row per student at a prediction week |
| `scripts/` | command-line entry points |
| `reports/` | outputs of the scripts; every number quoted in the docs comes from here |
| `docs/design/` | design notes |
| `docs/model_card.md` | model card for the model in use |
| `docs/v1-to-v2.md` | what changed from the 2025 notebook this project replaces |
| `docs/deploy.md` | AWS account setup, deployment and teardown |
| `Dockerfile` | the image for the API and the batch job |
| `infra/` | Terraform: `foundation` (once, by a person) and `service` (by the deploy job) |
| `.github/workflows/` | `ci.yml` (tests, image, Terraform), `retrain.yml` (retrain, gate, audit, deploy), `score.yml` (weekly lists) |
