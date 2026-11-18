# Day 2 design: weekly features and logistic regression baseline

Date: 2026-11-18. Status: approved by Reju.

## Why
An advisor needs a list of the students in a module who are most likely to fail or withdraw,
early enough to act on it. UoA already sends Canvas engagement reports to student advisors
(https://auckland.ac.nz/en/privacy/our-use-data-analytics-automated-decision-making.html).
Day 2 builds the first honest version of that list. Features use only what was known at the
prediction week, the model is a plain logistic regression, the test set is a later presentation,
and every result is saved to `reports/`. It is also compared with a no-ML ranking by clicks.
Otherwise later models have nothing fair to be compared against.

## Goals
- At weeks 2, 4, 8 and 12 of a module presentation, score P(`at_risk`) for each student still
  enrolled. `at_risk` = `final_result` is Fail or Withdrawn (`sql/views/student_module.sql`).
- Feature SQL that cannot see data dated after the cutoff, backed by a test that proves it.
- A time-aware split: train on 2013B, 2013J and 2014B, test once on 2014J.
- Metrics per week in `reports/`, with recall@top-20% per module presentation as the headline.

## Non-goals (Day 2)
- Gradient boosting, calibration, SHAP, hyperparameter tuning, MLflow (Day 3).
- The fairness audit by subgroup (Day 4). Day 2 only fits the comparison model with
  demographics added. It does not report gaps by group.
- API, Docker, AWS.

## Anti-goals
- Any feature that uses a row dated after the cutoff, or `date_unregistration` / `final_result`
  as an input.
- Tuning on, or looking at the test presentation more than once. A single final run fills `reports/`.
- Quoting any number in docs that is not in a `reports/` file.

## Decisions (settled with Reju, 2026-11-18)
1. **Prediction weeks: 2, 4, 8, 12.** "Week k" means scoring at the start of week k with data
   up to the end of week k-1. Cutoff day = 7k - 1, which gives days 13, 27, 55 and 83. Weeks
   follow `vle_weekly` (`week = floor(date / 7)`), so a filter of `week <= k - 1` matches
   `date <= 7k - 1`. VLE activity before day 0 counts, because it happened before the cutoff.
   Week 2 falls before most first assessments, so it is mostly clicks only. Week 4 is around
   the first assessment. Weeks 8 and 12 show what more evidence adds. Each week gets its own
   model.
2. **Cohort rule.** A student is in the week-k cohort unless `date_unregistration <= cutoff`.
   Students who leave later stay in and are labelled at risk. Students who are Withdrawn with no
   unregistration date (93 in the full data, `reports/data_profile.json`,
   `has_unregistration_date_by_final_result`) stay in, because their leaving cannot be seen.
   `date_unregistration` is used for this filter and nothing else.
3. **Demographics are audit-only.** The main model leaves out `gender`, `age_band`, `region`,
   `highest_education`, `imd_band` and `disability`. A comparison model per week adds them, and
   both sets of metrics are saved. That shows the accuracy cost of leaving them out (UoA
   Learning Analytics Principles and Policy:
   https://auckland.ac.nz/en/about-us/about-the-university/policy-hub/enabling-environment/digital/data-governance-analytics/learning-analytics-principles-policy.html).
4. **Headline metric: recall@top-20%, with precision@top-20% next to it.** Students are ranked
   within each module presentation, because each module would get its own list. Recall and
   precision are pooled over all test module presentations (sum of true positives / sum of
   positives). Also reported: the same at 10% and 30%, AUC-PR (average precision), ROC-AUC,
   cohort size and base rate. Advisor capacity at UoA is UNVERIFIED, so k is a share of the
   cohort, not a head count.
5. **Rankings compared each week:** `lr_main`, `lr_with_demographics`, and `clicks_rank`. The
   clicks ranking orders students by `clicks_total`, lowest first, with no model.
6. **If `lr_main` does not beat `clicks_rank`, the result is reported as it is.** There is no
   second try on 2014J and no feature changes made to win. The README states, week by week,
   where LR beats the clicks ranking and where it does not. On Day 3, a new model must beat
   both `lr_main` and `clicks_rank` on recall@top-20% at weeks 4, 8 and 12. Week 2 is
   reported but not gated.

## Approaches considered
1. **SQL-first (chosen).** One parameterised file, `sql/features/student_week.sql`, takes
   `$cutoff` and returns one row per student in the cohort. Python runs it for each week and
   fits a scikit-learn Pipeline. This follows the Day 1 pattern (`sql/views/`), and the cutoff
   filter sits in one place a reader can check.
2. Feature code in pandas. It is familiar, but it moves away from Day 1's SQL layer, and the
   cutoff filters end up spread over several functions.
3. A long table of all weeks built once, then filtered. It is quicker to re-run, but it keeps
   future weeks next to each row, which is the leakage this design is trying to rule out.

## Features (main model)
All features are computed per (`code_module`, `code_presentation`, `id_student`) at the cutoff.

| Group | Feature | Definition |
|---|---|---|
| Engagement | `clicks_total` | sum of `vle_weekly.clicks`, `week <= k-1` (includes weeks before day 0) |
| | `clicks_last_week` | clicks in week k-1 |
| | `active_days_total` | sum of `active_days`, `week <= k-1` |
| | `weeks_active` | count of weeks 0..k-1 with any clicks |
| | `clicks_before_start` | clicks in weeks < 0 |
| Assessment | `n_due` | non-exam assessments of the module with `due_date <= cutoff` |
| | `n_submitted` | of those, submitted with `date_submitted <= cutoff` |
| | `n_missed` | `n_due - n_submitted`. A submission after the cutoff counts as missed at the cutoff |
| | `n_late` | submitted by the cutoff with `days_late > 0`, not banked |
| | `mean_score` | mean `score` over submissions with `date_submitted <= cutoff`. NULL if none |
| Study record | `num_of_prev_attempts`, `studied_credits`, `date_registration` | from `student_module` |
| | `code_module` | one-hot |

Students with no VLE rows get 0 for the engagement features. `lr_with_demographics` adds the six
demographic fields, one-hot encoded, with missing `imd_band` as its own category.

**Marking lag.** OULAD does not say when a score became visible to staff. A score counts once
`date_submitted <= cutoff`. This may be a little optimistic, and the README will say so.
The alternative is a guessed lag (for example 7 days), and the data gives no basis for one.

## Model
scikit-learn Pipeline: median impute plus a missing-value flag for `mean_score`, standard scaling
for the numeric columns, one-hot for the categorical ones, then `LogisticRegression` with the
default regularisation, `max_iter` high enough to converge, and no class weights. There is no
tuning on Day 2, and no validation presentation is needed. One model is fitted per week and per
feature set, on training presentations 2013B, 2013J and 2014B, and scored once on 2014J.

## Leakage tests
- **Truncation test** on the fixture. For each cutoff, compute the features. Then delete every
  `student_vle` row with `date > cutoff` and every `student_assessment` row with
  `date_submitted > cutoff`, and compute again. The two results must be identical.
- **Cohort test**: no student with `date_unregistration <= cutoff` appears in the week's rows.
- **Column test**: the feature output has no `date_unregistration`, `final_result` or
  demographic column, except in the demographics feature set.
- **Metric test**: recall/precision@top-k checked on a small hand-worked example, including
  rounding of k within each module presentation.

## Outputs
- `reports/cohort_by_week.json`: cohort size and base rate per week and presentation, plus
  student counts per presentation (these replace the one-off counts in the plan).
- `reports/baseline_metrics.json`: per week, per ranking, every metric listed in decision 4.
- `reports/baseline_coefficients.csv`: LR coefficients per week (main model), so each can be
  read as a reason.
- Script `scripts/train_baseline.py`. Logic lives in `src/early_warning/` (features, metrics).
- `docs/data.md` / README text quote only these files.

## Constraints
- Python 3.12, uv. Add `scikit-learn` and lock it in `uv.lock`. DuckDB:
  `duckdb.execute(...).fetchone()` style.
- Tests run on the committed fixture in CI (AAA and BBB 2013J, 80 students). The full run is
  local only, because data is not in git.
- Code simple enough for Reju to explain line by line.

## Builder calls (made while writing this design, Reju to check)
- **Late registrants.** Students with `date_registration > cutoff` are not yet enrolled at the
  cutoff and are left out of that week's cohort. Counts per week and presentation are in
  `reports/cohort_by_week.json` (`not_yet_registered`).
- **Banked submissions** count as submitted and their `score` is used. They are known when the
  student registers. They are left out of `n_late` only.
- **NULL `date_registration`** (45 rows) is median-imputed by the pipeline.
- **Dev check before the test run.** Train on 2013B and 2013J, score 2014B, with the same code.
  This catches pipeline bugs (empty cohorts, broken joins, no convergence) on real data without
  touching 2014J. Results go to `reports/dev_metrics.json`, labelled as a development check.
  Feature or model changes made after the dev check are listed in the README.
- **CI** runs the unit tests (truncation, cohort, column, metric), plus a smoke test that fits
  the pipeline on the fixture with no presentation split and writes to a temporary directory.
  `scripts/train_baseline.py` takes the output directory as an argument, so a sample run never
  overwrites `reports/`.

## Left to the builder
- Exact SQL structure (CTEs), function names, test names, JSON layout of the reports.
- How k is rounded per module presentation (state the rule in code and in the metric test).
- Tie-breaking in `clicks_rank` (it must be deterministic, for example by `id_student`).
