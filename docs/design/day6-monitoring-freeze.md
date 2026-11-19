# Day 6 design: monitoring, README numbers, v1 to v2 note, freeze

Date: 2026-11-19. Status: decisions below are open to review.

## Why
Days 1 to 5 built, checked and packaged a model. A model in use also needs someone to notice
when the students it scores stop looking like the students it learned from, and when its
lists stop reaching the students who go on to fail or withdraw. Day 6 adds that check, makes
sure the README quotes only numbers from `reports/`, writes down honestly how this project
differs from the 2025 notebook it replaces, and freezes the MVP.

## Goals
- A monitoring step that answers, for each prediction week of a new presentation:
  1. Do this cohort's model inputs look like the training cohort's? (input drift)
  2. Do its risk scores look like the training cohort's? (score drift)
  3. Once outcomes are known: did the lists perform as expected? (realised performance)
- The same code runs in two places: inside the scoring container, next to each weekly risk
  list (no outcomes, no training data available there), and as a report over 2014J with
  outcomes, written to `reports/`.
- Fixed thresholds and a written action for each status, set before the first run.
- README: Status and a Monitoring section, every number traceable to a file in `reports/`.
- `docs/v1-to-v2.md`: what the 2025 notebook did, what was wrong with it, what changed.
- Freeze: full test suite, lint, retrain chain reproducible, CI green, a version tag.

## Non-goals
- A monitoring service, dashboard or alerting system (CloudWatch alarms, Evidently server).
  The outputs are JSON and Markdown files, and a warning in the GitHub Actions run summary.
- Logging individual API requests for monitoring. The weekly list is the product; the API is
  scored on the same cohort and model, so the batch run is the monitored path.
- Per-module drift. Lists are per module, so a cohort that is larger or smaller in one module
  still gets a 20% list; the module mix is still checked as one input.
- Retraining automatically when drift is found.

## Anti-goals
- Any change to a model or to an earlier number in `reports/`. Monitoring reads the released
  bundle and the data; it fits nothing.
- Student-level data in a drift file. Drift files hold bin shares and summary statistics of
  whole cohorts only, never an id, an outcome or a demographic field.
- Thresholds or rules changed after seeing the 2014J result.
- A README number that is not in a file under `reports/`.

## Approaches considered
1. **Population Stability Index (PSI) per input and on the score, against a reference profile
   saved with the model (chosen).** The release step stores, for each week, the training
   cohort's bin edges and shares for every input and for the model's own scores. Any later
   cohort is binned with the same edges. PSI is the standard drift measure in credit-risk
   model monitoring and is used by monitoring tools such as Evidently and SageMaker Model
   Monitor; it gives one number per input that does not grow with sample size.
2. **Statistical tests (Kolmogorov-Smirnov, chi-squared) on raw training data.** With
   thousands of students per week (8,571 to 9,333 in 2014J, `reports/monitoring.json`) almost
   any difference is "significant", so p-values would
   flag everything; and the container would need the training data. Rejected.
3. **A third-party monitoring library (Evidently, NannyML).** Adds a large dependency to the
   image for a calculation of about 40 lines, and its defaults would need explaining in an
   interview anyway. NannyML's performance estimation without labels is interesting but adds
   a second model to explain. Rejected for the MVP.

## Decisions
1. **Reference = the training cohort of the released model, per week.** For week k that is
   the 2013B, 2013J and 2014B students in the week-k cohort, and the scores are the released
   model's own predictions for them. Stored in the bundle as `reference.json`, with its
   SHA-256 in `manifest.json`, so the drift check and the model always travel together and
   the container needs no training data.
2. **Bins.** Numeric inputs: edges at the reference deciles (duplicates dropped, so a
   mostly-zero input such as `n_late` gets fewer bins), open at both ends, plus one bin for
   missing values (`mean_score` is missing until a first score). `code_module`: one bin per
   module seen in training, plus one for any unseen module. Score: reference deciles.
3. **PSI** = sum over bins of (current share - reference share) * ln(current / reference),
   with empty shares replaced by 0.0001 so the log is defined. Bands, the usual ones in
   credit scoring: below 0.1 `stable`, 0.1 to 0.25 `moderate`, 0.25 or more `major`.
4. **Status of a weekly run (no outcomes yet):** `alert` if the score PSI or any input PSI is
   major; `watch` if any is moderate; else `ok`. Also reported per input: mean and missing
   share in reference and current.
5. **Realised performance (once the presentation has ended):** for each week, recall and
   precision at the top 20%, observed at-risk rate, mean predicted risk and its gap to the
   observed rate (calibration in the large), Brier score and ECE. Expected values come from
   the development check (`reports/dev_metrics.json`: the same pipeline trained on 2013B and
   2013J and scored on 2014B), the only out-of-time estimate available before release.
   Status `alert` if recall@top-20% is below 0.9 times the expected value or the calibration
   gap is over 0.05 either way (the same 0.05 as the fairness audit); else `ok`. The script
   stops if the realised recall and precision differ from `reports/baseline_metrics.json`,
   so the monitor and the Day 2 test agree.
6. **What a status asks of a person** (written into the report): `watch` = note it and look
   again next week; `alert` on drift = check the data pipeline for a change in recording
   first, then compare the list with last year's in the affected modules; `alert` on
   realised performance = a retraining review: retrain with the newest presentations, run
   the gate and the fairness audit, and promote only if both pass. Nothing is retrained or
   promoted automatically, which matches UoA's Learning Analytics Principles (a person
   decides).
7. **Where it runs.**
   - `early_warning.batch` writes `drift.json` next to each week's lists (and so to S3 in
     `score.yml`). `score.yml` puts the status in the run summary and raises a GitHub
     warning annotation on `alert`. The batch job does not fail on drift: the advisors still
     get their list, and a person reviews.
   - `scripts/monitor.py --mode test` writes `reports/monitoring.json` and
     `reports/monitoring_report.md` for 2014J, all four weeks, drift and realised
     performance. `--mode smoke` runs on the test sample and refuses to write to `reports/`.
   - `retrain.yml` runs `monitor.py --mode test` after the release, so the existing
     "reports must come out unchanged" step covers the monitoring report too.
8. **Freeze.** Tag `v0.1.0` (the version in `pyproject.toml`) on main after CI is green.
   After the tag, changes are fixes only, until the application is in.

## v1 to v2 note
`docs/v1-to-v2.md`, plain and factual, linked from the README. Facts from the 2025 notebook
(`ML_Model_for_predicting_student_performance.ipynb` in
https://github.com/rejusam/university-data-science-projects): 100 synthetic students, the
pass/fail label drawn at random with no link to the features, no random seed for the data, a
20-student test split; logistic regression and SVM reported 0.85 accuracy and 1.00 recall,
which with a label drawn at random means they predicted "pass" for everyone; the positive
class was "pass", so recall measured passes, not students at risk. Then a table of what v2
does about each point, pointing to the file that shows it.

## Left to the builder
- Function and file names, report layout, test cases.
- Exact wording in the README and the model card's new monitoring section.

## Changes during the build
1. **Per-module table of assessments due, added after the first run.** The first 2014J run
   raised a drift alert at every week, and the largest PSI was always `n_due`. Following the
   report's own advice (check how assessments are recorded), the cause was module assessment
   calendars that changed in 2014J. So the report gained a drill-down: for each module and
   week, the mean `n_due` over the training presentations and its value in 2014J
   (`assessments_due` in `reports/monitoring.json`). This adds information only: the
   thresholds, the statuses and every PSI are the same as in the first run.
2. **Monitoring reads the training presentations from the baseline's test mode** rather than
   from the bundle, which does not record them. The bundle's reference profile is still the
   only source for every PSI.

## Results (2014J, `reports/monitoring.json`)
- Drift status `alert` at every week, from `n_due` and `n_submitted` (and `mean_score` at
  week 4, 0.2741); score PSI stable at every
  week (largest 0.0931, week 4).
- Outcome status `ok` at weeks 4, 8 and 12; `alert` at week 2 from calibration (mean risk
  0.0621 above the observed rate). Recall above the expected value at every week.
- Two runs gave byte-identical reports; the batch job's `drift.json` for week 12 equals the
  report's week-12 drift entry apart from its extra `presentation` field.
