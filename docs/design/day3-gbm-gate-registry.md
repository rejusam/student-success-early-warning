# Day 3 design: gradient boosting, calibration, reasons, evaluation gate, model registry

Date: 2026-11-18. Status: decisions below are open to review.

## Why
The Day 2 logistic regression (`lr_main`) is the model to beat. Day 3 adds a second model
family, but only lets it replace the baseline if it does better on the held-out presentation.
A university team would run the same steps before a model reaches an advisor: tune on a
validation set, calibrate the probabilities, explain each score, check the result against an
agreed gate, and record the model and its evidence in a registry.

## Goals
- A gradient boosting model per prediction week (2, 4, 8, 12), using the same features and
  cohort as `lr_main` (`sql/features/student_week.sql`). No demographic fields.
- Hyperparameters chosen on a validation presentation, never on 2014J.
- Calibrated probabilities, so "0.7" means about 7 in 10 students with that score end up at risk.
- Up to three reasons per student, from SHAP values.
- An evaluation gate, run by the training script, with the result saved to `reports/gate.json`.
- Every run logged to MLflow. Each weekly model goes in the MLflow model registry, and gets
  the alias `champion` only if the gate passes.

## Non-goals
- Fairness audit by subgroup (Day 4). Deployment, API and scheduled retraining (Day 5).
- A model with demographics. The Day 2 comparison already shows what they add.
- New features. Day 3 changes the model, not the inputs, so the comparison with Day 2 is fair.

## Anti-goals
- Any choice (hyperparameters, calibration method, model family) made by looking at 2014J.
- Changing anything after the 2014J run to make the gate pass. If it fails, the result is
  reported as it is, the same rule as Day 2.
- A gate that can pass on numbers that were not computed in the same run.

## Decisions
1. **Model: scikit-learn `HistGradientBoostingClassifier`.** It is a histogram-based gradient
   boosting model of the same kind as LightGBM. LightGBM and XGBoost were tried first. Both
   fail to import on macOS without a system OpenMP library (`libomp`), which `uv sync` cannot
   install. The scikit-learn version needs no extra system library, is already pinned, and
   keeps the Docker image on Day 5 simple. It handles missing values itself, so `mean_score`
   and `date_registration` need no imputing. `code_module` is one-hot encoded and not passed
   as a native categorical, because SHAP values for native categorical splits do not add up
   to the model output (checked on 2026-11-18 with shap 0.54.1).
2. **Tuning on 2014B.** For each week, fit on 2013B and 2013J and score 2014B. The grid is
   `max_leaf_nodes` in {15, 31, 63}, `min_samples_leaf` in {20, 100}, `l2_regularization` in
   {0, 1}, with learning rate 0.05 and up to 500 trees. The number of trees is picked from the
   validation log loss after each tree (`staged_predict_proba`). The setting with the lowest
   validation log loss wins. Log loss is used instead of recall@top-20%, because the top-k
   metric is a step function that moves with a handful of students, so tuning on it would fit
   noise. The chosen settings are saved to `reports/gbm_tuning.json`, and the test run reads
   them from that file. It cannot tune.
3. **Final fit and calibration.** The chosen settings are refitted on 2013B, 2013J and 2014B.
   The probabilities are calibrated with Platt scaling (`CalibratedClassifierCV`,
   `method="sigmoid"`, `ensemble=False`). The calibration data comes from leave-one-
   presentation-out predictions: train on two presentations, predict the third, three times.
   That matches how the model is used, on a presentation it has not seen. Platt scaling is a
   strictly increasing function of the score, so the order of students, and so the top-k
   lists, are the same as the uncalibrated model's. Isotonic regression was not chosen,
   because its flat steps create tied scores, and ties change the top-k lists.
4. **Calibration measures:** Brier score and expected calibration error (ECE, 10 equal-size
   bins), for `lr_main`, the uncalibrated and the calibrated boosting model. A table of mean
   predicted against observed rate per bin is saved. These are reported but not gated.
5. **Reasons per student: SHAP.** `shap.TreeExplainer` on the uncalibrated boosting model
   gives each feature's push on the log-odds of risk for each student. The values add up
   exactly to the model's output, and a test checks this. A student's reasons are the three
   features with the largest positive push, written in plain words with the student's value
   (for example "missed 2 assessments due so far"). `code_module` is never a reason, because
   an advisor cannot act on it. Mean absolute SHAP value per feature and week is saved as
   the global view. The LR coefficients from Day 2 stay as they are.
6. **Gate.** The boosting model passes if its recall@top-20% on 2014J beats both `lr_main`
   and `clicks_rank` at weeks 4, 8 and 12 (agreed on Day 2). Week 2 is reported but not gated.
   The four weekly models go to an advisor together, so the gate is passed or failed by the set.
   Ties count as a fail ("beat" means strictly higher).
7. **Comparison numbers come from the same run.** The script refits `lr_main` and recomputes
   `clicks_rank` on the same 2014J cohort, in the same process. It then checks that these
   match `reports/baseline_metrics.json` exactly, and stops with an error if they do not.
   That catches a data or code change since Day 2. Refitting the deterministic LR baseline
   is not a second look at 2014J for tuning. Nothing is chosen from it.
8. **Where the gate runs.** In the training script (`scripts/train_gbm.py`). It writes
   `reports/gate.json` and exits with code 1 if the gate fails, so a scheduled retraining job
   (Day 5) can stop on a failure. CI runs the same code end to end on the test sample in
   smoke mode, and unit tests check the gate logic. CI does not run the real gate, because
   the full data is not in git.
9. **MLflow: local, SQLite.** Tracking store `sqlite:///mlruns/mlflow.db`, artifacts under
   `mlruns/` (gitignored). No server, because there is one user and no shared machine yet.
   The tracking URI is a command-line option, so a hosted server on Day 5 needs no code
   change. Each week's run logs the parameters, the test metrics, the gate
   result, the validation log loss of the chosen setting, the git commit and the OULAD file
   checksum, and the fitted model with an input
   signature. The model is registered as `early-warning-week-<k>`. Alias `champion` if the
   gate passed, `challenger` if not.
10. **Per-module AUC.** The Day 2 AUC-PR and ROC-AUC pool all modules into one ranking, while
    top-k is per module. Day 3 reports both: the pooled numbers (to compare with Day 2) and
    the mean of the per-module numbers, which matches how the lists are used.

## Approaches considered
1. **Separate script, shared modules (chosen).** `scripts/train_gbm.py` with logic in
   `src/early_warning/{gbm,calibration,reasons,gate,registry}.py`, reusing `features`,
   `metrics` and `baseline.build_pipeline`. The Day 2 baseline stays as it is and can be rerun
   alone.
2. **Extend `baseline.py`** with a model argument. Fewer files, but it would change the
   Day 2 code and outputs that the README already quotes.
3. **One MLflow project and pipeline framework** (MLflow Projects, Kedro or similar).
   More structure than a seven-day project needs, and harder to explain line by line.

## Outputs
| File | Contents |
|---|---|
| `reports/gbm_tuning.json` | grid results on 2014B per week, chosen settings and tree count |
| `reports/gbm_validation.json` | on 2014B: tuned model, `lr_main` and `clicks_rank`, all metrics |
| `reports/gbm_metrics.json` | on 2014J: the same rankings plus `gbm_uncalibrated`, all metrics |
| `reports/gbm_calibration.json` | per week and model: Brier, ECE, bin table |
| `reports/gate.json` | per gated week: the three recalls, pass or fail; overall result |
| `reports/gbm_shap_importance.csv` | mean absolute SHAP per feature and week on 2014J |
| `reports/example_risk_list.csv` | top 10 students of one module at week 8 with probability and reasons |

## Modes of `scripts/train_gbm.py`
- `--mode tune`: train 2013B and 2013J, validate on 2014B, write `gbm_tuning.json` and
  `gbm_validation.json`. No MLflow registration.
- `--mode test`: read `gbm_tuning.json`, refit on three presentations, score 2014J once,
  write the rest, log to MLflow, register, set the alias, exit 1 on a failed gate.
- `--mode smoke`: both steps on the test sample with a one-setting grid, output and MLflow
  store in a given directory. Used by CI.

## Tests
- Gate: pass, fail on one week, tie counts as fail, week 2 ignored.
- Calibration: same order as the uncalibrated scores; probabilities in [0, 1].
- SHAP: values plus base value equal the model's decision function; `code_module` never a
  reason; reasons only for positive pushes.
- Model inputs: no demographic, `date_unregistration`, `final_result`, `at_risk` or
  `id_student` column.
- Comparison check: a changed baseline number raises an error.
- Registry: on a temporary SQLite store, a passing gate gives alias `champion`, a failing one
  `challenger`.
- Smoke: `main([... "--mode", "smoke"])` writes only to the given directory.

## Constraints
- Same as Day 2: Python 3.12, uv lock, code simple enough to explain line by line, every
  number quoted in docs from `reports/`.
- New dependencies: `shap`, `mlflow`, both pinned in `uv.lock`.

## Left to the builder
- JSON layout of the reports, function names, wording of the plain-language reasons.
- Which module is shown in `example_risk_list.csv` (the first in alphabetical order that has
  a 2014J presentation).
