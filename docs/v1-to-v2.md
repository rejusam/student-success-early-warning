# From the 2025 notebook (v1) to this project (v2)

In March 2025 I shared a notebook that predicted whether a student would pass or fail a course:
`student-performance-prediction/ML_Model_for_predicting_student_performance.ipynb` in
https://github.com/rejusam/university-data-science-projects. This project replaces it. This
note says plainly what was wrong with v1 and where v2 deals with each point. The v1 facts below
come from that notebook; the v2 numbers come from files in `reports/`.

## What v1 did
- Built a table of 100 made-up students with `np.random`: programme, department, courses taken,
  average grade, and a pass/fail label drawn at random (80% pass), with no random seed.
- Label-encoded programme and department as 0, 1, 2, split the 100 rows 80/20 at random, and
  trained a random forest, logistic regression, SVM, XGBoost and k-NN.
- Reported accuracy, precision and recall on the 20 test rows. The random forest, logistic
  regression and SVM each scored 0.85 accuracy, 0.85 precision and 1.00 recall.

## What was wrong with it
1. **There was nothing to learn.** The label was drawn at random, separately from the features,
   so no model could do better than guessing the most common class.
2. **The headline result was the most common class.** With "pass" as the positive class,
   recall 1.00 with accuracy equal to precision (0.85) means the models predicted "pass" for
   every student. The notebook called recall 1.0 excellent and added that the model might be
   over-predicting "pass"; in fact that was all it did.
3. **The wrong class was measured.** Recall counted passes found, but the purpose was to find
   students who would fail.
4. **Results could not be repeated.** The data had no seed, so every run made different
   students, and 20 test rows give no stable comparison between five models.
5. **The test was not like use.** A random split of one table says nothing about how a model
   trained on past students does on next year's.
6. **Notebook only.** No tests, no checks on the data, no record of which model was chosen or
   why, no fairness check, nothing that could run as a service.

## What v2 does about each point
| v1 problem | v2 | Where |
|---|---|---|
| Random labels on made-up data | Open University Learning Analytics Dataset (OULAD, CC BY 4.0), 32,593 student registrations with real outcomes; download checked against a fixed SHA-256 | `reports/data_profile.json`, `docs/data.md` |
| Measured the majority class | Target is Fail or Withdrawn. Headline is recall at the top 20% of each module's list, with precision beside it, compared with a no-model clicks ranking | `reports/baseline_metrics.json` |
| Results not repeatable | Fixed seeds and row order; the retraining workflow fails if any report changes | `.github/workflows/retrain.yml` |
| Random split | Train on 2013B, 2013J and 2014B; tune on 2014B only; test once on the later 2014J presentation. Features use only data dated before the prediction day, and a test checks this | `docs/design/day2-weekly-baseline.md`, `tests/test_features.py` |
| Model chosen by eye on 20 rows | A gradient boosting challenger had to beat the logistic regression at weeks 4, 8 and 12. It failed at week 12, so the simpler model stays | `reports/gate.json` |
| No fairness check | Recall, precision and calibration by gender, age, region, education, IMD band and disability, with bootstrap ranges, and a model card | `reports/fairness_report.md`, `docs/model_card.md` |
| Notebook only | Tests and CI, a container with an API and a weekly batch job, AWS infrastructure as code with keyless deployment, a retraining workflow, and drift and performance monitoring | README, "Serving and deployment" and "Monitoring" |

## What v2 does not fix
- OULAD is from the Open University in England, 2013 and 2014. None of v2 has been checked on
  University of Auckland data.
- The AWS deployment is written and validated, but not run yet (README, "Serving and
  deployment").
