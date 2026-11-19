# Model card: weekly early-warning model (`lr_main`)

Sections follow Mitchell et al. (2019), "Model Cards for Model Reporting"
(https://arxiv.org/abs/1810.03993). Every number below names the file in `reports/` it comes
from. Last updated 2026-11-19.

## Model details
- **What it is:** four logistic regression models, one for each prediction week (2, 4, 8 and 12
  of a module presentation). Each gives every enrolled student a probability of ending the
  module with Fail or Withdrawn.
- **Inputs:** clicks in the virtual learning environment, assessment submissions and scores,
  prior attempts, credits studied, registration day and module. Only data dated on or before
  the day before the prediction week is used. Gender, age band, region, highest education,
  deprivation band (IMD) and disability are **not** inputs.
- **Training:** scikit-learn `LogisticRegression` with default settings after median
  imputation, scaling and one-hot encoding (`src/early_warning/baseline.py`). Trained on the
  2013B, 2013J and 2014B presentations.
- **Why this model and not the gradient boosting challenger:** a gradient boosting model was
  built on Day 3 and had to beat this model and a clicks-only ranking at weeks 4, 8 and 12 to
  replace it. It failed at week 12 (see Quantitative analyses), so it is registered in MLflow
  as `challenger` and not used. This model is registered as `champion` (tag `role=incumbent`)
  in the same registered models, `early-warning-week-<k>` (`src/early_warning/release.py`).
- **How it is served:** a weekly batch job writes each module's top-20% list with reasons, and
  an API scores students sent to it. Both run the same container image, built from the
  registered `champion` files, and the release step stops unless those files reproduce
  `reports/baseline_metrics.json` exactly. Details: README, "Serving and deployment".
- **Reasons given with each score:** up to three, from the model's own coefficients: each
  input's push on the log-odds compared with an average training student, largest upward push
  first (`src/early_warning/explain.py`). Inputs that move together are reported as one
  reason: assessments due, submitted and missed; and the five course-site activity counts. On
  their own their coefficients can point the wrong way (at week 8, total clicks +0.099 next
  to weeks active -0.4179, `reports/baseline_coefficients.csv`), which would read as "more
  clicks raise the risk". The module is never a reason.
- **Built by:** Reju Sam John, as a portfolio project. Code: this repository. Licence: MIT.

## Intended use
- **Use:** at the start of weeks 2, 4, 8 and 12, give an advisor one list per module
  presentation, ranked by risk, so they can decide whom to contact first. The headline
  setting is the top 20% of each list.
- **Users:** student advisors and the analysts who support them.
- **Not for:** any automatic decision about a student (grades, progression, funding,
  sanctions); ranking students across different modules; judging staff or modules; use on a
  different institution's data without re-checking everything in this card. A person decides
  what happens to each student, in line with the University of Auckland's Learning Analytics
  Principles and Policy, which treats data as an incomplete view of a student and keeps a person
  in any individual decision
  (https://auckland.ac.nz/en/about-us/about-the-university/policy-hub/enabling-environment/digital/data-governance-analytics/learning-analytics-principles-policy.html).

## Factors
The audit looks at six student attributes from the data: gender, age band, region, highest
education, IMD band (Index of Multiple Deprivation) and disability. The OULAD description
(`data/raw/OULAD.names`) does not say which end of the IMD scale is most deprived, so that is
UNVERIFIED for this dataset. This card reads 0-10% as the most deprived, as in the English
Indices of Deprivation, where decile 1 is the most deprived 10% of areas
(https://www.gov.uk/government/statistics/english-indices-of-deprivation-2019). The data agrees:
at week 12 the at-risk rate falls from 0.4409 in the 0-10% band to 0.2851 in the 90-100% band
(`reports/fairness_groups.csv`). A missing IMD band is its own group, "Unknown". One combination was also checked: disability
by IMD band (in three bands). These are the attributes the dataset records. It has no
ethnicity field, so the card says nothing about ethnicity.

## Metrics
- **Recall at the top 20%** (headline): of the students who went on to fail or withdraw, the
  share the top 20% of their module's list reached. For a group, the share of that group's
  at-risk students who were on the list. Equal recall across groups is "equal opportunity"
  (Hardt, Price and Srebro 2016, https://arxiv.org/abs/1610.02413).
- **Precision at the top 20%:** the share of listed students who were really at risk.
- **Calibration gap:** mean predicted risk minus the observed at-risk rate. Positive means
  the model predicts too high.
- **ROC-AUC:** how well the model orders students, with no cut-off. 0.5 is chance, 1 is perfect.
- **Ranges:** 95% bootstrap intervals from 1,000 resamples of students within each module
  presentation, with the lists rebuilt in each resample (`reports/fairness_summary.json`,
  "settings").
- **Review flags**, fixed before the audit ran: a group is flagged when its recall is below
  0.8 or above 1.25 times the largest group's and the range of the difference excludes 0, or
  when its calibration gap is over 0.05 either way and its range excludes 0. Groups with fewer
  than 100 students or 30 at-risk students are labelled small and never flagged. A flag asks a
  person to look; it is not a finding of unfairness. With 44 groups, four weeks and three
  models, some flags will appear by chance.

## Evaluation data
The 2014J presentation of the Open University Learning Analytics Dataset (OULAD; Kuzilek,
Hlosta and Zdrahal 2015, https://doi.org/10.24432/C5KK69, CC BY 4.0), held out from training and
tuning. Students who had unregistered or not yet registered by the prediction day are left out.
Cohort sizes (`reports/baseline_metrics.json`): 9,333 at week 2, 9,214 at week 4, 8,879 at
week 8 and 8,571 at week 12.

## Training data
2013B, 2013J and 2014B presentations of the same dataset, same cohort rule. The share of
students at risk differs between presentations, for example at week 2: 0.5162 in 2013B,
0.4238 in 2013J, 0.4961 in 2014B and 0.4165 in 2014J (`reports/cohort_by_week.json`). The test
presentation has fewer at-risk students than the training data as a whole, which matters for
calibration below.

## Quantitative analyses

### Whole cohort (2014J)
From `reports/baseline_metrics.json` (recall, precision, ROC-AUC) and
`reports/gbm_calibration.json` (ECE: average gap between predicted and observed risk over ten
equal-size groups of students).

| Week | Recall, top 20% | Precision, top 20% | ROC-AUC | ECE |
|---|---|---|---|---|
| 2 | 0.3092 | 0.6428 | 0.6912 | 0.0656 |
| 4 | 0.3550 | 0.7226 | 0.7171 | 0.0686 |
| 8 | 0.4135 | 0.7958 | 0.7862 | 0.0342 |
| 12 | 0.4496 | 0.8154 | 0.8071 | 0.0417 |

Between 0.3633 and 0.4165 of the cohort ended at risk, depending on the week, so a list of 20%
of students cannot reach most of them. At week 2 the model does not beat a plain ranking by
fewest clicks (recall 0.3139 for clicks, `reports/baseline_metrics.json`).

### Monitoring on 2014J
From `reports/monitoring.json` (method: README, "Monitoring"). Drift is measured against the
training cohort at the same week; expected values come from the development check on 2014B.

| Week | Score PSI | Drift status | Recall, top 20% (expected) | Mean risk minus observed rate | Outcome status |
|---|---|---|---|---|---|
| 2 | 0.0108 | alert | 0.3092 (0.3013) | 0.0621 | alert |
| 4 | 0.0931 | alert | 0.355 (0.3451) | 0.0335 | ok |
| 8 | 0.0462 | alert | 0.4135 (0.3896) | 0.0105 | ok |
| 12 | 0.0184 | alert | 0.4496 (0.4263) | 0.0391 | ok |

The drift alerts come from inputs set by the assessment calendar (assessments due and
submitted), which changed in two modules; the scores stayed stable and recall was above the
expected value at every week. The week-2 outcome alert is the same over-prediction as in the
first recommendation below.

### The evaluation gate and the gradient boosting challenger
Rule (agreed before the test run): the challenger's recall at the top 20% must be strictly
higher than both this model's and the clicks ranking's at weeks 4, 8 and 12
(`reports/gate.json`).

| Week | Gradient boosting | `lr_main` | Clicks ranking | Gate |
|---|---|---|---|---|
| 4 | 0.3619 | 0.3550 | 0.3302 | pass |
| 8 | 0.4155 | 0.4135 | 0.3568 | pass |
| 12 | 0.4454 | 0.4496 | 0.3754 | **fail** |

**The gate failed at week 12, so the challenger is not used.** The differences are small. A
paired bootstrap of the difference (gradient boosting minus `lr_main`,
`reports/model_gap_bootstrap.json`) gives:

| Week | Difference | 95% range | Share of resamples with gradient boosting ahead |
|---|---|---|---|
| 2 | 0.0072 | -0.0003 to 0.0154 | 0.972 |
| 4 | 0.0069 | 0.0006 to 0.0150 | 0.986 |
| 8 | 0.0020 | -0.0030 to 0.0097 | 0.825 |
| 12 | -0.0042 | -0.0097 to 0.0042 | 0.231 |

At weeks 8 and 12 the range includes 0: on this data the two models cannot be told apart
there. The gate was not changed after seeing this. A model that is not clearly better does not
justify losing the plain coefficients of the logistic regression.

### By student group
Full tables for every week, group and model: `reports/fairness_report.md`
(`reports/fairness_groups.csv` and `reports/fairness_intersectional.csv` at full precision).
Numbers below are for `lr_main` at week 12 unless stated, from `reports/fairness_groups.csv`.
Each group's at-risk rate is given because the list ranks by risk: groups with more at-risk
students are expected to be listed more often.

**Gender.** At-risk women are reached less often than at-risk men at every week. At week
12, recall is 0.4132 (range 0.3920 to 0.4372) for women (3,640 students) and 0.4742 (0.4560 to
0.4890) for men (4,931), a ratio of 0.8715. The difference is real (its range, -0.0905 to
-0.0261, excludes 0) but the ratio is inside the 0.8 rule, so it is not flagged for recall. The
model also ranks women less well (ROC-AUC 0.7591, range 0.7431 to 0.7742, against 0.8441, range
0.8336 to 0.8550) and predicts too high for women (calibration gap 0.0748, range 0.0602 to
0.0890, flagged) while being close for men (0.0127, range 0.0021 to 0.0230). The gradient
boosting model shows the same pattern (women 0.4037, range 0.3803 to 0.4286; men 0.4736, range
0.4574 to 0.4920).

**Deprivation (IMD band).** Recall is lowest in the least deprived band and where the band is
unknown, which are also the groups with the fewest at-risk students:

| IMD band | Students | At-risk rate | Recall [95% range] | Ratio to 30-40% band | Flag |
|---|---|---|---|---|---|
| 0-10% (most deprived) | 744 | 0.4409 | 0.4695 [0.4185, 0.5242] | 0.968 | |
| 90-100% (least deprived) | 698 | 0.2851 | 0.3618 [0.2903, 0.4255] | 0.746 | recall gap; calibration |
| Unknown | 352 | 0.2585 | 0.3187 [0.2308, 0.4312] | 0.657 | recall gap |

A list built from risk concentrates on the groups where risk is common. An at-risk student in
a group where risk is rare looks less like the typical at-risk student and is more often
missed. The reason the IMD band is missing for some students is not recorded in the data.

**Disability.** At-risk disabled students (769 students) are reached more often than others
(7,802): recall 0.5233 (0.4661 to 0.5662) against 0.4398 (0.4265 to 0.4525), ratio 1.1898, not
flagged. Their calibration gap is close to 0 (0.0016, range -0.0284 to 0.0325). In the
combination, disabled students in the most deprived bands (IMD 0-30%, 291 students) are flagged
for recall above the reference (0.6038, range 0.5172 to 0.6689, ratio 1.3874;
`reports/fairness_intersectional.csv`). That group is reached more, not less.

**Age, region, highest education.** No age group is flagged for recall. Students aged 55 and
over (80 students, 24 at risk) are too small to judge. Students with an HE qualification
(1,466 students) are reached less: recall 0.3958 (0.3521 to 0.4385), ratio 0.8643 at week 12.
They are flagged for recall at week 8, not at week 12. Two small education groups (No Formal
quals, Post Graduate) are too small to judge. No region is flagged for recall at week 12.

**Calibration across the cohort.** Many calibration flags at week 2 come from one shared
offset: the model predicts too high for the whole 2014J cohort at week 2 (calibration gap
0.0621, range 0.0522 to 0.0724, the "All students" row). The test presentation has fewer
at-risk students than the training data (see Training data). Compare a group's gap with the
"All students" row before reading it as a group problem.

Some groups go well beyond that offset at week 2. Over-prediction is largest for the less
deprived and better qualified: for example HE qualification 0.1290 (range 0.1061 to 0.1511,
1,575 students), IMD 90-100% 0.1272 (0.0952 to 0.1567, 758 students) and unknown IMD 0.1652
(0.1216 to 0.2104, 369 students). One group goes the other way: disabled students in the most
deprived bands (IMD 0-30%, 340 students at week 2) are predicted too low, by 0.1121 (range
0.0597 to 0.1606; `reports/fairness_intersectional.csv`). A risk score for them understates
their real risk at week 2. By week 12 their gap is within the noise (-0.0202, range -0.0664 to
0.0276).

**Flags by week.** Groups flagged out of 44 audited per model, excluding "All students"
(`reports/fairness_summary.json`, "flag_counts"):

| Week | `lr_main`: any / recall / calibration | Gradient boosting: any / recall / calibration | `lr_with_demographics`: any / recall / calibration |
|---|---|---|---|
| 2 | 29 / 0 / 29 | 27 / 1 / 26 | 38 / 24 / 27 |
| 4 | 14 / 3 / 14 | 19 / 2 / 19 | 22 / 17 / 7 |
| 8 | 7 / 4 / 3 | 14 / 5 / 11 | 13 / 13 / 0 |
| 12 | 14 / 3 / 12 | 9 / 5 / 4 | 22 / 12 / 10 |

A group can be flagged for both reasons, so the last two numbers can add up to more than the
first. `lr_main` has the fewest recall-gap flags at weeks 2, 8 and 12; at week 4 the gradient
boosting model has one fewer (2 against 3). The gradient boosting model has fewer
calibration flags at week 12 (4 against 12), in line with its lower overall calibration gap
that week (0.0246 against 0.0391, "All students" rows). Fairness was not part of the gate, so
this does not change which model is used.

**Would adding demographics as inputs help?** The comparison model `lr_with_demographics`
uses the six attributes as inputs. Its overall recall is about the same (0.4518 against 0.4496
at week 12, "All students" row), but it widens the deprivation and disability gaps: it
reaches at-risk students in the least deprived band and with unknown IMD band even less
(0.3015, range 0.2410 to 0.3704; and 0.2308, range 0.1485 to 0.3293) and at-risk disabled
students even more (0.5699, range 0.5208 to 0.6180, flagged). It narrows the gender gap a
little at week 12 (women 0.4244, range 0.4002 to 0.4455; ratio 0.9021 against 0.8715 for
`lr_main`). It has more recall-gap flags at every week (table above). On balance, leaving
demographics out of `lr_main` is the better choice on this evidence.

## Ethical considerations
- **Harm from a miss:** an at-risk student not on the list is not offered help early. The
  audit's headline is therefore recall by group.
- **Harm from a listing:** being contacted because a model listed you can feel like
  surveillance or a label. Advisors should offer help, not record a judgement, and the list
  should not be kept on the student's record.
- **Data is an incomplete view:** the model sees clicks and submissions only. It knows nothing
  of work, caring duties, health or study done offline.
- **Attributes not used as inputs are still in the data,** so the audit can check outcomes by
  group. Leaving them out of the model does not make it fair by itself; the audit above is how
  that is checked.
- **No changes were made because of the audit.** No reweighting, no separate cut-offs by group.
  Separate cut-offs would mean using gender or disability to decide who is contacted, which
  needs a policy decision, not a modelling one.

## Caveats and recommendations
- **One test presentation.** All results are from 2014J. The ranges cover sampling of students,
  not change between presentations or years.
- **Marking lag.** Scores are used from the day work was submitted. Staff may see marks later,
  so early-week results may be a little optimistic.
- **Many comparisons.** Treat single flags as prompts. Patterns that repeat across weeks and
  models (women's recall, the least deprived and unknown IMD groups) deserve the most
  attention.
- **Recommendations before real use:**
  1. Recalibrate on the most recent presentation, or monitor calibration by week and group,
     given the shared week-2 offset and the under-prediction for disabled students in the most
     deprived bands that week. Until then, advisors should not read a low week-2 score for
     those students as reassurance.
  2. Review the gender gap with advisors: whether women's lower recall comes from how women
     engage online in this data, and whether advisors should look beyond the list for them.
  3. Agree a fairness check for the retraining pipeline with the people who own the decision
     (advisors, equity and disability services), with thresholds they set. Until then the
     audit reports but does not block a model.
  4. Review `drift.json` with each weekly list and the monitoring report after each
     presentation (`uv run python scripts/monitor.py --mode test`); an outcome alert starts a
     retraining review, and nothing is retrained or promoted automatically.
  5. Rerun this audit at each retraining (`uv run python scripts/audit_fairness.py --mode
     test`). The retraining workflow (`.github/workflows/retrain.yml`) does this before it
     releases a model.
- **Reasons are the model's, not causes.** A reason says which inputs raised this student's
  score in this model. It does not say why the student is struggling.
- **Out of scope here:** the student population of the University of Auckland differs from
  the Open University's. Nothing in this card has been checked on University of Auckland data.
