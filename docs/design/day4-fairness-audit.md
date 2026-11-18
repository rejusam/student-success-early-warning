# Day 4 design: fairness audit by student group, and a model card

Date: 2026-11-18. Status: decisions below are open to review.

## Why
An early-warning list decides which students an advisor contacts first. If the list reaches
at-risk students in one group less often than in another, the students it misses lose the
offer of help. The University of Auckland's Learning Analytics Principles and Policy asks for
analytics that avoid bias, treat data as an incomplete view of a student, and keep a person in
any decision about an individual
(https://auckland.ac.nz/en/about-us/about-the-university/policy-hub/enabling-environment/digital/data-governance-analytics/learning-analytics-principles-policy.html).
Day 4 checks the served model against that, and writes down what the model is, what it is for,
how well it works and for whom, in a model card a non-specialist can read.

## Goals
- Measure, for each student group, how well the served model (`lr_main`) works on the 2014J
  test presentation, at every prediction week (2, 4, 8, 12).
- Do the same for the gradient boosting challenger (`gbm`) and for the comparison model with
  demographics (`lr_with_demographics`), so the three can be compared group by group.
- Give every number an uncertainty range, and mark the groups too small to judge.
- Put a range on the week-12 gap that failed the Day 3 gate.
- Write `docs/model_card.md` in plain English, including the failed gate.

## Non-goals
- Changing any model because of what the audit finds. No reweighting, no group-specific
  thresholds, no retraining. Findings go in the model card as they are.
- A fairness gate that blocks a model. The audit reports and flags groups for review. Turning
  it into a gate needs thresholds agreed with the people who own the decision (advisors,
  equity and disability services), not chosen by the model builder. The model card says so.
- New features or new demographic fields.

## Anti-goals
- Any change to a model, its settings or its inputs after seeing the audit.
- Quoting a group number without its size and range, or judging a group too small to judge.
- Treating different flag rates between groups as unfair by themselves. Groups have different
  shares of at-risk students, so a list that reaches them equally will flag them at different
  rates.

## Decisions
1. **Models audited, and where their scores come from.**
   - `lr_main` and `lr_with_demographics`: refitted on 2013B, 2013J and 2014B as on Day 2
     (deterministic). The script stops if their 2014J metrics differ from
     `reports/baseline_metrics.json`.
   - `gbm`: the registered models `early-warning-week-<k>`, alias `challenger`, loaded from
     MLflow and not refitted, so the audit checks the exact model that was registered. The
     script stops if their 2014J metrics differ from `reports/gbm_metrics.json`. (Checked on
     2026-11-18: all four weeks match exactly.)
   - Scoring 2014J with models that are already fixed is evaluation, not tuning.
2. **Groups.** The six fields in `DEMOGRAPHIC_FEATURES`, one at a time: gender, age band,
   region, highest education, IMD band and disability. A missing IMD band is its own group,
   "Unknown". One intersection, chosen before seeing any results: disability × IMD band,
   with IMD put into three bands (0-30%, 30-70%, 70-100%, plus Unknown) so the cells are big
   enough. Disability and deprivation are the two fields most tied to equity policy.
   Every table also has an "All students" row (added after the first run, see below).
3. **The lists are not rebuilt per group.** Each module presentation's top-20% list is built
   once over all its students, as an advisor would get it. Each student's flag is then
   counted in their group. This measures the list as used.
4. **Measures per group.**
   - Recall at the top 20% (headline): the share of the group's at-risk students that the
     list reaches. Equal recall across groups is "equal opportunity" (Hardt, Price and
     Srebro 2016, https://arxiv.org/abs/1610.02413). It is the headline because missing an
     at-risk student is the harm that matters most here.
   - Precision at the top 20%: the share of the group's flagged students who were at risk.
   - Calibration in the large: mean predicted risk against the observed at-risk rate, and
     their difference. A model that predicts too low for a group ranks that group's students
     too low. ECE by bins is not used per group, because ten bins of a group of 80 students
     hold eight students each.
   - ROC-AUC within the group: how well the model orders the group's students, with no
     threshold.
   - Size, number at risk, at-risk rate and flag rate, as context. Flag rate is not judged
     (anti-goals).
5. **Reference group:** the largest group of each attribute. Gaps are given as a difference
   (group minus reference) and a ratio (group divided by reference).
6. **Uncertainty: stratified bootstrap.** 1,000 resamples. In each, students are drawn with
   replacement within each module presentation, the top-20% lists are rebuilt on the resample,
   and every group measure and gap is recomputed. 95% intervals are the 2.5th and 97.5th
   percentiles. Resampling within module presentations keeps each list's size; rebuilding the
   lists carries the uncertainty of where the 20% cut falls. The same resamples are used for
   all three models, so differences between models are paired. The seed is fixed.
7. **Groups too small to judge.** A group with fewer than 100 students or fewer than 30 at-risk
   students is reported with its numbers and range, labelled "small", and never flagged.
   Intersection cells under the same floor are listed with their counts only.
8. **Review flags.** A group is flagged for review when either:
   - its recall ratio to the reference is below 0.8 or above 1.25 (the four-fifths rule, as
     used by the Aequitas toolkit, http://aequitas.dssg.io/), and the 95% interval of the
     recall difference does not include 0; or
   - its calibration gap is larger than 0.05 either way, and its 95% interval does not
     include 0.
   A flag is a prompt for a person to look, not a verdict. Six attributes, 44 groups per model
   (36 single-attribute groups and 8 intersection cells),
   four weeks and three models make many comparisons, so some flags are expected by chance.
   The rules are fixed here, before the audit runs.
9. **The week-12 gate gap.** With the same bootstrap, a 95% interval for the difference in
   recall@top-20% between `gbm` and `lr_main` at each week (paired). Reported in
   `reports/model_gap_bootstrap.json` and in the model card. The gate result does not change.
10. **Model card.** `docs/model_card.md`, in the sections of Mitchell et al. 2019, "Model Cards
    for Model Reporting" (https://arxiv.org/abs/1810.03993): model details, intended use,
    factors, metrics, evaluation data, training data, quantitative analyses, ethical
    considerations, caveats and recommendations. It covers the served `lr_main` models, says
    the gradient boosting challenger failed the gate at week 12 and why it is not served, and
    states what the audit found, flags included. Every number in it comes from a file in
    `reports/`, named next to the number.
11. **Generated tables.** The script also writes `reports/fairness_report.md`, the group tables
    in Markdown, so the model card quotes numbers from a file the script wrote rather than
    copied by hand.

## Change after the first run
The first run flagged calibration in most groups at week 2. That pointed to the whole 2014J
cohort being predicted too high, not to single groups (Day 3 already reported `lr_main` ECE of
0.0656 at week 2, `reports/gbm_calibration.json`). To let a reader tell a shared offset from a
group-specific one, an "All students" row was added to every table. It is a reference line
only: the flag rules, thresholds, groups and models did not change, and the bootstrap seed is
the same.

The second run showed that the ranges were not reproducible: with the same seed they moved a
little between runs. The feature query (`sql/features/student_week.sql`) returns rows in no
fixed order, and the bootstrap draws rows by position, so each run drew different students.
The bootstrap now sorts rows by module presentation and student before drawing, and a test
checks that shuffled input gives identical results. Point values were not affected (they do
not depend on row order). A fourth run added flag counts to `fairness_summary.json`; every
other output was byte-identical to the third run's.

The final review then found that a Day 3 smoke test failed now and then, for the same reason:
calibration folds on the small test sample are cut by row position. The feature query now ends
with `ORDER BY code_presentation, code_module, id_student`, and a test checks the order. The
fifth and last run used that query. It also stopped `fairness_report.md` from printing measures
for intersection cells under the size floor (the CSV already hid them), and prints the model
gap with four decimals. Its consistency checks passed, so `lr_main` and `lr_with_demographics`
still reproduce `reports/baseline_metrics.json` exactly with the fixed order.

## Approaches considered
1. **Own module in pandas, following the pattern of an earlier equity audit (chosen).**
   `src/early_warning/fairness.py`: a stratified table per attribute, gaps to a reference
   group, a size floor, an intersectional table and a summary (range, worst group). It works
   on the per-module top-k lists the project already uses, and the bootstrap rebuilds those
   lists. Every line can be explained.
2. **Fairlearn `MetricFrame`.** The standard open-source tool for group metrics. It computes
   per-group values and differences well, but the flags here come from per-module lists that
   must be rebuilt in each bootstrap resample, which sits outside it. It would add a
   dependency for a small part of the work.
3. **Aequitas.** Built for exactly this kind of audit and the source of the 0.8 rule, but it
   expects one fixed binary decision per row and has no bootstrap. Its rule is used; the
   package is not.

## Outputs
| File | Contents |
|---|---|
| `reports/fairness_groups.csv` | one row per week, model, attribute and group: sizes, measures, gaps, 95% ranges, small and flag labels |
| `reports/fairness_intersectional.csv` | the same for disability × IMD band (cells under the floor: counts only) |
| `reports/fairness_summary.json` | per week, model and attribute: reference group, recall range, worst group, flagged groups; settings used |
| `reports/model_gap_bootstrap.json` | per week: `gbm` minus `lr_main` recall@top-20%, point value and 95% range |
| `reports/fairness_report.md` | the tables above in Markdown, for the model card |

## Script
`scripts/audit_fairness.py`, logic in `src/early_warning/fairness.py` (measures, bootstrap,
flags) and `src/early_warning/audit.py` (load, score, check, write).
- `--mode test`: train 2013B, 2013J, 2014B; audit 2014J; MLflow store `mlruns/`.
- `--mode smoke`: the test sample, 20 resamples, a lower size floor, a given output directory
  and a given MLflow store (made by `train_gbm.py --mode smoke`). Refuses `reports/` and
  `mlruns/`.

## Tests
- Group measures on a small hand-made frame match values worked out by hand.
- Flags come from the per-module list built once, not per group.
- Small groups are labelled and never flagged; the flag rules fire on made-up gaps.
- Bootstrap: resamples keep each module presentation's size; fixed seed gives the same
  result; the range contains the point value on a well-behaved example.
- No model input changes: the audit never passes demographic columns to `lr_main` or `gbm`.
- Consistency check: a changed metric raises an error.
- Smoke: `main([... "--mode", "smoke"])` writes only to the given directory.

## Constraints
- Same as Days 2 and 3: Python 3.12, uv lock, code simple enough to explain line by line,
  every number quoted in docs from `reports/`. No new dependencies.

## Left to the builder
- Column names and JSON layout, the exact Markdown tables, function names.
- The wording of the model card, within the sections above.
