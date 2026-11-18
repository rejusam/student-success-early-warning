# Data: Open University Learning Analytics Dataset (OULAD)

## Source and licence
- Kuzilek, J., Hlosta, M. & Zdrahal, Z. (2015). *Open University Learning Analytics Dataset.*
  UCI Machine Learning Repository. https://doi.org/10.24432/C5KK69
- Page: https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset
- Licence: Creative Commons Attribution 4.0 International (CC BY 4.0), as stated on the UCI page.
  The description file inside the zip (`OULAD.names`) does not state a licence.
- Download: `scripts/download_oulad.py`. The zip's SHA-256 is fixed in
  `src/early_warning/download.py`, so the pipeline stops if the file changes.

The data is public and identifies students only by a numeric `id_student`. No University of
Auckland data is used anywhere in this project.

## Tables
All counts are from `reports/data_profile.json` (produced by `scripts/run_checks.py`).

| Table (CSV) | Rows | One row per | Main columns |
|---|---|---|---|
| courses (`courses.csv`) | 22 | module presentation | `module_presentation_length` (days) |
| assessments (`assessments.csv`) | 206 | assessment | `assessment_type` (TMA, CMA, Exam), `date` (due day), `weight` |
| vle (`vle.csv`) | 6,364 | VLE resource | `activity_type`, `week_from`, `week_to` |
| student_info (`studentInfo.csv`) | 32,593 | student per module presentation | demographics, `final_result` |
| student_registration (`studentRegistration.csv`) | 32,593 | student per module presentation | `date_registration`, `date_unregistration` |
| student_assessment (`studentAssessment.csv`) | 173,912 | submission | `date_submitted`, `is_banked`, `score` |
| student_vle (`studentVle.csv`) | 10,655,280 | click record (see quirks) | `id_site`, `date`, `sum_click` |

A module presentation is one run of a module, such as `AAA` `2013J`. "B" presentations start in
February and "J" presentations in October (`OULAD.names`, section 4). All `date` columns count days
from the start of the presentation, which is day 0.

## Target
`final_result` counts: Distinction 3,024; Fail 7,052; Pass 12,361; Withdrawn 10,156.
The early-warning target is `at_risk = final_result IN ('Fail', 'Withdrawn')`
(`sql/views/student_module.sql`).

## Fields for the fairness audit
`student_info` has `gender`, `region`, `highest_education`, `imd_band` (Index of Multiple
Deprivation band), `age_band` and `disability`. They are used to check model performance by
group. They are not inputs to the main model. A comparison model that adds them is reported
next to it (`docs/design/day2-weekly-baseline.md`, decision 3).

## Data quirks found while checking
- Missing values are written as `?` in the CSVs. Ingest reads them as NULL. Missing counts
  (from `reports/data_profile.json`): `imd_band` 1,111; `score` 173; assessment `date` 11;
  `date_registration` 45; `week_from` / `week_to` 5,243. `date_unregistration` is NULL for
  22,521 students. NULL does not always mean the student stayed: 93 Withdrawn students have
  no unregistration date (`has_unregistration_date_by_final_result`).
- `imd_band` has the value `10-20` without a `%`, while the other bands have one (`0-10%`,
  `20-30%`, ...). It is kept as it is in the source.
- `OULAD.names` gives "students in courses: 32953". The file has 32,593 rows, and so does
  `studentRegistration.csv`. The description file looks to have swapped two digits.
- VLE activity starts before the module does. `student_vle.date` runs from day -25 to day 269.
  Weeks use `floor(date / 7)`, so day -1 is week -1, not week 0.
- `student_vle` is not one row per student, resource and day. 1,614,505 (student, resource,
  day) combinations appear more than once (`student_vle_repeated_student_site_day`).
  `vle_weekly` sums `sum_click`, so repeats are counted correctly there.

## Leakage risks for the weekly model
- `date_unregistration` almost gives away the target. 10,063 Withdrawn students have a date,
  against 9 students with any other result, all of them Fail
  (`has_unregistration_date_by_final_result`). It must never be a model input. It may only be
  used to drop students who had already left before the prediction week.
- `score`, `date_submitted` and `days_late` exist only after a submission. `vle_weekly` holds
  every week of the presentation. Features for week *k* must filter both to days on or before
  the cutoff.
- Banked assessments (`is_banked = 1`) carry a result from an earlier presentation, so their
  `days_late` has no meaning.

## Module registrations per presentation and prediction cohorts
From `reports/cohort_by_week.json`. Module registrations (one row per student per module
presentation; a student can take more than one module) per presentation: 2013B 4,684;
2013J 8,845; 2014B 7,804; 2014J 11,260.

At week k a student is in the cohort unless they unregistered on or before day 7k - 1, or had
not registered yet by then. Cohort size (and share at risk) per week:

| Week | 2013B | 2013J | 2014B | 2014J (test) |
|---|---|---|---|---|
| 2 | 4,324 (0.5162) | 7,756 (0.4238) | 6,656 (0.4961) | 9,333 (0.4165) |
| 4 | 4,127 (0.4929) | 7,684 (0.4179) | 6,510 (0.4843) | 9,214 (0.4079) |
| 8 | 3,965 (0.4719) | 7,465 (0.4007) | 6,214 (0.4598) | 8,879 (0.3854) |
| 12 | 3,822 (0.4519) | 7,313 (0.3882) | 6,023 (0.4425) | 8,571 (0.3633) |

"B" presentations have a higher share at risk than "J" presentations in both years.

Registrations left out of the 2014J cohort: 1,897 had unregistered and 30 had not registered
yet at week 2; 2,688 had unregistered and 1 had not registered yet at week 12. Counts for every
week and presentation are in the same file (`left_by_cutoff`, `not_yet_registered`).

## Data checks
`sql/checks/` holds one SQL file per rule. Each returns the rows that break the rule. Results
for the full data are in `reports/data_checks.json`, where every check is 0.

| Check | Rule |
|---|---|
| `student_info_key_unique` | one row per student per module presentation |
| `registration_key_unique` | one registration row per student per module presentation |
| `registration_matches_info` | the same students appear in both tables |
| `final_result_allowed` | `final_result` is one of the four documented values |
| `vle_orphan_students` | every click row belongs to a known student |
| `assessment_orphan_ids` | every submission refers to a known assessment |
| `sum_click_positive` | every click row has at least one click |
| `courses_key_unique` | one row per module presentation in `courses` |
| `assessment_id_unique` | one row per assessment |
| `vle_site_unique` | one row per VLE resource |
| `vle_orphan_sites` | every click row refers to a known VLE resource |
