-- One row per student in the cohort at a prediction week, with features known by the cutoff.
-- Parameters: $cutoff = last day of data that may be used (7 * week - 1),
--             $last_week = week - 1 (the last full week before the prediction is made).
-- Cohort: drop students who unregistered on or before the cutoff, and students who had not
-- registered yet. date_unregistration is used only for this filter, never as a feature.
-- Leakage rule: every click and submission used here is dated on or before the cutoff.
-- Banked submissions carry a result from an earlier presentation, so they count from the start.
WITH cohort AS (
    SELECT *
    FROM student_module
    WHERE NOT coalesce(date_unregistration <= $cutoff, false)
      AND NOT coalesce(date_registration > $cutoff, false)
),

engagement AS (
    SELECT
        code_module,
        code_presentation,
        id_student,
        sum(clicks) AS clicks_total,
        sum(clicks) FILTER (WHERE week = $last_week) AS clicks_last_week,
        sum(active_days) AS active_days_total,
        count(*) FILTER (WHERE week >= 0) AS weeks_active,
        sum(clicks) FILTER (WHERE week < 0) AS clicks_before_start
    FROM vle_weekly
    WHERE week <= $last_week
    GROUP BY ALL
),

-- Non-exam assessments due by the cutoff, per module presentation
due AS (
    SELECT code_module, code_presentation, id_assessment
    FROM assessments
    WHERE assessment_type <> 'Exam' AND date <= $cutoff
),

-- Non-exam submissions known by the cutoff
known_submissions AS (
    SELECT *
    FROM assessment_submissions
    WHERE assessment_type <> 'Exam'
      AND (is_banked = 1 OR date_submitted <= $cutoff)
),

due_counts AS (
    SELECT code_module, code_presentation, count(*) AS n_due
    FROM due
    GROUP BY ALL
),

submitted AS (
    SELECT
        s.code_module,
        s.code_presentation,
        s.id_student,
        count(*) AS n_submitted,
        count(*) FILTER (WHERE s.is_banked = 0 AND s.days_late > 0) AS n_late
    FROM known_submissions AS s
    JOIN due AS d USING (code_module, code_presentation, id_assessment)
    GROUP BY ALL
),

scores AS (
    SELECT code_module, code_presentation, id_student, avg(score) AS mean_score
    FROM known_submissions
    GROUP BY ALL
)

SELECT
    c.code_module,
    c.code_presentation,
    c.id_student,
    coalesce(e.clicks_total, 0) AS clicks_total,
    coalesce(e.clicks_last_week, 0) AS clicks_last_week,
    coalesce(e.active_days_total, 0) AS active_days_total,
    coalesce(e.weeks_active, 0) AS weeks_active,
    coalesce(e.clicks_before_start, 0) AS clicks_before_start,
    coalesce(dc.n_due, 0) AS n_due,
    coalesce(sub.n_submitted, 0) AS n_submitted,
    coalesce(dc.n_due, 0) - coalesce(sub.n_submitted, 0) AS n_missed,
    coalesce(sub.n_late, 0) AS n_late,
    sc.mean_score,
    c.num_of_prev_attempts,
    c.studied_credits,
    c.date_registration,
    c.gender,
    c.age_band,
    c.region,
    c.highest_education,
    c.imd_band,
    c.disability,
    c.at_risk
FROM cohort AS c
LEFT JOIN engagement AS e USING (code_module, code_presentation, id_student)
LEFT JOIN due_counts AS dc USING (code_module, code_presentation)
LEFT JOIN submitted AS sub USING (code_module, code_presentation, id_student)
LEFT JOIN scores AS sc USING (code_module, code_presentation, id_student)
