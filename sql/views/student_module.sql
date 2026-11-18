-- One row per student per module presentation: who they are, when they registered or left,
-- and the outcome we want to predict. at_risk is the target (Fail or Withdrawn).
-- date_unregistration is almost a copy of the target (see docs/data.md). It must never be a
-- model input; it may only be used to drop students who had already left by the cutoff day.
CREATE OR REPLACE VIEW student_module AS
SELECT
    i.code_module,
    i.code_presentation,
    i.id_student,
    i.gender,
    i.region,
    i.highest_education,
    i.imd_band,
    i.age_band,
    i.num_of_prev_attempts,
    i.studied_credits,
    i.disability,
    r.date_registration,
    r.date_unregistration,
    c.module_presentation_length,
    i.final_result,
    i.final_result IN ('Fail', 'Withdrawn') AS at_risk
FROM student_info AS i
JOIN student_registration AS r USING (code_module, code_presentation, id_student)
JOIN courses AS c USING (code_module, code_presentation)
