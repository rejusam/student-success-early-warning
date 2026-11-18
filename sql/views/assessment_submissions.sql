-- One row per submitted assessment, with its due date. due_date is NULL for some exams
-- in the source data. days_late > 0 means submitted after the due date.
-- Banked rows (is_banked = 1) carry a result from an earlier presentation, so their
-- days_late is not meaningful. score and date_submitted are only known once submitted:
-- features for week k must use rows with date_submitted <= the cutoff day.
CREATE OR REPLACE VIEW assessment_submissions AS
SELECT
    a.code_module,
    a.code_presentation,
    s.id_student,
    s.id_assessment,
    a.assessment_type,
    a.date AS due_date,
    a.weight,
    s.date_submitted,
    s.is_banked,
    s.score,
    s.date_submitted - a.date AS days_late
FROM student_assessment AS s
JOIN assessments AS a USING (id_assessment)
