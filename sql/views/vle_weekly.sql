-- Clicks per student per week. Day 0 is the module start, so days -7..-1 are week -1.
-- floor() is used (not integer division) so negative days round down, not toward zero.
CREATE OR REPLACE VIEW vle_weekly AS
SELECT
    code_module,
    code_presentation,
    id_student,
    CAST(floor(date / 7) AS INTEGER) AS week,
    sum(sum_click) AS clicks,
    count(DISTINCT date) AS active_days
FROM student_vle
GROUP BY ALL
