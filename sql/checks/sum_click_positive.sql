-- A click row records at least one click.
SELECT *
FROM student_vle
WHERE sum_click IS NULL OR sum_click < 1
