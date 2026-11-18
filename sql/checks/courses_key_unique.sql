-- One row per module presentation. A duplicate would multiply rows in student_module.
SELECT code_module, code_presentation, count(*) AS n
FROM courses
GROUP BY ALL
HAVING count(*) > 1
