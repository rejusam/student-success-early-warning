-- One row per student per module presentation. Returns keys that appear more than once.
SELECT code_module, code_presentation, id_student, count(*) AS n
FROM student_info
GROUP BY ALL
HAVING count(*) > 1
