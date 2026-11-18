-- One registration row per student per module presentation.
SELECT code_module, code_presentation, id_student, count(*) AS n
FROM student_registration
GROUP BY ALL
HAVING count(*) > 1
