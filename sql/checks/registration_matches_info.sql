-- Every student in student_info has a registration row, and the other way round.
SELECT i.id_student AS info_student, r.id_student AS registration_student
FROM student_info AS i
FULL OUTER JOIN student_registration AS r
  ON  i.code_module = r.code_module
  AND i.code_presentation = r.code_presentation
  AND i.id_student = r.id_student
WHERE i.id_student IS NULL OR r.id_student IS NULL
