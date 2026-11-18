-- Every click row belongs to a known student on that module presentation.
SELECT v.*
FROM student_vle AS v
ANTI JOIN student_info AS i
  USING (code_module, code_presentation, id_student)
