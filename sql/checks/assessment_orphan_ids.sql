-- Every submission refers to a known assessment.
SELECT s.*
FROM student_assessment AS s
ANTI JOIN assessments AS a
  USING (id_assessment)
