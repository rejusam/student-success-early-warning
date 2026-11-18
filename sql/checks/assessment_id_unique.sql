-- One row per assessment. A duplicate would multiply rows in assessment_submissions.
SELECT id_assessment, count(*) AS n
FROM assessments
GROUP BY ALL
HAVING count(*) > 1
