-- The target is built from final_result, so it must be one of the four documented values.
SELECT *
FROM student_info
WHERE final_result IS NULL
   OR final_result NOT IN ('Distinction', 'Pass', 'Fail', 'Withdrawn')
