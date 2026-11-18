-- Every click row refers to a known VLE resource.
SELECT v.*
FROM student_vle AS v
ANTI JOIN vle USING (id_site)
