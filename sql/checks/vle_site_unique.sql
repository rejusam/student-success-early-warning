-- One row per VLE resource.
SELECT id_site, count(*) AS n
FROM vle
GROUP BY ALL
HAVING count(*) > 1
