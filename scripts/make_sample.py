"""Cut a small OULAD sample for tests from data/raw.

Keeps two module presentations (AAA 2013J, BBB 2013J) and the 40 lowest student ids in each,
with all of their rows in every table. Values are kept as text so '?' stays '?'.
Usage: uv run python scripts/make_sample.py
"""

from pathlib import Path

import duckdb

RAW = Path("data/raw")
OUT = Path("tests/fixtures/oulad_sample")
PRESENTATIONS = "('AAA', '2013J'), ('BBB', '2013J')"
STUDENTS_PER_PRESENTATION = 40


def main() -> None:
    con = duckdb.connect()
    for stem in [
        "courses",
        "assessments",
        "vle",
        "studentInfo",
        "studentRegistration",
        "studentVle",
        "studentAssessment",
    ]:
        con.execute(
            f"CREATE VIEW {stem} AS "
            f"SELECT * FROM read_csv('{RAW / stem}.csv', all_varchar = true, header = true)"
        )

    con.execute(f"""
        CREATE TABLE keep_students AS
        SELECT code_module, code_presentation, id_student
        FROM (
            SELECT *, row_number() OVER (
                PARTITION BY code_module, code_presentation
                ORDER BY CAST(id_student AS INTEGER)
            ) AS rn
            FROM studentInfo
            WHERE (code_module, code_presentation) IN ({PRESENTATIONS})
        )
        WHERE rn <= {STUDENTS_PER_PRESENTATION}
    """)

    by_presentation = f"WHERE (code_module, code_presentation) IN ({PRESENTATIONS})"
    by_student = "SEMI JOIN keep_students USING (code_module, code_presentation, id_student)"
    queries = {
        "courses": f"SELECT * FROM courses {by_presentation}",
        "assessments": f"SELECT * FROM assessments {by_presentation}",
        "vle": f"SELECT * FROM vle {by_presentation}",
        "studentInfo": f"SELECT * FROM studentInfo {by_student}",
        "studentRegistration": f"SELECT * FROM studentRegistration {by_student}",
        "studentVle": f"SELECT * FROM studentVle {by_student}",
        "studentAssessment": """
            SELECT sa.* FROM studentAssessment sa
            JOIN assessments a USING (id_assessment)
            SEMI JOIN keep_students k
              ON k.id_student = sa.id_student
             AND k.code_module = a.code_module
             AND k.code_presentation = a.code_presentation
        """,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    for stem, query in queries.items():
        path = OUT / f"{stem}.csv"
        con.execute(f"COPY ({query}) TO '{path}' (HEADER, DELIMITER ',')")
        rows = con.execute(f"SELECT count(*) FROM ({query})").fetchone()[0]
        print(f"{stem}: {rows} rows, {path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
