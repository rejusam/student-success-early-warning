"""Convert the OULAD CSV files to Parquet with fixed column types.

OULAD writes missing values as '?'. They are read as NULL. Column names and types are written
out here rather than guessed. The header of each CSV must match the expected column order,
because DuckDB assigns the given types by position, not by name.
"""

from pathlib import Path

import duckdb

# CSV file stem -> table name used everywhere else in the project
TABLES = {
    "courses": "courses",
    "assessments": "assessments",
    "vle": "vle",
    "studentInfo": "student_info",
    "studentRegistration": "student_registration",
    "studentAssessment": "student_assessment",
    "studentVle": "student_vle",
}

KEY = {"code_module": "VARCHAR", "code_presentation": "VARCHAR"}

SCHEMAS = {
    "courses": {**KEY, "module_presentation_length": "INTEGER"},
    "assessments": {
        **KEY,
        "id_assessment": "INTEGER",
        "assessment_type": "VARCHAR",
        "date": "INTEGER",
        "weight": "DOUBLE",
    },
    "vle": {
        "id_site": "INTEGER",
        **KEY,
        "activity_type": "VARCHAR",
        "week_from": "INTEGER",
        "week_to": "INTEGER",
    },
    "studentInfo": {
        **KEY,
        "id_student": "INTEGER",
        "gender": "VARCHAR",
        "region": "VARCHAR",
        "highest_education": "VARCHAR",
        "imd_band": "VARCHAR",
        "age_band": "VARCHAR",
        "num_of_prev_attempts": "INTEGER",
        "studied_credits": "INTEGER",
        "disability": "VARCHAR",
        "final_result": "VARCHAR",
    },
    "studentRegistration": {
        **KEY,
        "id_student": "INTEGER",
        "date_registration": "INTEGER",
        "date_unregistration": "INTEGER",
    },
    "studentAssessment": {
        "id_assessment": "INTEGER",
        "id_student": "INTEGER",
        "date_submitted": "INTEGER",
        "is_banked": "INTEGER",
        "score": "DOUBLE",
    },
    "studentVle": {
        **KEY,
        "id_student": "INTEGER",
        "id_site": "INTEGER",
        "date": "INTEGER",
        "sum_click": "INTEGER",
    },
}


def csv_to_parquet(raw_dir: Path, out_dir: Path) -> dict[str, int]:
    """Write one Parquet file per table into out_dir. Returns table -> row count."""
    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    counts = {}
    for stem, table in TABLES.items():
        header = con.execute(
            f"SELECT * FROM read_csv('{raw_dir / stem}.csv', header = true, all_varchar = true) "
            "LIMIT 0"
        ).description
        found = [column[0] for column in header]
        if found != list(SCHEMAS[stem]):
            raise ValueError(f"{stem}.csv columns {found} do not match {list(SCHEMAS[stem])}")
        source = f"read_csv('{raw_dir / stem}.csv', header = true, nullstr = '?', columns = $cols)"
        target = out_dir / f"{table}.parquet"
        con.execute(
            f"COPY (SELECT * FROM {source}) TO '{target}' (FORMAT parquet)", {"cols": SCHEMAS[stem]}
        )
        counts[table] = con.execute(f"SELECT count(*) FROM '{target}'").fetchone()[0]
    con.close()
    return counts


def register_tables(con: duckdb.DuckDBPyConnection, parquet_dir: Path) -> None:
    """Create one view per table over its Parquet file."""
    for table in TABLES.values():
        con.execute(
            f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM '{parquet_dir / table}.parquet'"
        )
