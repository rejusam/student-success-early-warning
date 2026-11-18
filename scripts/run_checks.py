"""Run data checks on data/processed and save the results to reports/.

Writes reports/data_checks.json (check -> failing rows) and reports/data_profile.json
(row counts, final_result counts, missing values per column). Exits with code 1 if any
check fails. Usage: uv run python scripts/run_checks.py
"""

import json
import sys
from pathlib import Path

import duckdb

from early_warning.checks import run_checks
from early_warning.ingest import TABLES, register_tables

PROCESSED = Path("data/processed")
REPORTS = Path("reports")


def profile(con: duckdb.DuckDBPyConnection) -> dict:
    result = {"row_counts": {}, "missing_values": {}}
    for table in TABLES.values():
        result["row_counts"][table] = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        columns = [row[0] for row in con.execute(f"DESCRIBE {table}").fetchall()]
        counts = con.execute(
            "SELECT " + ", ".join(f"count(*) - count({c})" for c in columns) + f" FROM {table}"
        ).fetchone()
        result["missing_values"][table] = {
            c: n for c, n in zip(columns, counts, strict=True) if n > 0
        }
    result["final_result"] = dict(
        con.execute(
            "SELECT final_result, count(*) FROM student_info GROUP BY 1 ORDER BY 1"
        ).fetchall()
    )
    first_day, last_day = con.execute("SELECT min(date), max(date) FROM student_vle").fetchone()
    result["student_vle_day_range"] = {"first": first_day, "last": last_day}
    result["student_vle_repeated_student_site_day"] = con.execute(
        "SELECT count(*) FROM (SELECT 1 FROM student_vle GROUP BY code_module, "
        "code_presentation, id_student, id_site, date HAVING count(*) > 1)"
    ).fetchone()[0]
    rows = con.execute(
        "SELECT i.final_result, r.date_unregistration IS NOT NULL, count(*) "
        "FROM student_info i JOIN student_registration r "
        "USING (code_module, code_presentation, id_student) GROUP BY 1, 2 ORDER BY 1, 2"
    ).fetchall()
    result["has_unregistration_date_by_final_result"] = {
        f"{final_result} | {'has date' if has_date else 'no date'}": n
        for final_result, has_date, n in rows
    }
    return result


def main() -> int:
    con = duckdb.connect()
    register_tables(con, PROCESSED)
    checks = run_checks(con)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "data_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    (REPORTS / "data_profile.json").write_text(json.dumps(profile(con), indent=2) + "\n")
    failed = {name: n for name, n in checks.items() if n > 0}
    print("All checks passed" if not failed else f"Failed checks: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
