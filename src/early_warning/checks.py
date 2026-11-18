"""Run the data checks in sql/checks/.

Each .sql file is a SELECT that returns the rows breaking one rule, so a check passes
when it returns zero rows.
"""

from pathlib import Path

import duckdb

CHECKS_DIR = Path(__file__).resolve().parents[2] / "sql" / "checks"
CHECK_NAMES = tuple(sorted(p.stem for p in CHECKS_DIR.glob("*.sql")))


def run_checks(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Return check name -> number of rows that break it (0 means it passed)."""
    if not CHECK_NAMES:
        raise FileNotFoundError(f"no check files found in {CHECKS_DIR}")
    results = {}
    for name in CHECK_NAMES:
        query = (CHECKS_DIR / f"{name}.sql").read_text()
        results[name] = con.execute(f"SELECT count(*) FROM ({query})").fetchone()[0]
    return results
