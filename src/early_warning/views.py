"""Create the SQL views in sql/views/ on a connection that already has the base tables."""

from pathlib import Path

import duckdb

VIEWS_DIR = Path(__file__).resolve().parents[2] / "sql" / "views"


def create_views(con: duckdb.DuckDBPyConnection) -> None:
    for path in sorted(VIEWS_DIR.glob("*.sql")):
        con.execute(path.read_text())
