"""Convert data/raw CSV files to data/processed Parquet. Usage: uv run python scripts/ingest.py"""

from pathlib import Path

from early_warning.ingest import csv_to_parquet

if __name__ == "__main__":
    counts = csv_to_parquet(Path("data/raw"), Path("data/processed"))
    for table, rows in counts.items():
        print(f"{table}: {rows:,} rows")
