from pathlib import Path

import duckdb
import pytest

from early_warning.ingest import csv_to_parquet, register_tables

SAMPLE_DIR = Path(__file__).parent / "fixtures" / "oulad_sample"


@pytest.fixture(scope="session")
def parquet_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("processed")
    csv_to_parquet(SAMPLE_DIR, out)
    return out


@pytest.fixture
def con(parquet_dir):
    connection = duckdb.connect()
    register_tables(connection, parquet_dir)
    yield connection
    connection.close()
