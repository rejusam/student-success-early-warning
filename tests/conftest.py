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


@pytest.fixture(scope="session")
def bundle_dir(parquet_dir, tmp_path_factory):
    """A serving bundle released from the test sample (smoke mode)."""
    from early_warning.release import main

    out = tmp_path_factory.mktemp("release")
    args = ["--mode", "smoke", "--processed", str(parquet_dir)]
    main([*args, "--mlflow-dir", str(out / "ml"), "--bundle", str(out / "bundle")])
    return out / "bundle"


@pytest.fixture
def frames(con):
    from early_warning.features import WEEKS, load_week
    from early_warning.views import create_views

    create_views(con)
    return {week: load_week(con, week) for week in WEEKS}
