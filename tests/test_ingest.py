import shutil

import duckdb
import pytest

from early_warning.ingest import TABLES, csv_to_parquet
from tests.conftest import SAMPLE_DIR


def test_all_tables_written(parquet_dir):
    for table in TABLES.values():
        path = parquet_dir / f"{table}.parquet"
        assert path.exists(), table
        assert duckdb.execute(f"SELECT count(*) FROM '{path}'").fetchone()[0] > 0


def test_question_mark_becomes_null(tmp_path):
    raw = tmp_path / "raw"
    shutil.copytree(SAMPLE_DIR, raw)
    lines = (raw / "studentAssessment.csv").read_text().splitlines()
    header, first = lines[0], lines[1].split(",")
    first[-1] = "?"  # score is the last column
    (raw / "studentAssessment.csv").write_text("\n".join([header, ",".join(first)]) + "\n")

    csv_to_parquet(raw, tmp_path / "out")
    score, score_type = duckdb.execute(
        f"SELECT score, typeof(score) FROM '{tmp_path / 'out' / 'student_assessment.parquet'}'"
    ).fetchone()
    assert score is None
    assert score_type == "DOUBLE"


def test_rerun_overwrites(tmp_path):
    first = csv_to_parquet(SAMPLE_DIR, tmp_path)
    second = csv_to_parquet(SAMPLE_DIR, tmp_path)
    assert first == second
    assert first["student_info"] == 80


def test_reordered_header_rejected(tmp_path):
    raw = tmp_path / "raw"
    shutil.copytree(SAMPLE_DIR, raw)
    path = raw / "studentRegistration.csv"
    lines = path.read_text().splitlines()
    lines[0] = lines[0].replace(
        "date_registration,date_unregistration", "date_unregistration,date_registration"
    )
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="studentRegistration"):
        csv_to_parquet(raw, tmp_path / "out")
