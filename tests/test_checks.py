from early_warning.checks import CHECK_NAMES, run_checks


def make_editable(con, table):
    """Copy a Parquet-backed view into a table so a test can break it."""
    con.execute(f"CREATE TABLE {table}_edit AS SELECT * FROM {table}")
    con.execute(f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM {table}_edit")
    return f"{table}_edit"


def test_sample_passes_all_checks(con):
    results = run_checks(con)
    assert set(results) == set(CHECK_NAMES)
    assert all(count == 0 for count in results.values()), results


def test_duplicate_key_detected(con):
    edit = make_editable(con, "student_info")
    con.execute(f"INSERT INTO {edit} SELECT * FROM {edit} LIMIT 1")
    assert run_checks(con)["student_info_key_unique"] == 1


def test_bad_final_result_detected(con):
    edit = make_editable(con, "student_info")
    con.execute(
        f"UPDATE {edit} SET final_result = 'Passed' WHERE id_student = "
        f"(SELECT min(id_student) FROM {edit})"
    )
    assert run_checks(con)["final_result_allowed"] >= 1


def test_expected_checks_are_found():
    assert len(CHECK_NAMES) == 11


def test_duplicate_assessment_id_detected(con):
    edit = make_editable(con, "assessments")
    con.execute(f"INSERT INTO {edit} SELECT * FROM {edit} LIMIT 1")
    assert run_checks(con)["assessment_id_unique"] == 1


def test_duplicate_course_detected(con):
    edit = make_editable(con, "courses")
    con.execute(f"INSERT INTO {edit} SELECT * FROM {edit} LIMIT 1")
    assert run_checks(con)["courses_key_unique"] == 1
