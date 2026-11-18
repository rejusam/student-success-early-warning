import pandas as pd
import pytest

from early_warning.features import (
    CATEGORICAL_FEATURES,
    DEMOGRAPHIC_FEATURES,
    NUMERIC_FEATURES,
    WEEKS,
    cutoff_day,
    load_week,
)
from early_warning.views import create_views

KEYS = ["code_module", "code_presentation", "id_student"]


@pytest.fixture
def fcon(con):
    create_views(con)
    return con


def replace_table(con, table, edits):
    """Swap a base table for an edited copy named edited_<table>. edits run on that copy."""
    con.execute(f"CREATE TABLE edited_{table} AS SELECT * FROM {table}")
    for sql in edits:
        con.execute(sql)
    con.execute(f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM edited_{table}")


def sort(df):
    return df.sort_values(KEYS).reset_index(drop=True)


def row_of(df, student):
    rows = df[df.id_student == student]
    assert len(rows) == 1
    return rows.iloc[0]


def a_kept_student(con):
    """A student who registered before day 0 and never unregistered."""
    return con.execute(
        "SELECT id_student FROM student_registration WHERE date_unregistration IS NULL "
        "AND date_registration <= 0 ORDER BY id_student LIMIT 1"
    ).fetchone()[0]


def test_cutoff_day():
    assert [cutoff_day(w) for w in WEEKS] == [13, 27, 55, 83]


def test_no_forbidden_columns(fcon):
    df = load_week(fcon, 4)
    assert "date_unregistration" not in df.columns
    assert "final_result" not in df.columns
    assert not set(DEMOGRAPHIC_FEATURES) & set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    for column in NUMERIC_FEATURES + CATEGORICAL_FEATURES + DEMOGRAPHIC_FEATURES + ["at_risk"]:
        assert column in df.columns


def test_one_row_per_student(fcon):
    df = load_week(fcon, 4)
    assert len(df) > 0
    assert not df.duplicated(KEYS).any()


def test_cohort_drops_students_gone_by_cutoff(fcon):
    gone, stays = fcon.execute(
        "SELECT id_student FROM student_registration WHERE date_unregistration IS NULL "
        "AND date_registration <= 0 ORDER BY id_student LIMIT 2"
    ).fetchall()
    replace_table(
        fcon,
        "student_registration",
        [
            "UPDATE edited_student_registration SET date_unregistration = 27 "
            f"WHERE id_student = {gone[0]}",
            "UPDATE edited_student_registration SET date_unregistration = 28 "
            f"WHERE id_student = {stays[0]}",
        ],
    )
    ids = set(load_week(fcon, 4).id_student)
    assert gone[0] not in ids
    assert stays[0] in ids


def test_cohort_drops_late_registrants(fcon):
    student = a_kept_student(fcon)
    replace_table(
        fcon,
        "student_registration",
        [
            "UPDATE edited_student_registration SET date_registration = 14 "
            f"WHERE id_student = {student}"
        ],
    )
    assert student not in set(load_week(fcon, 2).id_student)
    assert student in set(load_week(fcon, 4).id_student)


@pytest.mark.parametrize("week", WEEKS)
def test_truncation_does_not_change_features(fcon, week):
    before = load_week(fcon, week)
    c = cutoff_day(week)
    replace_table(fcon, "student_vle", [f"DELETE FROM edited_student_vle WHERE date > {c}"])
    # Banked rows are kept on purpose: they are known from registration (spec, Builder calls).
    replace_table(
        fcon,
        "student_assessment",
        [
            "DELETE FROM edited_student_assessment "
            f"WHERE NOT (is_banked = 1 OR date_submitted <= {c})"
        ],
    )
    after = load_week(fcon, week)
    pd.testing.assert_frame_equal(sort(before), sort(after))


def an_early_submission(con):
    """A non-banked submission of a non-exam assessment due by day 27, made by day 27."""
    return con.execute(
        "SELECT id_student, id_assessment FROM assessment_submissions "
        "WHERE assessment_type <> 'Exam' AND due_date <= 27 AND date_submitted <= 27 "
        "AND is_banked = 0 ORDER BY id_student LIMIT 1"
    ).fetchone()


def test_late_submission_counts_as_missed(fcon):
    student, assessment = an_early_submission(fcon)
    before = row_of(load_week(fcon, 4), student)
    replace_table(
        fcon,
        "student_assessment",
        [
            f"UPDATE edited_student_assessment SET date_submitted = 40 "
            f"WHERE id_student = {student} AND id_assessment = {assessment}"
        ],
    )
    after = row_of(load_week(fcon, 4), student)
    assert after.n_submitted == before.n_submitted - 1
    assert after.n_missed == before.n_missed + 1


def test_banked_after_cutoff_still_counts(fcon):
    student, assessment = an_early_submission(fcon)
    before = row_of(load_week(fcon, 4), student)
    replace_table(
        fcon,
        "student_assessment",
        [
            f"UPDATE edited_student_assessment SET date_submitted = 40, is_banked = 1 "
            f"WHERE id_student = {student} AND id_assessment = {assessment}"
        ],
    )
    after = row_of(load_week(fcon, 4), student)
    assert after.n_submitted == before.n_submitted
    assert after.n_missed == before.n_missed
    assert after.mean_score == before.mean_score


def test_student_without_clicks_gets_zero(fcon):
    student = a_kept_student(fcon)
    replace_table(
        fcon, "student_vle", [f"DELETE FROM edited_student_vle WHERE id_student = {student}"]
    )
    row = row_of(load_week(fcon, 4), student)
    assert row.clicks_total == 0
    assert row.active_days_total == 0
    assert row.weeks_active == 0
    assert row.clicks_last_week == 0
    assert row.clicks_before_start == 0
