import pytest

from early_warning.views import create_views


@pytest.fixture
def vcon(con):
    create_views(con)
    return con


def test_student_module_one_row_per_key(vcon):
    total, distinct = vcon.execute(
        "SELECT count(*), count(DISTINCT (code_module, code_presentation, id_student)) "
        "FROM student_module"
    ).fetchone()
    assert total == distinct == 80


def test_at_risk_flag(vcon):
    rows = dict(
        vcon.execute(
            "SELECT final_result, bool_and(at_risk) FROM student_module GROUP BY 1"
        ).fetchall()
    )
    assert rows == {"Withdrawn": True, "Fail": True, "Pass": False, "Distinction": False}


def test_negative_day_maps_to_negative_week(vcon):
    vcon.execute("CREATE TABLE sv AS SELECT * FROM student_vle LIMIT 3")
    vcon.execute("UPDATE sv SET date = -1 WHERE rowid = 0")
    vcon.execute("UPDATE sv SET date = -8 WHERE rowid = 1")
    vcon.execute("UPDATE sv SET date = 6 WHERE rowid = 2")
    vcon.execute("CREATE OR REPLACE VIEW student_vle AS SELECT * FROM sv")
    weeks = sorted(w for (w,) in vcon.execute("SELECT week FROM vle_weekly").fetchall())
    assert weeks == [-2, -1, 0]


def test_vle_weekly_clicks_sum_matches_raw(vcon):
    raw = vcon.execute("SELECT sum(sum_click) FROM student_vle").fetchone()[0]
    weekly = vcon.execute("SELECT sum(clicks) FROM vle_weekly").fetchone()[0]
    assert weekly == raw


def test_assessment_days_late(vcon):
    row = vcon.execute(
        "SELECT due_date, date_submitted, days_late FROM assessment_submissions "
        "WHERE due_date IS NOT NULL LIMIT 1"
    ).fetchone()
    assert row[2] == row[1] - row[0]
