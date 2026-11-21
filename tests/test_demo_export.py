import math

import pandas as pd

from early_warning.api import ScoreRequest
from early_warning.demo_export import REQUEST_FIELDS, facts, module_file, request_rows
from early_warning.metrics import flag_top_k


def student(id_student, **changes):
    row = {
        "id_student": id_student,
        "code_module": "AAA",
        "clicks_total": 100,
        "clicks_last_week": 10,
        "active_days_total": 12,
        "weeks_active": 5,
        "clicks_before_start": 5,
        "n_due": 2,
        "n_submitted": 1,
        "n_missed": 1,
        "n_late": 0,
        "mean_score": 70.0,
        "num_of_prev_attempts": 0,
        "studied_credits": 60,
        "date_registration": -20,
    }
    row.update(changes)
    return row


def ranked_case(risks):
    rows = [student(i + 1) for i in range(len(risks))]
    scored = [
        {"id_student": r["id_student"], "risk": risk, "reasons": [f"reason {r['id_student']}"]}
        for r, risk in zip(rows, risks, strict=True)
    ]
    return rows, scored


def test_request_rows_have_exact_fields():
    df = pd.DataFrame([student(1), student(2, mean_score=float("nan"), n_submitted=0, n_missed=2)])
    df["code_presentation"] = "2014J"
    df["final_result"] = "Pass"
    rows = request_rows(df)
    assert [list(r) for r in rows] == [REQUEST_FIELDS, REQUEST_FIELDS]
    assert rows[1]["mean_score"] is None
    assert type(rows[0]["clicks_total"]) is int
    ScoreRequest(week=8, students=rows)


def test_module_file_ranks_like_flag_top_k():
    risks = [0.1, 0.5, 0.9, 0.5, 0.2, 0.3, 0.7, 0.05, 0.6, 0.4, 0.15]  # ids 2 and 4 tie
    rows, scored = ranked_case(risks)
    out = module_file(8, "AAA", rows, list(reversed(scored)))
    assert out["cohort"] == 11
    assert out["listed"] == 3
    assert out["columns"] == [*REQUEST_FIELDS, "risk", "reasons"]
    ids = [r[0] for r in out["rows"]]
    expected = sorted(range(1, 12), key=lambda i: (-risks[i - 1], i))
    assert ids == expected
    df = pd.DataFrame({"id_student": range(1, 12), "risk": risks})
    df["code_module"], df["code_presentation"] = "AAA", "2014J"
    flagged = set(df.loc[flag_top_k(df, "risk", 0.2), "id_student"])
    assert set(ids[: out["listed"]]) == flagged


def test_reasons_kept_for_list_and_three_below():
    rows, scored = ranked_case([1 - i / 100 for i in range(20)])
    out = module_file(4, "AAA", rows, scored)
    assert out["listed"] == math.ceil(0.2 * 20) == 4
    reasons = [r[-1] for r in out["rows"]]
    assert all(isinstance(x, list) for x in reasons[:7])
    assert all(x is None for x in reasons[7:])


def test_every_fixture_row_passes_api_rules(frames):
    rows = request_rows(frames[8])
    assert rows
    for start in range(0, len(rows), 500):
        ScoreRequest(week=8, students=rows[start : start + 500])


def test_facts_keep_value_and_source():
    out = facts({"list_pct": ("src/early_warning/scoring.py#LIST_FRACTION", 20)})
    assert out == {
        "list_pct": {"value": 20, "source": "src/early_warning/scoring.py#LIST_FRACTION"}
    }
