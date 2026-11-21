import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from early_warning.api import MAX_STUDENTS, create_app
from early_warning.features import NUMERIC_FEATURES, model_input
from early_warning.scoring import load_bundle

FIELDS = ["id_student", "code_module", *NUMERIC_FEATURES]


@pytest.fixture(scope="module")
def client(bundle_dir):
    return TestClient(create_app(bundle_dir))


def as_requests(df: pd.DataFrame) -> list[dict]:
    """Rows as JSON-ready dicts: missing values become null, numbers plain Python numbers."""
    rows = df[FIELDS].astype(object).where(df[FIELDS].notna(), None)
    return [
        {k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()}
        for r in rows.to_dict("records")
    ]


@pytest.fixture
def student(frames):
    return as_requests(frames[8].head(1))[0]


def post(client, week, students):
    return client.post("/v1/score", json={"week": week, "students": students})


def test_health_lists_model_versions(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert sorted(body["model_versions"]) == ["12", "2", "4", "8"]


def test_valid_request_is_scored(client, student):
    response = post(client, 8, [student])
    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["id_student"] == student["id_student"]
    assert 0 <= result["risk"] <= 1
    assert isinstance(result["reasons"], list)


def test_every_sample_row_is_accepted_and_scored_like_the_pipeline(client, frames, bundle_dir):
    bundle = load_bundle(bundle_dir)
    for week, df in frames.items():
        response = post(client, week, as_requests(df))
        assert response.status_code == 200, response.text
        risks = [r["risk"] for r in response.json()["results"]]
        expected = bundle.models[week].predict_proba(model_input(df))[:, 1]
        assert np.allclose(risks, expected, atol=5e-5)


@pytest.mark.parametrize(
    "change",
    [
        {"gender": "F"},  # demographic fields are not model inputs
        {"code_module": "HHH"},  # not an OULAD module
        {"clicks_total": -1},
        {"clicks_total": 10**400},  # too large: must be a 422, not an overflow error
        {"id_student": 10**400},
        {"id_student": -1},
        {"mean_score": 101},
        {"n_missed": 99},  # must equal n_due - n_submitted
        {"date_registration": 500},  # registered after the week-8 cutoff day
        {"weeks_active": 9},  # more weeks than have passed
    ],
)
def test_bad_student_is_rejected(client, student, change):
    assert post(client, 8, [{**student, **change}]).status_code == 422


def test_largest_oulad_student_id_is_accepted(client, student):
    # OULAD ids go up to 2,716,795 (2014J), above the bound used for counts
    response = post(client, 8, [{**student, "id_student": 2_716_795}])
    assert response.status_code == 200, response.text
    assert response.json()["results"][0]["id_student"] == 2_716_795


def test_unknown_week_is_rejected(client, student):
    assert post(client, 3, [student]).status_code == 422


def test_batch_size_limit(client, student):
    assert post(client, 8, []).status_code == 422
    assert post(client, 8, [student] * (MAX_STUDENTS + 1)).status_code == 422


def test_missing_score_is_accepted(client, student):
    no_score = {**student, "mean_score": None, "date_registration": None}
    assert post(client, 8, [no_score]).status_code == 200
