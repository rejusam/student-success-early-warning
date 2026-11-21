"""Scoring API for the model in use.

Routes:
- GET /health: status and the version of each week's model.
- POST /v1/score: a prediction week and 1 to 500 students' features; returns each student's
  risk (probability of Fail or Withdrawn) and up to three reasons.

The API does not rank students. A top-20% list needs every student in the module, which is the
batch job's work (batch.py). Run locally with:
  MODEL_DIR=build/model uv run uvicorn early_warning.api:create_app --factory --port 8080
"""

import os
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, model_validator

from early_warning.features import cutoff_day
from early_warning.scoring import load_bundle, score

MODULES = Literal["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"]  # the OULAD modules
MAX_STUDENTS = 500
# Upper bound well above anything in OULAD (largest click total per student and module: 24,139),
# so a huge number gets a 422 rather than an overflow error later.
Count = Field(ge=0, le=1_000_000)
# OULAD student ids go up to 2,716,795, so the id gets its own, larger bound.
StudentId = Field(ge=0, le=100_000_000)


class Student(BaseModel):
    """One student's features at the prediction week, as built by sql/features/student_week.sql.

    Unknown fields are rejected, so demographic fields cannot reach the model.
    """

    model_config = ConfigDict(extra="forbid")

    id_student: int = StudentId
    code_module: MODULES
    clicks_total: int = Count
    clicks_last_week: int = Count
    active_days_total: int = Count
    weeks_active: int = Count
    clicks_before_start: int = Count
    n_due: int = Count
    n_submitted: int = Count
    n_missed: int = Count
    n_late: int = Count
    mean_score: float | None = Field(default=None, ge=0, le=100)  # None: no score yet
    num_of_prev_attempts: int = Count
    studied_credits: int = Count
    date_registration: int | None = Field(default=None, ge=-1000)  # day from module start

    @model_validator(mode="after")
    def counts_agree(self):
        """Rules that hold for every row the feature query builds."""
        if self.n_missed != self.n_due - self.n_submitted:
            raise ValueError("n_missed must equal n_due - n_submitted")
        if self.n_late > self.n_submitted:
            raise ValueError("n_late cannot be more than n_submitted")
        if max(self.clicks_last_week, self.clicks_before_start) > self.clicks_total:
            raise ValueError("clicks_last_week and clicks_before_start are part of clicks_total")
        return self


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    week: Literal[2, 4, 8, 12]
    students: list[Student] = Field(min_length=1, max_length=MAX_STUDENTS)

    @model_validator(mode="after")
    def known_by_the_week(self):
        """A student in the week's cohort registered by the cutoff day, and can have been active
        in at most `week` weeks (weeks 0 to week - 1)."""
        for i, s in enumerate(self.students):
            if s.date_registration is not None and s.date_registration > cutoff_day(self.week):
                raise ValueError(f"students[{i}] registered after the cutoff day of the week")
            if s.weeks_active > self.week:
                raise ValueError(f"students[{i}] has more active weeks than have passed")
        return self


class StudentScore(BaseModel):
    id_student: int
    risk: float
    reasons: list[str]


class ScoreResponse(BaseModel):
    week: int
    model_version: int
    results: list[StudentScore]


def create_app(model_dir: Path | None = None) -> FastAPI:
    """Load the bundle once (MODEL_DIR, default /app/model) and build the app."""
    bundle = load_bundle(model_dir or Path(os.environ.get("MODEL_DIR", "/app/model")))
    weeks = bundle.manifest["weeks"]
    app = FastAPI(title="Student early-warning scoring API", version="1")

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "model_versions": {week: entry["version"] for week, entry in weeks.items()},
            "released_at_commit": bundle.manifest["released_at_commit"],
        }

    @app.post("/v1/score")
    def score_students(request: ScoreRequest) -> ScoreResponse:
        df = pd.DataFrame([s.model_dump() for s in request.students])
        scored = score(bundle, request.week, df)
        results = [
            StudentScore(id_student=sid, risk=round(float(risk), 4), reasons=reasons)
            for sid, risk, reasons in zip(df.id_student, scored.risk, scored.reasons, strict=True)
        ]
        return ScoreResponse(
            week=request.week, model_version=weeks[str(request.week)]["version"], results=results
        )

    return app
