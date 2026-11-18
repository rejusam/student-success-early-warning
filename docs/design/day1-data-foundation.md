# Day 1 design: project scaffold and data foundation

Date: 2026-11-18. Status: approved by Reju.

## Why
The project has to show that a model can be run the way a production team runs one: versioned
data, tests, CI, and later automated deployment. Day 1 builds the base everything else depends
on, which is a reproducible download of OULAD, validated tables, and SQL views that are keyed by
day so that weekly features cannot leak future data.

## Goals
- `uv sync --locked` followed by `uv run pytest` passes on a clean machine.
- `scripts/download_oulad.py` downloads OULAD from UCI and checks its SHA-256.
- Ingest step converts CSV to Parquet. Data checks are SQL queries run by pytest.
- SQL views cover the student-module base, weekly VLE clicks and assessment submissions, all
  carrying a `date` / `week` column.
- `docs/data.md` records the licence, citation and fields, checked against the downloaded files.
- CI on GitHub Actions runs ruff, pytest on a small committed sample, and gitleaks.

## Non-goals (Day 1)
- No models, feature engineering beyond the views, MLflow, Docker or AWS.

## Anti-goals
- Raw or processed OULAD files committed to git (only a small sample fixture).
- Any number or dataset fact in docs that a file or URL does not back up.
- Code that Reju cannot explain line by line.

## Approaches considered
1. **Plain SQL files run by DuckDB, checks as pytest + SQL (chosen).** One tool to explain, and
   it runs locally and in CI with no server.
2. dbt-duckdb. This is the industry standard for SQL models and tests, but it adds a project
   layout, a profiles file and a CLI to explain. It isn't worth it for about 5 views in a 6-day build.
3. pandera or Great Expectations for validation. Both are capable, but SQL checks cover row
   counts, keys and allowed values just as clearly.

## Decided practices
- uv with Python 3.12, `uv.lock` committed, CI uses `uv sync --locked`.
- ruff (lint and format) and pytest. Layout: `src/early_warning/`, `sql/`, `scripts/`, `tests/`,
  `data/` (gitignored), `reports/`.
- pre-commit hooks: ruff, ruff-format, gitleaks, and a large-file guard, so problems are caught
  before a commit.
- GitHub Actions: third-party actions pinned to full commit SHAs, `permissions: contents: read`,
  Dependabot weekly for actions and uv.
- `.gitignore` covers `data/`, `.env`, `mlruns/`. `.env.example` holds names only.

## Left to the builder
- Exact view columns, the sample fixture size, and test names.
