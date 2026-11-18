# student-success-early-warning

Goal: an early-warning model for students at risk of failing or withdrawing, built to run the way
a university data science team would run it in production. Planned parts: versioned data pipeline,
tested training job, evaluation gate, model registry, fairness audit, cloud-deployed scoring API,
scheduled retraining and drift monitoring. See Status for what exists today.

Data: Open University Learning Analytics Dataset (OULAD), CC BY 4.0
(Kuzilek, Hlosta & Zdrahal, 2015, https://doi.org/10.24432/C5KK69). Details, quirks and checks:
[`docs/data.md`](docs/data.md).

## Status
Done so far: data foundation. Download with checksum, typed CSV-to-Parquet ingest, SQL data
checks, and day-keyed SQL views. Models, deployment and monitoring are not built yet.

## Quickstart
Needs [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked
uv run python scripts/download_oulad.py   # data/raw/, checksum verified
uv run python scripts/ingest.py           # data/processed/*.parquet
uv run python scripts/run_checks.py       # reports/data_checks.json, reports/data_profile.json
uv run pytest                             # runs on a small sample in tests/fixtures/
```

Optional: `uv run pre-commit install` runs ruff and a gitleaks secret scan before each commit.

## Layout
| Path | Contents |
|---|---|
| `src/early_warning/` | download, ingest, checks and views code |
| `sql/checks/` | one SQL file per data rule; each returns the rows that break it |
| `sql/views/` | `student_module`, `vle_weekly`, `assessment_submissions` |
| `scripts/` | command-line entry points |
| `reports/` | outputs of the scripts; every number quoted in the docs comes from here |
| `docs/design/` | design notes |
