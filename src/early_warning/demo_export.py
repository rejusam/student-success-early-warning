"""Export the 2014J cohort, scored by the live API, for the demo page.

The page shows one file per week and module: every student's request row, the risk the live API
returned, and the reasons for the students on the list and just below it. Scoring goes through
the live API rather than a local model bundle, so the risk in a list always equals the risk the
page gets when it scores the same student live.

Run with scripts/export_demo.py.
"""

import argparse
import csv
import json
import math
import re
import time
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import httpx
import pandas as pd

from early_warning.api import MAX_STUDENTS, Student
from early_warning.drift import MAJOR, MODERATE
from early_warning.features import WEEKS, load_week
from early_warning.ingest import register_tables
from early_warning.metrics import flag_top_k
from early_warning.scoring import LIST_FRACTION
from early_warning.views import create_views

REQUEST_FIELDS = list(Student.model_fields)
INT_FIELDS = [f for f, info in Student.model_fields.items() if info.annotation in (int, int | None)]
BELOW_LIST = 3  # students shown "just below the list"


def request_rows(df: pd.DataFrame) -> list[dict]:
    """One API request row per student: exactly the Student fields, plain Python values."""
    rows = []
    for record in df[REQUEST_FIELDS].to_dict("records"):
        row = {}
        for field in REQUEST_FIELDS:
            value = record[field]
            if isinstance(value, float) and math.isnan(value):
                value = None
            elif field in INT_FIELDS and value is not None:
                value = int(value)
            elif field == "mean_score" and value is not None:
                value = float(value)
            row[field] = value
        rows.append(row)
    return rows


def module_file(week: int, module: str, rows: list[dict], scored: list[dict]) -> dict:
    """The page's file for one week and module, ranked by risk (ties: lower id first).

    The first `listed` rows are the advisor's list (same rule as the batch job). Reasons are kept
    for the list and the students just below it; the rest of the cohort gets None to keep the
    file small.

    Ranking uses the risk the API returns, rounded to 4 decimal places. The batch job ranks on
    the unrounded probabilities, so when two students tie after rounding at the cut, the batch
    list can hold a different one of them; check_against_s3 reports such cases.
    """
    by_id = {s["id_student"]: s for s in scored}
    if set(by_id) != {r["id_student"] for r in rows}:
        raise ValueError(f"scores do not match the students of week {week} {module}")
    df = pd.DataFrame({"id_student": [r["id_student"] for r in rows]})
    df["risk"] = [by_id[i]["risk"] for i in df.id_student]
    df["code_module"], df["code_presentation"] = module, "2014J"
    df["listed"] = flag_top_k(df, "risk", LIST_FRACTION)
    df = df.sort_values(["risk", "id_student"], ascending=[False, True])
    listed = int(df.listed.sum())
    row_by_id = {r["id_student"]: r for r in rows}
    out_rows = []
    for position, sid in enumerate(df.id_student):
        reasons = by_id[sid]["reasons"] if position < listed + BELOW_LIST else None
        values = [row_by_id[sid][f] for f in REQUEST_FIELDS]
        out_rows.append([*values, by_id[sid]["risk"], reasons])
    return {
        "week": week,
        "module": module,
        "presentation": "2014J",
        "cohort": len(rows),
        "listed": listed,
        "columns": [*REQUEST_FIELDS, "risk", "reasons"],
        "rows": out_rows,
    }


def facts(sources: dict[str, tuple[str, object]]) -> dict:
    """{key: (source, value)} -> {key: {"value", "source"}}, the page's numbers and where each
    comes from."""
    return {key: {"value": value, "source": source} for key, (source, value) in sources.items()}


PRESENTATION = "2014J"
RETRY_STATUS = {429, 500, 502, 503, 504}


def score_week(
    client: httpx.Client,
    week: int,
    rows: list[dict],
    pause: float = 0.3,
    waits: tuple[float, ...] = (1, 2, 4, 8),
) -> list[dict]:
    """Score rows through POST /v1/score, 500 students per call.

    The pause keeps the calls under the API's throttle of 5 a second. A 429, a 5xx, a timeout
    or a dropped connection is retried after each wait in turn (five tries in all); then the
    export stops.
    """
    results = []
    for start in range(0, len(rows), MAX_STUDENTS):
        chunk = rows[start : start + MAX_STUDENTS]
        body = post_with_retry(client, {"week": week, "students": chunk}, waits)
        ids = [s["id_student"] for s in chunk]
        if sorted(r["id_student"] for r in body["results"]) != sorted(ids):
            raise RuntimeError(f"week {week}: the API returned different ids from those sent")
        results.extend(body["results"])
        time.sleep(pause)
    return results


def post_with_retry(client: httpx.Client, payload: dict, waits: tuple[float, ...]) -> dict:
    problem = ""
    for attempt in range(len(waits) + 1):
        if attempt:
            time.sleep(waits[attempt - 1])
        try:
            response = client.post("/v1/score", json=payload, timeout=30)
        except httpx.TransportError as e:  # timeouts and dropped connections
            problem = type(e).__name__
            continue
        if response.status_code in RETRY_STATUS:
            problem = f"status {response.status_code}"
            continue
        if response.status_code != 200:
            raise RuntimeError(f"status {response.status_code}: {response.text[:500]}")
        return response.json()
    raise RuntimeError(f"gave up after {len(waits) + 1} tries, last: {problem}")


def terraform_default(path: Path, variable: str) -> int:
    """The number after `default =` in a Terraform variable block."""
    block = re.search(rf'variable "{variable}" {{(.*?)\n}}', path.read_text(), re.S)
    return int(re.search(r"default\s*=\s*(\d+)", block.group(1)).group(1))


def fairness_value(path: Path, week: int, group: str, column: str) -> float:
    """One value from fairness_groups.csv for lr_main."""
    with path.open() as f:
        for row in csv.DictReader(f):
            if (row["model"], row["week"], row["group"]) == ("lr_main", str(week), group):
                return float(row[column])
    raise KeyError(f"{group} at week {week} not in {path}")


def build_facts(repo: Path, meta: dict, meta_path: str = "build/demo/meta.json") -> dict:
    """Every number the page's text uses, read from the file it comes from.

    Cohort sizes come from meta.json, which this export writes to meta_path.
    """
    limits = json.loads((repo / "proxy/limits.json").read_text())
    tf = repo / "infra/service/variables.tf"
    metrics = json.loads((repo / "reports/baseline_metrics.json").read_text())["metrics"]
    fairness = repo / "reports/fairness_groups.csv"
    sources = {
        "list_pct": (
            "src/early_warning/scoring.py#LIST_FRACTION",
            round(LIST_FRACTION * 100),
        ),
        "api_max_students": ("src/early_warning/api.py#MAX_STUDENTS", MAX_STUDENTS),
        "aws_rate": (
            "infra/service/variables.tf#throttle_rate",
            terraform_default(tf, "throttle_rate"),
        ),
        "aws_burst": (
            "infra/service/variables.tf#throttle_burst",
            terraform_default(tf, "throttle_burst"),
        ),
        "proxy_max_students": ("proxy/limits.json#max_students", limits["max_students"]),
        "proxy_max_bytes": ("proxy/limits.json#max_body_bytes", limits["max_body_bytes"]),
        "proxy_ip_per_min": ("proxy/limits.json#ip_per_minute", limits["ip_per_minute"]),
        "proxy_upstream_per_min": (
            "proxy/limits.json#upstream_per_minute",
            limits["upstream_per_minute"],
        ),
    }
    sources["psi_moderate"] = ("src/early_warning/drift.py#MODERATE", MODERATE)
    sources["psi_major"] = ("src/early_warning/drift.py#MAJOR", MAJOR)
    for week, modules in meta["weeks"].items():
        total = sum(m["cohort"] for m in modules.values())
        sources[f"cohort_w{week}"] = (f"{meta_path}#weeks.{week}", total)
    for week, name in [(2, "w2"), (8, "w8")]:
        top = metrics[str(week)]["lr_main"]["top_20"]
        path = f"reports/baseline_metrics.json#metrics.{week}.lr_main.top_20"
        sources[f"recall_{name}"] = (f"{path}.recall", top["recall"])
        sources[f"precision_{name}"] = (f"{path}.precision", top["precision"])
    sources["recall_clicks_w2"] = (
        "reports/baseline_metrics.json#metrics.2.clicks_rank.top_20.recall",
        metrics["2"]["clicks_rank"]["top_20"]["recall"],
    )
    for key, week, group, column in [
        ("recall_women_w12", 12, "F", "recall"),
        ("recall_men_w12", 12, "M", "recall"),
        ("calibration_gap_w2", 2, "All students", "calibration_gap"),
    ]:
        source = f"reports/fairness_groups.csv#lr_main,{week},{group},{column}"
        sources[key] = (source, fairness_value(fairness, week, group, column))
    return facts(sources)


def check_against_s3(out: Path, s3_dir: Path) -> dict:
    """Compare the week-12 export with the batch job's lists (CSVs copied from S3).

    Per module: are the same ids listed, and are their risks equal to 4 decimal places? A tie
    in risk at the cut can list a different student; that is reported, not treated as an error.
    """
    paths = sorted(s3_dir.glob("*.csv"))
    if not paths:
        raise ValueError(f"no CSV files in {s3_dir}; copy the week-12 lists from S3 first")
    modules = {}
    for path in paths:
        module = path.stem
        batch = {int(r["id_student"]): float(r["risk"]) for r in csv.DictReader(path.open())}
        export = json.loads((out / f"w12-{module}.json").read_text())
        listed = export["rows"][: export["listed"]]
        mine = {r[0]: r[-2] for r in listed}
        same_ids = set(batch) & set(mine)
        modules[module] = {
            "batch_listed": len(batch),
            "export_listed": len(mine),
            "same_ids": len(same_ids),
            "only_in_batch": sorted(set(batch) - set(mine)),
            "only_in_export": sorted(set(mine) - set(batch)),
            "risk_differs_4dp": sorted(i for i in same_ids if round(batch[i], 4) != mine[i]),
        }
    return {
        "checked_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "presentation": PRESENTATION,
        "week": 12,
        "modules": modules,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export 2014J, scored live, for the demo page.")
    parser.add_argument("--api-url", help="the live API; needed unless --check-s3 is given")
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("build/demo"))
    parser.add_argument("--check-s3", type=Path, help="folder of week-12 CSVs copied from S3")
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[2]

    if args.check_s3:
        result = check_against_s3(args.out, args.check_s3)
        (repo / "reports/demo_export_check.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result["modules"], indent=2))
        return 0
    if not args.api_url:
        parser.error("--api-url is required to export")

    con = duckdb.connect()
    register_tables(con, args.processed)
    create_views(con)
    frames = {week: load_week(con, week) for week in WEEKS}
    con.close()

    # Score everything before writing anything, so a failed run leaves no partial export.
    files = {}
    with httpx.Client(base_url=args.api_url.rstrip("/")) as client:
        health = client.get("/health", timeout=30).raise_for_status().json()
        weeks = {}
        for week in WEEKS:
            df = frames[week][frames[week].code_presentation == PRESENTATION]
            weeks[str(week)] = {}
            for module, part in df.groupby("code_module"):
                rows = request_rows(part)
                file = module_file(week, module, rows, score_week(client, week, rows))
                files[f"w{week}-{module}.json"] = file
                weeks[str(week)][module] = {"cohort": file["cohort"], "listed": file["listed"]}
            print(f"week {week}: {sum(m['cohort'] for m in weeks[str(week)].values())} students")

    meta = {
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "api_url": args.api_url,
        "health": health,
        "presentation": PRESENTATION,
        "list": f"top {LIST_FRACTION:.0%} of each module presentation by risk",
        "weeks": weeks,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    for name, file in files.items():
        (args.out / name).write_text(json.dumps(file, separators=(",", ":")))
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (args.out / "facts.json").write_text(json.dumps(build_facts(repo, meta), indent=2) + "\n")
    return 0
