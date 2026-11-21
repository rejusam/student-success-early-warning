import json
from pathlib import Path

import httpx
import pytest

from early_warning.demo_export import build_facts, score_week

REPO = Path(__file__).resolve().parents[1]


def rows(n):
    return [{"id_student": i, "code_module": "AAA"} for i in range(n)]


def answer(request, status=200):
    body = json.loads(request.content)
    results = [
        {"id_student": s["id_student"], "risk": 0.5, "reasons": []} for s in body["students"]
    ]
    return httpx.Response(status, json={"week": body["week"], "results": results})


def client(handler):
    return httpx.Client(base_url="https://api.example", transport=httpx.MockTransport(handler))


def fast(c, week, data):
    return score_week(c, week, data, pause=0, waits=(0, 0, 0, 0))


def test_scores_in_chunks_of_500():
    bodies = []

    def handler(request):
        assert request.url.path == "/v1/score"
        bodies.append(json.loads(request.content))
        return answer(request)

    out = fast(client(handler), 8, rows(1201))
    assert [len(b["students"]) for b in bodies] == [500, 500, 201]
    assert all(set(b) == {"week", "students"} and b["week"] == 8 for b in bodies)
    assert len(out) == 1201


def test_retries_429_then_succeeds():
    statuses = iter([429, 503, 200])
    calls = []

    def handler(request):
        calls.append(1)
        status = next(statuses)
        return answer(request) if status == 200 else httpx.Response(status, json={})

    assert len(fast(client(handler), 4, rows(3))) == 3
    assert len(calls) == 3


def test_gives_up_after_five_failures():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(503, json={})

    with pytest.raises(RuntimeError, match="503"):
        fast(client(handler), 4, rows(3))
    assert len(calls) == 5


def test_results_must_cover_every_id():
    def handler(request):
        response = answer(request)
        body = json.loads(response.content)
        body["results"] = body["results"][1:]
        return httpx.Response(200, json=body)

    with pytest.raises(RuntimeError, match="ids"):
        fast(client(handler), 8, rows(3))


def test_facts_values_match_sources():
    meta = {"weeks": {"2": {"AAA": {"cohort": 10, "listed": 2}, "BBB": {"cohort": 5, "listed": 1}}}}
    facts = build_facts(REPO, meta)
    value = {k: v["value"] for k, v in facts.items()}
    assert value["aws_rate"] == 5
    assert value["aws_burst"] == 10
    assert value["api_max_students"] == 500
    assert value["list_pct"] == 20
    assert value["proxy_ip_per_min"] == 60
    assert value["proxy_max_students"] == 5
    assert value["cohort_w2"] == 15
    assert value["recall_w8"] == 0.4135
    assert value["recall_women_w12"] == 0.4132
    assert (value["psi_moderate"], value["psi_major"]) == (0.1, 0.25)
    assert all(v["source"] for v in facts.values())


def test_retries_a_dropped_connection():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) < 3:
            raise httpx.ConnectError("connection reset", request=request)
        return answer(request)

    assert len(fast(client(handler), 8, rows(2))) == 2
    assert len(calls) == 3


def test_failed_export_writes_no_module_files(tmp_path, monkeypatch):
    import pandas as pd

    from early_warning import demo_export

    def frame(module, ids):
        row = {f: 1 for f in demo_export.REQUEST_FIELDS}
        return [
            {**row, "id_student": i, "code_module": module, "code_presentation": "2014J"}
            for i in ids
        ]

    data = pd.DataFrame(frame("AAA", range(5)) + frame("BBB", range(5, 10)))
    monkeypatch.setattr(demo_export, "register_tables", lambda con, path: None)
    monkeypatch.setattr(demo_export, "create_views", lambda con: None)
    monkeypatch.setattr(demo_export, "load_week", lambda con, week: data)
    monkeypatch.setattr(demo_export.time, "sleep", lambda s: None)

    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        body = json.loads(request.content)
        if body["students"][0]["code_module"] == "BBB":
            return httpx.Response(400, json={"detail": "bad"})
        return answer(request)

    real_client = httpx.Client
    monkeypatch.setattr(
        demo_export.httpx,
        "Client",
        lambda base_url: real_client(base_url=base_url, transport=httpx.MockTransport(handler)),
    )
    out = tmp_path / "demo"
    with pytest.raises(RuntimeError, match="400"):
        demo_export.main(["--api-url", "https://api.example", "--out", str(out)])
    assert not list(out.glob("*.json"))


def test_s3_check_refuses_an_empty_folder(tmp_path):
    from early_warning.demo_export import main

    empty = tmp_path / "s3"
    empty.mkdir()
    with pytest.raises(ValueError, match="no CSV"):
        main(["--check-s3", str(empty), "--out", str(tmp_path)])


def test_export_needs_api_url():
    from early_warning.demo_export import main

    with pytest.raises(SystemExit):
        main([])


def test_fact_sources_are_repo_files():
    meta = {"weeks": {"2": {"AAA": {"cohort": 10, "listed": 2}}}}
    facts = build_facts(REPO, meta, meta_path="build/demo/meta.json")
    for key, fact in facts.items():
        path = fact["source"].split("#")[0]
        if key.startswith("cohort_"):
            assert fact["source"] == "build/demo/meta.json#weeks.2"
        else:
            assert (REPO / path).is_file(), (key, path)


def test_readme_limits_match_their_sources():
    from early_warning.api import MAX_STUDENTS
    from early_warning.demo_export import terraform_default

    limits = json.loads((REPO / "proxy/limits.json").read_text())
    rate = terraform_default(REPO / "infra/service/variables.tf", "throttle_rate")
    readme = " ".join((REPO / "README.md").read_text().split())
    proxy_readme = " ".join((REPO / "proxy/README.md").read_text().split())
    assert (
        f"at most {limits['max_students']} students and {limits['max_body_bytes']} bytes a call"
        f" (the API allows {MAX_STUDENTS}), allows {limits['ip_per_minute']} calls a minute per"
        f" address and {limits['upstream_per_minute']} a minute to AWS"
    ) in readme
    assert f"rate limited ({rate} requests a second)" in readme
    assert f"at most {limits['max_body_bytes']} bytes" in proxy_readme
    assert f"{limits['upstream_per_minute']} calls a minute to AWS" in proxy_readme
    assert f"API throttle of {rate} a second" in proxy_readme
