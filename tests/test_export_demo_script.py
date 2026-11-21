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
