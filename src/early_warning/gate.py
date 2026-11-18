"""Evaluation gate: may the new model replace the baseline?

Agreed on Day 2: the new model must beat both lr_main and clicks_rank on recall@top-20% at
weeks 4, 8 and 12. Week 2 is reported but not gated. "Beat" means strictly higher, so a tie
fails. The weekly models go to advisors as one set, so the set passes only if every gated
week passes.
"""

GATED_WEEKS = (4, 8, 12)
CANDIDATE = "gbm"
BASELINES = ("lr_main", "clicks_rank")
METRIC = "top_20"


def recall(metrics: dict, week: int, ranking: str) -> float:
    return metrics[week][ranking][METRIC]["recall"]


def check_gate(metrics: dict) -> dict:
    """metrics[week][ranking] is the output of metrics.evaluate(). Returns the gate result."""
    weeks = {}
    for week in GATED_WEEKS:
        candidate = recall(metrics, week, CANDIDATE)
        baselines = {name: recall(metrics, week, name) for name in BASELINES}
        weeks[str(week)] = {
            CANDIDATE: candidate,
            **baselines,
            "passed": all(candidate > value for value in baselines.values()),
        }
    return {
        "rule": (
            f"{CANDIDATE} recall@top-20% must be strictly higher than "
            f"{' and '.join(BASELINES)} at weeks {', '.join(map(str, GATED_WEEKS))}"
        ),
        "weeks": weeks,
        "passed": all(w["passed"] for w in weeks.values()),
    }


def check_baseline_matches(fresh: dict, saved: dict, rankings: tuple = BASELINES) -> None:
    """Stop if the baseline numbers computed now differ from the saved Day 2 report.

    fresh[week][ranking] comes from this run; saved is reports/baseline_metrics.json
    ("metrics" keyed by the week as a string). A difference means the data or the code
    changed since the baseline was reported, and the comparison would not be fair.
    """
    for week, by_ranking in fresh.items():
        for ranking in rankings:
            now = by_ranking[ranking]
            before = saved["metrics"][str(week)][ranking]
            if now != before:
                raise ValueError(
                    f"week {week} {ranking}: this run differs from baseline_metrics.json"
                )
