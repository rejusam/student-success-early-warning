import numpy as np

from early_warning.baseline import build_pipeline
from early_warning.explain import GROUPS, pushes, top_reasons
from early_warning.labels import LABELS, MISSING_LABELS

KNOWN = (*LABELS.values(), *MISSING_LABELS.values(), "submitted ", "course-site activity")


def fit(df):
    return build_pipeline(False).fit(df, df.at_risk.astype(int))


def test_pushes_add_up_to_log_odds(frames):
    df = frames[8]
    model = fit(df)
    total = pushes(model, df).sum(axis=1) + model[-1].intercept_[0]
    assert np.allclose(total, model.decision_function(df))


def test_pushes_have_one_column_per_original_input(frames):
    df = frames[8]
    columns = list(pushes(fit(df), df).columns)
    assert "code_module" in columns
    assert not [c for c in columns if c.startswith(("missingindicator_", "code_module_"))]


def test_reasons_skip_module_and_use_only_upward_pushes(frames):
    df = frames[8]
    model = fit(df)
    values = pushes(model, df)
    reasons = top_reasons(model, df)
    for index, row in values.iterrows():
        assert len(reasons[index]) == min(3, int((row.drop("code_module") > 0).sum()))
        assert all(text.startswith(KNOWN) for text in reasons[index])


def test_missing_score_is_described_as_missing(frames):
    df = frames[2]
    model = fit(df)
    # Make a missing score push the risk up strongly, so it is always among the reasons.
    names = list(model[:-1].get_feature_names_out())
    model[-1].coef_[0][names.index("missingindicator_mean_score")] = 10.0
    reasons = top_reasons(model, df)
    no_score = df.mean_score.isna()
    assert no_score.any()
    texts = [r for index in df.index[no_score] for r in reasons[index]]
    assert "no assessment score yet" in texts
    assert not [r for r in texts if r.startswith("average assessment score")]


def test_grouped_features_are_one_reason_each(frames):
    df = frames[12]
    model = fit(df)
    columns = set(pushes(model, df).columns)
    assert {"submissions", "activity"} <= columns
    assert not columns & {*GROUPS["submissions"], *GROUPS["activity"]}
    texts = [r for reasons in top_reasons(model, df) for r in reasons]
    assert not [r for r in texts if r.startswith(("assessments due", "clicks on the course"))]
