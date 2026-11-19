import json

import pandas as pd
import pytest

from early_warning.features import WEEKS
from early_warning.monitor import main, outcomes


def test_smoke_run_writes_both_reports(parquet_dir, bundle_dir, tmp_path):
    args = ["--mode", "smoke", "--processed", str(parquet_dir), "--bundle", str(bundle_dir)]
    assert main([*args, "--out", str(tmp_path)]) == 0
    report = json.loads((tmp_path / "monitoring.json").read_text())
    assert sorted(report["weeks"]) == sorted(str(w) for w in WEEKS)
    for result in report["weeks"].values():
        assert result["drift"]["status"] == "ok"  # same students as training
        assert result["outcomes"]["status"] in {"ok", "alert"}
    text = (tmp_path / "monitoring_report.md").read_text()
    assert "| 12 |" in text and "id_student" not in text


def test_smoke_refuses_reports(parquet_dir, bundle_dir):
    with pytest.raises(SystemExit):
        main(["--mode", "smoke", "--processed", str(parquet_dir), "--bundle", str(bundle_dir)])


def students(n=10):
    return pd.DataFrame(
        {
            "code_module": "AAA",
            "code_presentation": "2014J",
            "id_student": range(n),
            "at_risk": [True] * (n // 2) + [False] * (n - n // 2),
        }
    )


def test_recall_below_nine_tenths_of_expected_is_an_alert():
    df = students()
    risk = pd.Series([0.9, 0.1, 0.1, 0.1, 0.1, 0.8, 0.8, 0.8, 0.8, 0.8])  # mean 0.5 = rate
    # top 20% = 2 students: one at risk, recall 1/5 = 0.2
    assert outcomes(df, risk, {"recall": 0.2, "precision": 0.5, "rate": 0.5})["status"] == "ok"
    result = outcomes(df, risk, {"recall": 0.23, "precision": 0.5, "rate": 0.5})
    assert result["status"] == "alert"  # 0.2 < 0.9 * 0.23


def test_calibration_gap_over_five_points_is_an_alert():
    df = students()
    risk = pd.Series([0.9, 0.9, 0.5, 0.5, 0.5, 0.1, 0.1, 0.5, 0.5, 0.6])  # mean 0.51
    assert outcomes(df, risk, {"recall": 0.4, "precision": 1, "rate": 0.5})["status"] == "ok"
    assert (
        outcomes(df, risk + 0.05, {"recall": 0.4, "precision": 1, "rate": 0.5})["status"] == "alert"
    )
