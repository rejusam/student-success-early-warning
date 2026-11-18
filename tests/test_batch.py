import json
import math

import pandas as pd
import pytest

from early_warning.batch import main


def run(parquet_dir, bundle_dir, out, presentation="2013J"):
    return main(
        [
            "--processed", str(parquet_dir),
            "--bundle", str(bundle_dir),
            "--week", "8",
            "--presentation", presentation,
            "--out", str(out),
        ]
    )  # fmt: skip


def test_batch_writes_one_list_per_module(parquet_dir, bundle_dir, frames, tmp_path):
    assert run(parquet_dir, bundle_dir, tmp_path) == 0
    folder = tmp_path / "2013J" / "week-8"
    manifest = json.loads((folder / "manifest.json").read_text())
    for module, n in frames[8].groupby("code_module").size().items():
        listed = pd.read_csv(folder / f"{module}.csv")
        assert len(listed) == math.ceil(0.2 * n)
        assert manifest["modules"][module] == {"cohort": n, "listed": len(listed)}


def test_unknown_presentation_stops_without_writing(parquet_dir, bundle_dir, tmp_path):
    with pytest.raises(SystemExit):
        run(parquet_dir, bundle_dir, tmp_path, presentation="2014X")
    assert not any(tmp_path.iterdir())
