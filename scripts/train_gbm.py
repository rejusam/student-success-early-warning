"""Tune, test, gate and register the Day 3 model. The logic is in src/early_warning/train_gbm.py.

Usage:
  uv run python scripts/train_gbm.py --mode tune   # fit 2013B+2013J, choose settings on 2014B
  uv run python scripts/train_gbm.py --mode test   # refit on three presentations, score 2014J once
"""

import sys

from early_warning.train_gbm import main

if __name__ == "__main__":
    sys.exit(main())
