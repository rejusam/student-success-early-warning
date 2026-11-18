"""Train and score the Day 2 baseline. The logic is in src/early_warning/baseline.py.

Usage:
  uv run python scripts/train_baseline.py --mode dev    # trains on 2013B+2013J, scores 2014B
  uv run python scripts/train_baseline.py --mode test   # trains on 2013B+2013J+2014B, scores 2014J
"""

import sys

from early_warning.baseline import main

if __name__ == "__main__":
    sys.exit(main())
