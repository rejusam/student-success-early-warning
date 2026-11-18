"""Audit the weekly models by student group. The logic is in src/early_warning/audit.py.

Usage:
  uv run python scripts/audit_fairness.py --mode test   # 2014J, after train_gbm.py --mode test
"""

import sys

from early_warning.audit import main

if __name__ == "__main__":
    sys.exit(main())
