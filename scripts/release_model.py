"""Register lr_main as champion and export the serving bundle. Logic: src/early_warning/release.py.

Usage:
  uv run python scripts/release_model.py --mode test   # after train_gbm.py --mode test
"""

import sys

from early_warning.release import main

if __name__ == "__main__":
    sys.exit(main())
