"""Write the week's risk lists. The logic is in src/early_warning/batch.py.

Usage:
  uv run python scripts/score_batch.py --week 8 --presentation 2014J
"""

import sys

from early_warning.batch import main

if __name__ == "__main__":
    sys.exit(main())
