"""Write the monitoring report. The logic is in src/early_warning/monitor.py.

Usage:
  uv run python scripts/monitor.py --mode test   # 2014J, after release_model.py --mode test
"""

import sys

from early_warning.monitor import main

if __name__ == "__main__":
    sys.exit(main())
