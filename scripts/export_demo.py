"""Export the 2014J cohort, scored by the live API, for the demo page.

The logic is in src/early_warning/demo_export.py.

Usage:
  uv run python scripts/export_demo.py --api-url https://au3hc4ic73.execute-api.ap-southeast-2.amazonaws.com
  uv run python scripts/export_demo.py --api-url ... --check-s3 build/s3-week-12
"""

import sys

from early_warning.demo_export import main

if __name__ == "__main__":
    sys.exit(main())
