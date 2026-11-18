"""Download OULAD into data/raw. Usage: uv run python scripts/download_oulad.py"""

from pathlib import Path

from early_warning.download import download_oulad

if __name__ == "__main__":
    path = download_oulad(Path("data/raw"))
    print(f"OULAD ready: {path}")
