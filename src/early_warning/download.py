"""Download the OULAD zip from UCI and check it has not changed."""

import hashlib
import urllib.request
import zipfile
from pathlib import Path

OULAD_URL = (
    "https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip"
)
# SHA-256 of the zip as downloaded on 2026-11-18. A different value means the file changed
# or the download was cut short, so we stop instead of ingesting it.
OULAD_SHA256 = "f2ed1902616c1fe8d2824d872c0b7d2d72be435bf0124d077044fe4be2c6d3e4"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_oulad(
    dest_dir: Path, url: str = OULAD_URL, expected_sha256: str = OULAD_SHA256
) -> Path:
    """Download the zip into dest_dir, verify it, and extract the CSV files next to it."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "oulad.zip"

    if not (zip_path.exists() and sha256_of(zip_path) == expected_sha256):
        part_path = dest_dir / "oulad.zip.part"
        urllib.request.urlretrieve(url, part_path)
        actual = sha256_of(part_path)
        if actual != expected_sha256:
            part_path.unlink()
            raise ValueError(f"checksum mismatch: expected {expected_sha256}, got {actual}")
        part_path.replace(zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    return zip_path
