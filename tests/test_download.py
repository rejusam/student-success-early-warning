import zipfile
from pathlib import Path

import pytest

from early_warning.download import download_oulad, sha256_of


def make_zip(path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("a.csv", "x\n1\n")
    return path


def test_sha256_of_known_bytes(tmp_path):
    f = tmp_path / "abc.txt"
    f.write_bytes(b"abc")
    assert sha256_of(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_download_rejects_bad_checksum(tmp_path):
    src = make_zip(tmp_path / "src.zip")
    dest = tmp_path / "raw"
    with pytest.raises(ValueError, match="checksum"):
        download_oulad(dest, url=src.as_uri(), expected_sha256="0" * 64)
    assert not (dest / "oulad.zip").exists()
    assert not (dest / "oulad.zip.part").exists()


def test_download_extracts_csvs(tmp_path):
    src = make_zip(tmp_path / "src.zip")
    dest = tmp_path / "raw"
    zip_path = download_oulad(dest, url=src.as_uri(), expected_sha256=sha256_of(src))
    assert zip_path == dest / "oulad.zip"
    assert (dest / "a.csv").read_text() == "x\n1\n"
