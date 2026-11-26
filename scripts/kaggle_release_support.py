"""Small, standalone data acquisition helpers embedded in the Kaggle releases."""

import hashlib
import shutil
import tarfile
import urllib.request
from pathlib import Path


def verified_download(url: str, target: Path, expected_sha256: str) -> Path:
    """Download to a temporary file and accept only the reviewed dataset bytes."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == expected_sha256:
        return target
    temporary = target.with_suffix(target.suffix + ".download")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError("Downloaded data changed; review its provenance before running.")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def unpack_regular_tar(archive: Path, destination: Path) -> None:
    """Reject links, special files and traversal before writing any archive member."""
    root = destination.resolve()
    with tarfile.open(archive) as bundle:
        members = bundle.getmembers()
        for member in members:
            path = root / member.name
            if (
                Path(member.name).is_absolute()
                or not path.resolve().is_relative_to(root)
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError("Unsafe dataset archive member.")
        for member in members:
            path = root / member.name
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(member) as source, path.open("wb") as target:
                    shutil.copyfileobj(source, target)
