"""Stage a reproducible image subset in fresh vision-comparison smoke caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
COMPARISONS = [
    "lab-comparative-analysis-of-keras-and-pytorch-models",
    "lab-land-classification-cnn-transformer-integration-evaluation",
]


def prepare(archive: Path, workspace: Path, per_class: int) -> dict:
    """Copy selected regular image members without extracting archive paths or links."""
    if per_class < 1:
        raise ValueError("per_class must be positive.")
    dataset = workspace / "images_dataSAT"
    if dataset.exists():
        raise FileExistsError("The cache already has images_dataSAT; use a fresh cache.")
    selected = []
    with tarfile.open(archive) as source:
        for label in ["class_0_non_agri", "class_1_agri"]:
            members = sorted(
                (
                    member
                    for member in source.getmembers()
                    if member.isfile()
                    and "\\" not in member.name
                    and ":" not in member.name
                    and len(PurePosixPath(member.name).parts) == 3
                    and PurePosixPath(member.name).parts[:2] == ("images_dataSAT", label)
                    and PurePosixPath(member.name).suffix.lower() in {".jpg", ".jpeg", ".png"}
                ),
                key=lambda member: member.name,
            )
            if len(members) < per_class:
                raise ValueError(f"Not enough images in {label}.")
            selected.extend(members[:per_class])
        contents = []
        for member in selected:
            image = source.extractfile(member)
            if image is None:
                raise ValueError("Selected archive member has no content.")
            with image:
                contents.append((PurePosixPath(member.name), image.read()))
    # Stage before publishing, so failed writes do not leave a partial dataset.
    workspace.mkdir(parents=True, exist_ok=True)
    images = []
    with tempfile.TemporaryDirectory(prefix="vision-subset-", dir=workspace) as folder:
        staged = Path(folder) / "images_dataSAT"
        staged.mkdir()
        for member_path, data in contents:
            relative = Path(*member_path.parts[1:])
            destination = staged / relative
            destination.parent.mkdir(exist_ok=True)
            destination.write_bytes(data)
            images.append({"file": relative.as_posix(), "sha256": hashlib.sha256(data).hexdigest()})
        staged.rename(dataset)
    manifest_path = workspace / "smoke-data.json"
    manifest = {
        "description": f"Uses the first {per_class} sorted regular images per class from the course archive; both frameworks receive the same bounded corpus. Execution check only.",
        "assets": [
            {
                "file": archive.name,
                "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "bytes": archive.stat().st_size,
            }
        ],
        "selected_images": images,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    """Prepare only the two comparison notebooks, before starting their runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=64)
    args = parser.parse_args()
    for stem in COMPARISONS:
        workspace = ROOT / ".lab-cache/06-computer-vision-capstone" / stem
        manifest = prepare(args.archive, workspace, args.per_class)
        print(f"{stem}: {len(manifest['selected_images'])} images")


if __name__ == "__main__":
    main()
