"""Shared provenance and evaluation helpers for the selected portfolio projects."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import random
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path


def file_sha256(path: Path) -> str:
    """Hash an input/checkpoint without reading the entire file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict | list) -> None:
    """Publish a complete JSON artifact atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def runtime(packages: list[str]) -> dict:
    """Capture versions without collecting environment variables or host paths."""
    return {
        "utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.system(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
    }


def source_identity(project_file: str | None, embedded_digest: str | None = None) -> dict:
    """Bind module runs and generated notebooks to their authored source files."""
    project_digest = file_sha256(Path(project_file)) if project_file else embedded_digest
    common_digest = (
        file_sha256(Path(__file__))
        if "__file__" in globals()
        else globals().get("COMMON_SOURCE_SHA256")
    )
    if not project_digest or not common_digest:
        raise ValueError("Project source identity is missing.")
    return {"project_sha256": project_digest, "common_sha256": common_digest}


def index_images(
    root: Path, class_names: list[str], directories: dict[str, str] | None = None
) -> list[dict]:
    """Index decoded RGB content as well as bytes, retaining only relative filenames."""
    from PIL import Image, ImageOps

    rows = []
    directories = directories or {"unsplit": "."}
    for role, directory in directories.items():
        for label, name in enumerate(class_names):
            folder = root / directory / name
            if not folder.is_dir():
                raise ValueError(f"Missing image class directory: {directory}/{name}")
            for path in sorted(folder.rglob("*")):
                if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                with Image.open(path) as original:
                    image = ImageOps.exif_transpose(original).convert("RGB")
                    pixels = str(image.size).encode() + image.tobytes()
                rows.append(
                    {
                        "file": path.relative_to(root).as_posix(),
                        "label": label,
                        "role": role,
                        "file_sha256": file_sha256(path),
                        "pixel_sha256": hashlib.sha256(pixels).hexdigest(),
                    }
                )
    if not rows:
        raise ValueError("No images found.")
    return rows


def clean_image_splits(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Remove exact duplicates and conflicting labels, preserving held-out copies first."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["pixel_sha256"]].append(row)
    retained, removed = [], []
    priority = {"test": 0, "validation": 1, "train": 2}
    for group in groups.values():
        if len({row["label"] for row in group}) > 1:
            removed.extend(
                {"file": row["file"], "reason": "conflicting_pixel_labels"} for row in group
            )
            continue
        ordered = sorted(group, key=lambda row: (priority[row["role"]], row["file"]))
        retained.append(ordered[0])
        removed.extend(
            {"file": row["file"], "reason": "exact_pixel_duplicate"} for row in ordered[1:]
        )
    return sorted(retained, key=lambda row: row["file"]), removed


def balanced_subset(rows: list[dict], per_class: int | None, seed: int) -> list[dict]:
    """Select a reproducible pilot without changing the complete split manifest."""
    if per_class is None:
        return rows
    rng = random.Random(seed)
    result = []
    for label in sorted({row["label"] for row in rows}):
        group = sorted((row for row in rows if row["label"] == label), key=lambda row: row["file"])
        rng.shuffle(group)
        result.extend(group[:per_class])
    return sorted(result, key=lambda row: row["file"])


def split_counts(rows: list[dict]) -> dict:
    """Describe the actual records, including class support."""
    return {
        role: dict(Counter(str(row["label"]) for row in rows if row["role"] == role))
        for role in ["train", "validation", "test"]
    }


def classification_metrics(labels: list[int], probabilities: list[float]) -> dict:
    """Evaluate exactly the predictions supplied, never the size of an upstream dataset."""
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        roc_auc_score,
    )

    if not labels or len(labels) != len(probabilities):
        raise ValueError("Predictions and labels must have the same nonzero length.")
    probabilities = np.asarray(probabilities)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Expected finite probabilities in [0, 1].")
    predicted = (probabilities >= 0.5).astype(int)
    return {
        "samples": len(labels),
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_f1": float(
            f1_score(labels, predicted, labels=[0, 1], average="macro", zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if len(set(labels)) == 2 else None,
        "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
        "per_class": classification_report(
            labels, predicted, labels=[0, 1], output_dict=True, zero_division=0
        ),
    }
