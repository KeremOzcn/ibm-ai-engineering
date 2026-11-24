"""Create curated course copies from the untouched download archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
TITLE_OVERRIDES = {
    "Decision_trees.ipynb": "Decision Trees",
    "K-Means-Customer-Seg-v1.ipynb": "K-Means Customer Segmentation",
    "KNN_Classification.ipynb": "K-Nearest Neighbors Classification",
    "Mulitple-Linear-Regression-v1.ipynb": "Multiple Linear Regression",
    "Mulitple-Linear-Regression.ipynb": "Multiple Linear Regression",
    "Regression-Trees-Taxi-Tip-v1.ipynb": "Regression Trees for Taxi Tip Prediction",
    "Regression_Trees_Taxi_Tip.ipynb": "Regression Trees for Taxi Tip Prediction",
    "decision_tree_svm_ccFraud.ipynb": "Credit Card Fraud Detection with Decision Trees and SVM",
    "DL0101EN-2-1-Backpropagation-py-v1.jupyterlite.ipynb": "Backpropagation",
    "DL0101EN-3-1-Regression-with-Keras-py-v1_0__2_.ipynb": "Regression Models with Keras",
    "DL0101EN-4-1-Transformers-with-Keras-py-v1.ipynb": "Transformers with Keras",
    "Final_Project_Classification_and_Captioning.ipynb": "Aircraft Damage Classification and Captioning",
    "Final_Proj-Classify_Waste_Products_Using_TL_FT.ipynb": "Waste Classification Using Transfer Learning",
    "Learning_Objectives.ipynb": "Activation Functions and Max Pooling",
    "Objective.ipynb": "Linear Regression with Multiple Outputs",
    "Objective__d25d6030.ipynb": "Training Linear Regression with Multiple Outputs",
    "Deep_Neural_Network_for_Wine_Classification-v1.ipynb": "Deep Neural Network for Breast Cancer Classification",
    "Classifying_Document.ipynb": "Document Classification with Word Embeddings",
    "Gradio_Interface_Guide.md": "Gradio Interface Guide",
    "QA_Bot_Project_Guide.md": "QA Bot Project Guide",
}


def sha256(data: bytes) -> str:
    """Return a stable digest for source provenance."""
    return hashlib.sha256(data).hexdigest()


def content_digest(notebook: dict, skip_header: bool = False) -> str:
    """Hash cell types and source, excluding outputs and environment metadata."""
    cells = notebook["cells"][1:] if skip_header else notebook["cells"]
    content = [(cell["cell_type"], "".join(cell.get("source", []))) for cell in cells]
    return sha256(json.dumps(content, ensure_ascii=False).encode())


def slugify(title: str) -> str:
    """Convert a lab title into a readable, portable filename."""
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:110]


def main() -> None:
    """Import once, refusing to replace a previously curated collection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Directory with download_manifest.json")
    args = parser.parse_args()
    source = args.source.resolve()
    manifest_path = ROOT / "provenance/source-manifest.json"
    if (ROOT / "courses").exists() or manifest_path.exists():
        parser.error("The collection already exists; import into a fresh checkout only.")
    manifest = json.loads((source / "download_manifest.json").read_text(encoding="utf-8"))
    courses = {
        row["number"]: row
        for row in json.loads((ROOT / "provenance/courses.json").read_text(encoding="utf-8"))
    }

    # Verify the whole input before creating any course files.
    for row in manifest:
        path = (source / row["file"]).resolve()
        if not path.is_relative_to(source) or not path.is_file():
            raise ValueError(f"Invalid source path: {row['file']}")
        if sha256(path.read_bytes()) != row["sha256"]:
            raise ValueError(f"Source hash mismatch: {row['file']}")

    canonical: dict[tuple[int, str], dict] = {}
    targets: set[str] = set()
    imported: list[dict] = []
    with tempfile.TemporaryDirectory(prefix=".import-", dir=ROOT) as directory:
        staging = Path(directory)
        for row in sorted(manifest, key=lambda item: (item["course"], item["file"])):
            path = source / row["file"]
            course = courses[row["course"]]
            title = TITLE_OVERRIDES.get(path.name, row.get("title", path.stem))
            title = re.sub(r"[*_]+", "", title).strip()
            is_notebook = path.suffix == ".ipynb"
            notebook = json.loads(path.read_text(encoding="utf-8")) if is_notebook else None
            semantic_hash = content_digest(notebook) if notebook else row["sha256"]
            key = (row["course"], semantic_hash)
            if key in canonical:
                imported.append(
                    {
                        **row,
                        "title": title,
                        "target": canonical[key]["target"],
                        "semantic_sha256": semantic_hash,
                        "duplicate": True,
                    }
                )
                continue
            folder = "notebooks" if is_notebook else "guides"
            stem = slugify(title)
            target = f"courses/{course['slug']}/{folder}/{stem}{path.suffix}"
            variant = 2
            while target in targets:
                target = (
                    f"courses/{course['slug']}/{folder}/{stem}-variant-{variant:02}{path.suffix}"
                )
                variant += 1
            targets.add(target)
            output = staging / target
            output.parent.mkdir(parents=True, exist_ok=True)
            changes = ["descriptive filename"]
            if notebook is not None:
                errors = sum(
                    out.get("output_type") == "error"
                    for cell in notebook["cells"]
                    for out in cell.get("outputs", [])
                )
                executed = sum(
                    cell.get("execution_count") is not None
                    for cell in notebook["cells"]
                    if cell["cell_type"] == "code"
                )
                header = nbformat.v4.new_markdown_cell(
                    f"# {title}\n\n"
                    "**Course reference / exercise notebook — not locally execution-verified.**\n\n"
                    f"Part of course {course['number']:02}: {course['title']}. "
                    "Original IBM / Skills Network content and notices are retained below. "
                    "Exercise blanks are intentionally preserved; saved outputs were cleared during curation.\n\n"
                    "[Course study notes](../README.md) · "
                    "[Lab catalog](../../../docs/LAB_CATALOG.md) · "
                    "[Environment setup](../../../docs/SETUP.md)"
                )
                cleaned_cells = [header]
                for index, cell in enumerate(notebook["cells"], 1):
                    cell = nbformat.from_dict(cell)
                    text = "".join(cell.get("source", []))
                    cell["source"] = text
                    if cell["cell_type"] == "code" and re.fullmatch(
                        r"\s*Copyright © IBM Corporation\. All rights reserved\.\s*", text
                    ):
                        cell = nbformat.v4.new_markdown_cell(text)
                        changes.append(
                            f"source cell {index}: copyright text changed from code to markdown"
                        )
                    elif cell["cell_type"] == "code":
                        cell["outputs"] = []
                        cell["execution_count"] = None
                    else:
                        cell.pop("outputs", None)
                        cell.pop("execution_count", None)
                    cell["metadata"] = (
                        {"tags": cell["metadata"]["tags"]}
                        if cell.get("metadata", {}).get("tags")
                        else {}
                    )
                    cleaned_cells.append(cell)
                for index, cell in enumerate(cleaned_cells):
                    cell["id"] = sha256(f"{target}:{index}".encode())[:12]
                cleaned = nbformat.v4.new_notebook(
                    cells=cleaned_cells,
                    metadata={
                        "kernelspec": {
                            "display_name": "Python 3",
                            "language": "python",
                            "name": "python3",
                        },
                        "language_info": {"name": "python"},
                        "collection": {
                            "title": title,
                            "course": row["course"],
                            "status": "reference",
                            "source_semantic_sha256": semantic_hash,
                        },
                    },
                )
                nbformat.validate(cleaned)
                nbformat.write(cleaned, output)
                changes += [
                    "added status and navigation",
                    "normalized notebook schema and cell IDs",
                    "cleared outputs, execution counts and transient metadata",
                ]
                details = {
                    "original_error_outputs": errors,
                    "original_executed_cells": executed,
                    "curated_content_sha256": content_digest(cleaned, skip_header=True),
                }
            else:
                output.write_bytes(path.read_bytes())
                details = {"curated_content_sha256": sha256(path.read_bytes())}
            entry = {
                **row,
                "title": title,
                "target": target,
                "semantic_sha256": semantic_hash,
                "duplicate": False,
                "changes": changes,
                **details,
            }
            canonical[key] = entry
            imported.append(entry)

        # Local download paths belong only in the untouched private source archive.
        public_fields = {
            "course",
            "title",
            "file",
            "sha256",
            "target",
            "semantic_sha256",
            "duplicate",
            "changes",
            "note",
            "original_error_outputs",
            "original_executed_cells",
            "curated_content_sha256",
        }
        sanitized = [{k: v for k, v in row.items() if k in public_fields} for row in imported]
        staged_manifest = staging / "source-manifest.json"
        staged_manifest.write_text(
            json.dumps(sanitized, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        staged_courses = staging / "courses"
        published_courses = ROOT / "courses"
        staged_courses.rename(published_courses)
        try:
            staged_manifest.replace(manifest_path)
        except BaseException:
            published_courses.rename(staged_courses)
            raise
    print(f"Imported {len(imported)} source assets into {len(targets)} curated files.")


if __name__ == "__main__":
    main()
