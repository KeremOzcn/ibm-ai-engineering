"""Verify selected-project notebook builds, split separation and saved pilot evidence."""

from __future__ import annotations

import hashlib
import json

import nbformat
from build_project_notebooks import PROJECTS, ROOT, build


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check() -> dict:
    """Check saved artifacts without importing model frameworks or running training."""
    build(check=True)
    for slug in PROJECTS:
        folder = ROOT / f"reports/selected-projects/{slug}"
        evidence = json.loads((folder / "evidence.json").read_text())
        source = ROOT / f"projects/{slug}/notebook.ipynb"
        if evidence["status"] != "bounded-pilot-passed" or evidence[
            "source_notebook_sha256"
        ] != digest(source):
            raise ValueError(f"Stale selected-project execution: {slug}")
        for filename in ["executed.ipynb", "metrics.json", "split-manifest.json"]:
            if evidence["files"][filename] != digest(folder / filename):
                raise ValueError(f"Changed project evidence: {slug}/{filename}")
        notebook = nbformat.read(folder / "executed.ipynb", as_version=4)
        nbformat.validate(notebook)
        reference = nbformat.read(source, as_version=4)
        if [(cell.cell_type, cell.source) for cell in notebook.cells] != [
            (cell.cell_type, cell.source) for cell in reference.cells
        ]:
            raise ValueError(f"Executed project code differs from its source notebook: {slug}")
        for cell in notebook.cells:
            if (
                cell.cell_type == "code"
                and cell.source.strip()
                and (
                    cell.execution_count is None
                    or any(o.output_type == "error" for o in cell.outputs)
                )
            ):
                raise ValueError(f"Incomplete selected-project execution: {slug}")
        metrics = json.loads((folder / "metrics.json").read_text())
        if metrics["project"] != slug or metrics["profile"] != "bounded-pilot":
            raise ValueError(f"Unexpected selected-project evidence profile: {slug}")
        if metrics["source"]["project_sha256"] != digest(
            ROOT / f"projects/{slug}/project.py"
        ) or metrics["source"]["common_sha256"] != digest(ROOT / "projects/common.py"):
            raise ValueError(f"Stale selected-project source: {slug}")
        if metrics["split_manifest_sha256"] != digest(folder / "split-manifest.json"):
            raise ValueError(f"Stale split manifest: {slug}")
        manifest = json.loads((folder / "split-manifest.json").read_text())
        if slug == "instruction_tuning":
            groups = {
                role: {row["group"] for row in manifest[role]}
                for role in ["train", "validation", "test"]
            }
            if (
                not metrics["reload_verified"]
                or not metrics["adapter_changed"]
                or metrics["optimizer_steps"] < 1
            ):
                raise ValueError("LoRA update/reload evidence missing.")
        else:
            groups = {
                role: {row["pixel_sha256"] for row in manifest["records"] if row["role"] == role}
                for role in ["train", "validation", "test"]
            }
            if slug == "satellite":
                scenes = {
                    role: {row["scene"] for row in manifest["records"] if row["role"] == role}
                    for role in groups
                }
                if any(
                    scenes[a] & scenes[b]
                    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]
                ):
                    raise ValueError("Satellite scenes overlap.")
                if not all(row["exact_reload"] for row in metrics["models"].values()):
                    raise ValueError("Satellite reload evidence missing.")
            elif not metrics["exact_reload"]:
                raise ValueError("Aircraft reload evidence missing.")
            if slug == "aircraft" and (folder / "visual-review.json").exists():
                review = json.loads((folder / "visual-review.json").read_text())
                files = {row["file"]: row["file_sha256"] for row in manifest["records"]}
                captions = {row["file"]: row["caption"] for row in metrics["captions"]}
                for row in review["reviews"]:
                    if row["file_sha256"] != files.get(row["file"]) or row[
                        "caption"
                    ] != captions.get(row["file"]):
                        raise ValueError("Aircraft visual review is stale.")
        if any(
            groups[a] & groups[b]
            for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]
        ):
            raise ValueError(f"Data groups leak across project splits: {slug}")
    return {
        "selected_projects": len(PROJECTS),
        "bounded_pilots": len(PROJECTS),
        "full_training_verified": False,
        "kaggle_execution_verified": False,
    }


if __name__ == "__main__":
    print(json.dumps(check()))
