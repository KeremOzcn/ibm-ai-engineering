"""Validate provenance, notebooks, generated files and collection navigation."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote

import nbformat
from build_catalog import ROOT, build_artifacts, load_inventory
from import_labs import content_digest

SECRET_PATTERNS = [
    re.compile(r"gh[pousr]_[A-Za-z0-9]{25,}"),
    re.compile(r"AKIA[A-Z0-9]{16}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY"),
]


def check_links(path: Path, text: str) -> None:
    """Check local file destinations in collection-authored navigation."""
    for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
        if target.startswith(("https://", "http://", "mailto:", "#")):
            continue
        target = unquote(target.split("#", 1)[0])
        destination = (path.parent / target).resolve()
        if not destination.is_relative_to(ROOT) or not destination.exists():
            raise ValueError(
                f"Broken or out-of-repository link in {path.relative_to(ROOT)}: {target}"
            )


def main() -> None:
    """Fail on lost assets, unrecorded lab edits or stale verification evidence."""
    courses, sources = load_inventory()
    if [course["number"] for course in courses] != list(range(1, 14)):
        raise ValueError("Expected courses 01 through 13 in order.")
    expected = {row["target"] for row in sources}
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "courses").rglob("*")
        if path.is_file() and path.name != "README.md"
    }
    if actual != expected:
        raise ValueError(
            f"Course assets differ: missing={expected - actual}, extra={actual - expected}"
        )
    canonical = {row["target"]: row for row in sources if not row["duplicate"]}
    for row in sources:
        if row["target"] not in canonical:
            raise ValueError(f"Source has no canonical target: {row['file']}")
        if (
            row["duplicate"]
            and row["semantic_sha256"] != canonical[row["target"]]["semantic_sha256"]
        ):
            raise ValueError(f"Invalid duplicate mapping: {row['file']}")
    for target, row in canonical.items():
        path = ROOT / target
        if path.suffix == ".ipynb":
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            digest = content_digest(notebook, skip_header=True)
            if notebook.metadata.collection.status != "reference":
                raise ValueError(f"Missing reference status: {target}")
            check_links(path, notebook.cells[0].source)
            if any(
                cell.get("outputs") or cell.get("execution_count") is not None
                for cell in notebook.cells
                if cell.cell_type == "code"
            ):
                raise ValueError(f"Course reference contains execution artifacts: {target}")
        else:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row["curated_content_sha256"]:
            raise ValueError(f"Course content changed without provenance update: {target}")
    for path, text in build_artifacts().items():
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            raise ValueError(f"Stale generated artifact: {path.relative_to(ROOT)}")
    authored_docs = [
        ROOT / "README.md",
        *sorted((ROOT / "docs").glob("*.md")),
        *sorted((ROOT / "courses").glob("*/README.md")),
        ROOT / "companions/README.md",
    ]
    for path in authored_docs:
        check_links(path, path.read_text(encoding="utf-8"))
    report = json.loads((ROOT / "reports/companion-execution.json").read_text(encoding="utf-8"))
    companions = sorted((ROOT / "companions").glob("*.ipynb"))
    reported = {row["notebook"]: row for row in report["results"]}
    if set(reported) != {path.relative_to(ROOT).as_posix() for path in companions}:
        raise ValueError("Companion report does not match the notebook inventory.")
    for path in companions:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        check_links(
            path, "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")
        )
        for cell in notebook.cells:
            if cell.cell_type == "code" and cell.source.strip():
                ast.parse(cell.source)
                if cell.execution_count is None or any(
                    out.output_type == "error" for out in cell.outputs
                ):
                    raise ValueError(f"Unexecuted or failed companion: {path.name}")
        row = reported[path.relative_to(ROOT).as_posix()]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Companion evidence is stale: {path.name}")
    scanned = [
        *authored_docs,
        *sorted((ROOT / "provenance").glob("*.json")),
        *sorted((ROOT / "reports").glob("*.json")),
        *sorted((ROOT / "courses").rglob("*.ipynb")),
        *companions,
        *sorted((ROOT / "scripts").glob("*.py")),
    ]
    for path in scanned:
        text = path.read_text(encoding="utf-8")
        if re.search(r"/(?:Users|home)/[a-zA-Z0-9_-]+/", text):
            raise ValueError(f"Personal filesystem path found: {path.relative_to(ROOT)}")
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            raise ValueError(f"Possible credential found: {path.relative_to(ROOT)}")
    print(
        f"PASS: {len(sources)} source mappings, {len(canonical)} course assets, "
        f"{len(companions)} executed companions, generated docs, links and credential patterns."
    )


if __name__ == "__main__":
    main()
