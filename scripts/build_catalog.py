"""Build course study pages and a deterministic, source-aware lab audit."""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict
from pathlib import Path

import nbformat
from IPython.core.inputtransformer2 import TransformerManager

ROOT = Path(__file__).resolve().parents[1]
TRANSFORMER = TransformerManager()
EXERCISE_MARKER = re.compile(
    r"your code|enter code|add code|complete the|fill (?:in|the)|TODO|_{3,}|\.{3}", re.I
)


def load_inventory() -> tuple[list[dict], list[dict]]:
    """Read the collection definition and all source mappings."""
    return (
        json.loads((ROOT / "provenance/courses.json").read_text(encoding="utf-8")),
        json.loads((ROOT / "provenance/source-manifest.json").read_text(encoding="utf-8")),
    )


def audit_lab(path: Path) -> dict:
    """Inspect reference code without installing dependencies or executing cells."""
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    code = [
        (index, cell) for index, cell in enumerate(notebook.cells, 1) if cell.cell_type == "code"
    ]
    issues = []
    imports: set[str] = set()
    install_commands: list[str] = []
    exercise_cells = []
    for index, cell in code:
        source = cell.source
        if EXERCISE_MARKER.search(source):
            exercise_cells.append(index)
        imports.update(re.findall(r"^\s*(?:from|import) +([a-zA-Z_][a-zA-Z0-9_]*)", source, re.M))
        install_commands.extend(
            line.strip()
            for line in source.splitlines()
            if re.match(r"\s*[!%](?:pip|conda)\b", line)
        )
        try:
            ast.parse(TRANSFORMER.transform_cell(source))
        except (SyntaxError, IndentationError) as error:
            issues.append({"cell": index, "line": error.lineno, "message": error.msg})
    dependencies = sorted(imports)
    notes = []
    if "torchtext" in imports:
        notes.append("legacy torchtext environment")
    if any(name.startswith("langchain") for name in imports):
        notes.append("version-specific LangChain APIs")
    if any(name.startswith("ibm_watson") for name in imports):
        notes.append("watsonx / Skills Network service required")
    if "bitsandbytes" in imports:
        notes.append("accelerator-dependent quantization")
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "title": notebook.metadata.collection.title,
        "course": notebook.metadata.collection.course,
        "status": "reference-not-executed",
        "code_cells": len(code),
        "exercise_marker_cells": exercise_cells,
        "syntax_issues": issues,
        "imports": dependencies,
        "install_commands": install_commands,
        "environment_notes": notes,
    }


def bullets(items: list[str]) -> str:
    """Render one level of study-note bullets."""
    return "\n".join(f"- {item}" for item in items)


def build_artifacts() -> dict[Path, str]:
    """Return every generated artifact without changing the filesystem."""
    courses, sources = load_inventory()
    by_target: dict[str, list[dict]] = defaultdict(list)
    for row in sources:
        by_target[row["target"]].append(row)
    labs = [audit_lab(ROOT / target) for target in sorted(by_target) if target.endswith(".ipynb")]
    lab_map = {lab["path"]: lab for lab in labs}
    stats = {
        "source_assets": len(sources),
        "source_notebooks": sum(row["file"].endswith(".ipynb") for row in sources),
        "curated_notebooks": len(labs),
        "curated_guides": sum(target.endswith(".md") for target in by_target),
        "consolidated_duplicates": sum(row["duplicate"] for row in sources),
        "code_cells": sum(lab["code_cells"] for lab in labs),
        "syntax_issue_cells": sum(len(lab["syntax_issues"]) for lab in labs),
        "notebooks_with_syntax_issues": sum(bool(lab["syntax_issues"]) for lab in labs),
        "reference_notebooks_execution_verified": 0,
        "original_saved_error_outputs": sum(
            row.get("original_error_outputs", 0) for row in sources
        ),
    }
    artifacts = {
        ROOT / "reports/lab-audit.json": json.dumps(
            {"summary": stats, "labs": labs}, indent=2, ensure_ascii=False
        )
        + "\n"
    }
    catalog = [
        "# Lab catalog",
        "",
        f"{stats['source_notebooks']} collected notebooks map to {len(labs)} distinct course notebooks; "
        f"{stats['consolidated_duplicates']} within-course copies were consolidated. "
        "Two source guides are also included. Shared notebooks used by different courses remain in both course folders.",
        "",
        "All course notebooks are **reference / exercise copies**. They were validated as notebook files, "
        "but were not executed during curation. Exercise markers are a search heuristic, not a completion score. "
        "Syntax issues below identify preserved blanks after IPython input transformation; valid syntax does not prove runtime correctness.",
        "",
        "[Setup](SETUP.md) · [Study roadmap](STUDY_ROADMAP.md) · "
        "[Machine-readable audit](../reports/lab-audit.json) · [Source mapping](../provenance/source-manifest.json)",
        "",
    ]
    for course in courses:
        number = course["number"]
        course_sources = [row for row in sources if row["course"] == number]
        targets = sorted({row["target"] for row in course_sources})
        notebook_count = sum(target.endswith(".ipynb") for target in targets)
        guide_count = len(targets) - notebook_count
        by_original = {Path(row["file"]).name: row for row in course_sources}
        starters = []
        for original, reason in course["start_with"]:
            if original not in by_original:
                raise ValueError(f"Missing recommended lab in course {number}: {original}")
            entry = by_original[original]
            relative = Path(entry["target"]).relative_to(f"courses/{course['slug']}").as_posix()
            starters.append(f"- [{entry['title']}]({relative}) — {reason}.")
        rows = []
        for target in targets:
            entry = by_target[target][0]
            relative = Path(target).relative_to(f"courses/{course['slug']}").as_posix()
            lab = lab_map.get(target)
            status = (
                f"Reference; {len(lab['syntax_issues'])} syntax blanks"
                if lab and lab["syntax_issues"]
                else "Reference"
            )
            rows.append(f"| [{entry['title']}]({relative}) | {status if lab else 'Source guide'} |")
        page = f"""# {number:02} · {course['title']}

{course['summary']}

[Repository home](../../README.md) · [Full catalog](../../docs/LAB_CATALOG.md) · [Environment setup](../../docs/SETUP.md)

## Concepts to retain

{bullets(course['concepts'])}

## Working model

{course['formula']}

## Start with these labs

{chr(10).join(starters)}

## Revisit with a purpose

{bullets(course['revisit'])}

## Common mistakes

{bullets(course['pitfalls'])}

## Check your understanding

{bullets(course['questions'])}

## Collected material

{notebook_count} reference notebooks{f' and {guide_count} source guides' if guide_count else ''}. Course source, authorship and copyright notices remain in the collected files. No course notebook has been certified as runnable by this curation.

| Material | Status |
|---|---|
{chr(10).join(rows)}
"""
        artifacts[ROOT / f"courses/{course['slug']}/README.md"] = page
        catalog += [
            f"## {number:02} · {course['title']}",
            "",
            f"[Study notes](../courses/{course['slug']}/README.md)",
            "",
            "| Material | Code cells | Exercise marker cells | Syntax issue cells |",
            "|---|---:|---:|---:|",
        ]
        for target in targets:
            entry = by_target[target][0]
            lab = lab_map.get(target)
            if lab:
                catalog.append(
                    f"| [{entry['title']}](../{target}) | {lab['code_cells']} | "
                    f"{len(lab['exercise_marker_cells'])} | {len(lab['syntax_issues'])} |"
                )
            else:
                catalog.append(f"| [{entry['title']}](../{target}) | Guide | — | — |")
        catalog.append("")
    artifacts[ROOT / "docs/LAB_CATALOG.md"] = "\n".join(catalog).rstrip() + "\n"
    return artifacts


def main() -> None:
    """Regenerate documentation, or check that committed artifacts are current."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    stale = []
    for path, text in build_artifacts().items():
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                stale.append(path.relative_to(ROOT).as_posix())
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
    if stale:
        raise SystemExit("Stale generated files: " + ", ".join(stale))
    print(
        "Catalog and 13 course study pages are current."
        if args.check
        else "Generated catalog, audit and 13 course study pages."
    )


if __name__ == "__main__":
    main()
