"""Build worked study copies while preserving all original course references."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import nbformat
from IPython.core.inputtransformer2 import TransformerManager

ROOT = Path(__file__).resolve().parents[1]
TRANSFORMER = TransformerManager()


def digest(text: str) -> str:
    """Hash source text for an exact patch guard."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def python_source(source: str) -> str:
    """Expose Python wrapped in timing/capture magics to the syntax checker."""
    while re.match(r"^%%(?:capture|time|timeit)\b", source.lstrip()):
        source = source.lstrip().partition("\n")[2]
    return TRANSFORMER.transform_cell(source)


def unresolved(source: str) -> list[str]:
    """Find syntax errors, executable ellipsis stubs and empty answer markers."""
    try:
        tree = ast.parse(python_source(source))
    except SyntaxError as error:
        return [f"{error.msg} (line {error.lineno})"]
    slice_ellipses = {
        id(child)
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        for child in ast.walk(node.slice)
        if isinstance(child, ast.Constant) and child.value is Ellipsis
    }
    issues = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and node.value is Ellipsis
            and id(node) not in slice_ellipses
        ):
            issues.append(f"Executable ellipsis at line {node.lineno}")
        if isinstance(node, ast.Name) and re.fullmatch(r"_{3,}", node.id):
            issues.append(f"Blank identifier at line {node.lineno}")
    if not tree.body and re.search(
        r"your code|enter code|add code|your answer|your answser|your response|REPLACE THIS|TODO",
        source,
        re.I,
    ):
        issues.append("Unanswered comment-only exercise")
    return issues


def normalize_setup(source: str) -> tuple[str, bool]:
    """Keep notebook setup from changing packages inside the verification kernel."""
    lines = source.splitlines()
    result = []
    continuation = False
    changed = False
    for line in lines:
        install = bool(
            re.match(
                r"\s*[!%](?:(?:pip|pip3|conda)\b|python(?:3)?\s+-m\s+spacy\s+download\b)", line
            )
        )
        if install or continuation:
            continuation = line.rstrip().endswith("\\")
            if install:
                result.append(
                    "# Install this lab's dependencies in its isolated environment before execution."
                )
            changed = True
        else:
            result.append(line)
    return "\n".join(result), changed


def build(*, check: bool = False) -> dict:
    """Apply guarded answers and generate an honest collection-wide progress report."""
    specifications = json.loads(
        (ROOT / "provenance/solution-patches.json").read_text(encoding="utf-8")
    )
    reviewed = json.loads((ROOT / "provenance/solution-reviewed.json").read_text(encoding="utf-8"))
    patches = {(row["reference"], row["cell"]): row for row in specifications}
    inspected = {(row["reference"], row["cell"]): row for row in reviewed}
    if len(patches) != len(specifications) or len(inspected) != len(reviewed):
        raise ValueError("Duplicate guarded answer or reviewed exercise location.")
    inventory = json.loads(
        (ROOT / "provenance/exercise-inventory.json").read_text(encoding="utf-8")
    )
    verification_plan = json.loads(
        (ROOT / "provenance/verification-plan.json").read_text(encoding="utf-8")
    )
    verification_notes = {row["solution"]: row for row in verification_plan}
    expected_answers = {(row["reference"], row["cell"]) for row in inventory}
    unreviewed = expected_answers - patches.keys() - inspected.keys()
    references = sorted((ROOT / "courses").glob("*/notebooks/*.ipynb"))
    locations = {
        (path.relative_to(ROOT).as_posix(), index)
        for path in references
        for index in range(1, len(nbformat.read(path, as_version=4).cells) + 1)
    }
    unknown = (patches.keys() | inspected.keys() | expected_answers) - locations
    if unknown:
        raise ValueError(f"Exercise locations do not exist: {sorted(unknown)}")

    def save(path: Path, text: str) -> None:
        if check:
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                raise ValueError(f"Stale generated solution artifact: {path.relative_to(ROOT)}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")

    records = []
    for path in references:
        reference = path.relative_to(ROOT).as_posix()
        target = ROOT / "solutions" / path.relative_to(ROOT / "courses")
        notebook = nbformat.read(path, as_version=4)
        edits = []
        issues = []
        setup_cells = []
        for index, cell in enumerate(notebook.cells, 1):
            spec = patches.get((reference, index))
            if (reference, index) in unreviewed:
                issues.append({"cell": index, "issue": "Exercise answer has not been reviewed"})
            if spec:
                if digest(cell.source) != spec["source_sha256"]:
                    raise ValueError(
                        f"Answer patch no longer matches its reference: {reference}:{index}"
                    )
                cell.source = spec["solution"]
                cell.cell_type = spec.get("cell_type", cell.cell_type)
                if cell.cell_type != "code":
                    cell.pop("execution_count", None)
                    cell.pop("outputs", None)
                cell.metadata["worked_answer"] = {
                    "origin": spec["origin"],
                    "explanation": spec["explanation"],
                }
                edits.append(
                    {"cell": index, "origin": spec["origin"], "explanation": spec["explanation"]}
                )
            if cell.cell_type == "code":
                cell.source = cell.source.replace("\u202f", " ").replace("\u00a0", " ")
                cell.source, changed = normalize_setup(cell.source)
                if changed:
                    setup_cells.append(index)
                cell.execution_count = None
                cell.outputs = []
                for issue in unresolved(cell.source):
                    issues.append({"cell": index, "issue": issue})
        title = notebook.metadata.collection.title
        notebook.cells[0].source = (
            f"# {title} — worked study copy\n\n"
            "Answers prepared for revision with AI assistance. IBM / Skills Network source notices "
            "and embedded instructor explanations are retained. This is not a course submission.\n\n"
            "Code completion and execution are tracked separately in the "
            "[solution progress report](../../../docs/SOLUTION_PROGRESS.md). "
            "Training may require a separate environment, external data, compute or provider credentials.\n\n"
            f"[Original reference](../../../{reference}) · "
            f"[Course notes](../../../courses/{path.parent.parent.name}/README.md)"
        )
        notebook.metadata.collection.update(
            {
                "status": "worked-solution" if not issues else "solution-in-progress",
                "reference": reference,
                "reference_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        nbformat.validate(notebook)
        save(target, nbformat.writes(notebook) + "\n")
        records.append(
            {
                "reference": reference,
                "solution": target.relative_to(ROOT).as_posix(),
                "course": notebook.metadata.collection.course,
                "title": title,
                "answer_cells": edits,
                "setup_cells_normalized": setup_cells,
                "reviewed_cells": [row for (ref, _), row in inspected.items() if ref == reference],
                "unresolved": issues,
                "code_status": "prepared" if not issues else "in-progress",
                "execution_status": "not-run",
                "solution_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        )
    execution_path = ROOT / "reports/solution-execution.json"
    executions = (
        json.loads(execution_path.read_text(encoding="utf-8"))["results"]
        if execution_path.exists()
        else []
    )
    by_solution = {row["solution"]: row for row in executions}
    for record in records:
        record["verification_note"] = verification_notes.get(record["solution"])
        execution = by_solution.get(record["solution"])
        if execution:
            record["execution_status"] = (
                execution["status"]
                if execution["solution_sha256"] == record["solution_sha256"]
                else "rerun-needed"
            )
    report = {
        "scope": "Worked study copies. Code preparation does not imply successful execution.",
        "summary": {
            "notebooks": len(records),
            "answer_cells": len(specifications),
            "syntax_or_stub_issues": sum(len(row["unresolved"]) for row in records),
        },
        "notebooks": records,
    }
    execution_counts = Counter(row["execution_status"] for row in records)
    report["summary"]["execution_counts"] = dict(sorted(execution_counts.items()))
    save(
        ROOT / "reports/solution-progress.json",
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    )
    counts = Counter(row["course"] for row in records)
    page = [
        "# Lab solution progress",
        "",
        report["scope"],
        "",
        "Worked answers live under `solutions/`; `courses/` remains the untouched reference collection. "
        "The owner selected bounded local verification, with full runs listed separately.",
        "",
        "Current execution state: "
        + ", ".join(f"**{count} {status}**" for status, count in sorted(execution_counts.items()))
        + ".",
        "",
        "[Remaining full runs and service requirements](FULL_RUNS.md). Successful bounded execution does not establish model quality.",
        "",
        "| Course | Notebooks | Answer/code cells | Unresolved code/stubs | Bounded passed |",
        "|---|---:|---:|---:|---:|",
    ]
    for number in sorted(counts):
        group = [row for row in records if row["course"] == number]
        page.append(
            f"| {number:02} | {len(group)} | {sum(len(row['answer_cells']) for row in group)} | "
            f"{sum(len(row['unresolved']) for row in group)} | {sum(row['execution_status'] == 'smoke-passed' for row in group)} |"
        )
    page += [
        "",
        "## Notebook status",
        "",
        "`prepared` means the code passes the syntax/stub audit. It is not a runtime result. "
        "Execution reports also distinguish bounded runs from full prescribed training.",
        "",
        "| Notebook | Code | Execution | Follow-up |",
        "|---|---|---|---|",
    ]
    for row in records:
        page.append(
            f"| [{row['title']}](../{row['solution']}) | {row['code_status']} | {row['execution_status']} | "
            f"{row['verification_note']['reason'] if row['verification_note'] else 'Full prescribed run; inspect the execution ledger for any bounded-run failure.'} |"
        )
    save(ROOT / "docs/SOLUTION_PROGRESS.md", "\n".join(page) + "\n")
    return report


def main() -> None:
    """Build the solution layer and summarize remaining work."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Verify generated files without modifying them."
    )
    args = parser.parse_args()
    report = build(check=args.check)
    print(json.dumps(report["summary"]))
    if report["summary"]["syntax_or_stub_issues"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
