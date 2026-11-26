"""Validate worked-copy provenance and saved bounded execution evidence."""

from __future__ import annotations

import hashlib
import json

import nbformat
from build_solutions import ROOT, build
from run_solutions import execution_code_digest


def check() -> dict:
    """Reject stale answers or evidence while allowing explicitly unexecuted labs."""
    report = build(check=True)
    if report["summary"]["syntax_or_stub_issues"]:
        raise ValueError("Worked notebooks contain unresolved code or exercise stubs.")
    expected = {row["solution"] for row in report["notebooks"]}
    actual = {path.relative_to(ROOT).as_posix() for path in (ROOT / "solutions").rglob("*.ipynb")}
    if actual != expected:
        raise ValueError("Worked-copy inventory differs from the course references.")
    for name in ["verification-plan.json", "smoke-overrides.json"]:
        rows = json.loads((ROOT / "provenance" / name).read_text(encoding="utf-8"))
        keys = [(row["solution"], row.get("cell")) for row in rows]
        if len(set(keys)) != len(keys) or any(row["solution"] not in expected for row in rows):
            raise ValueError(f"Unknown or duplicated entry in {name}.")
        if name == "smoke-overrides.json":
            for row in rows:
                notebook = nbformat.read(ROOT / row["solution"], as_version=4)
                cell = notebook.cells[row["cell"] - 1]
                if (
                    cell.cell_type != "code"
                    or hashlib.sha256(cell.source.encode("utf-8")).hexdigest()
                    != row["source_sha256"]
                ):
                    raise ValueError("Smoke override no longer matches its worked source.")
    ledger = json.loads((ROOT / "reports/solution-execution.json").read_text(encoding="utf-8"))
    seen = set()
    passed = 0
    current_passed = 0
    for row in ledger["results"]:
        if row["solution"] not in expected or row["solution"] in seen:
            raise ValueError("Unknown or duplicated notebook in the execution ledger.")
        seen.add(row["solution"])
        if row["status"] != "smoke-passed":
            continue
        artifact = (ROOT / row["executed_notebook"]).resolve()
        if not artifact.is_relative_to(ROOT / "reports/executed-labs"):
            raise ValueError("Execution artifact points outside the evidence directory.")
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != row["executed_sha256"]:
            raise ValueError(f"Executed notebook changed: {row['solution']}")
        notebook = nbformat.read(artifact, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code" and cell.source.strip():
                if cell.execution_count is None or any(
                    out.output_type == "error" for out in cell.outputs
                ):
                    raise ValueError(f"Missing or failed execution: {row['solution']}")
        if (
            row.get("execution_code_sha256")
            and execution_code_digest(notebook) != row["execution_code_sha256"]
        ):
            raise ValueError(f"Verification code hash differs: {row['solution']}")
        passed += 1
        if (
            hashlib.sha256((ROOT / row["solution"]).read_bytes()).hexdigest()
            == row["solution_sha256"]
        ):
            current_passed += 1
    return {**report["summary"], "saved_smoke_runs": passed, "current_smoke_runs": current_passed}


def main() -> None:
    """Print checked counts; no model training or service calls are performed."""
    print(json.dumps(check()))


if __name__ == "__main__":
    main()
