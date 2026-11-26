"""Execute one standalone selected-project notebook and export verified local evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import nbformat
from build_project_notebooks import PROJECTS, ROOT, build
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient
from run_solutions import redact_paths


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(slug: str) -> None:
    """Use a fresh interpreter-specific kernel and require newly produced metrics."""
    build(check=True)
    source = ROOT / f"projects/{slug}/notebook.ipynb"
    notebook = nbformat.read(source, as_version=4)
    started = datetime.now(UTC)
    print(f"RUNNING standalone {slug}", flush=True)
    with tempfile.TemporaryDirectory(prefix="selected-project-kernel-") as folder:
        temp = Path(folder)
        kernel = temp / "kernels/python3"
        kernel.mkdir(parents=True)
        (kernel / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                    "display_name": "Project verification",
                    "language": "python",
                }
            )
        )
        manager = AsyncKernelManager(
            kernel_name="python3",
            kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(temp / "kernels")]),
        )
        client = NotebookClient(
            notebook,
            km=manager,
            timeout=180,
            allow_errors=False,
            record_timing=False,
            resources={"metadata": {"path": str(ROOT)}},
        )
        client.execute(
            cleanup_kc=True,
            env={
                **os.environ,
                "OMP_NUM_THREADS": "1",
                "TF_NUM_INTEROP_THREADS": "1",
                "TF_NUM_INTRAOP_THREADS": "1",
                "MPLCONFIGDIR": str(temp / "matplotlib"),
                "MPLBACKEND": "module://matplotlib_inline.backend_inline",
            },
        )
    cache = ROOT / f".lab-cache/selected-projects/{slug}"
    metrics = json.loads((cache / "metrics.json").read_text())
    if datetime.fromisoformat(metrics["runtime"]["utc"]) < started:
        raise ValueError("The notebook did not produce fresh metrics.")
    expected = notebook.metadata.selected_project
    if (
        metrics["source"]["project_sha256"] != expected["project_sha256"]
        or metrics["source"]["common_sha256"] != expected["common_sha256"]
    ):
        raise ValueError("Execution source identity does not match the generated notebook.")
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = [nbformat.from_dict(redact_paths(output)) for output in cell.outputs]
    destination = ROOT / f"reports/selected-projects/{slug}"
    destination.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, destination / "executed.ipynb")
    for name in ["metrics.json", "split-manifest.json"]:
        shutil.copyfile(cache / name, destination / name)
    evidence = {
        "project": slug,
        "status": "bounded-pilot-passed",
        "fresh_kernel": True,
        "source_notebook": source.relative_to(ROOT).as_posix(),
        "source_notebook_sha256": sha(source),
        "files": {
            name: sha(destination / name)
            for name in ["executed.ipynb", "metrics.json", "split-manifest.json"]
        },
    }
    (destination / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(
        f"PASSED standalone {slug}: fresh metrics, selected checkpoint reload, source hashes",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", choices=PROJECTS, required=True)
    run(parser.parse_args().project)
