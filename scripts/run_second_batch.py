"""Run a second-batch standalone notebook in a fresh local kernel with a bounded profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from datetime import UTC, datetime

import nbformat
from build_second_batch import ROOT, SPECS
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient
from run_solutions import redact_paths


def run(slug: str) -> None:
    """Execute unchanged source cells and bind fresh exported JSON evidence to their inputs."""
    source = ROOT / "kaggle" / slug / "notebook.ipynb"
    notebook = nbformat.read(source, as_version=4)
    started = datetime.now(UTC)
    print("FRESH_LOCAL_KERNEL", slug, flush=True)
    with tempfile.TemporaryDirectory(prefix="second-batch-kernel-") as folder:
        from pathlib import Path

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
            timeout=600,
            allow_errors=False,
            record_timing=False,
            resources={"metadata": {"path": str(ROOT)}},
        )
        client.execute(
            cleanup_kc=True,
            env={
                **os.environ,
                "SECOND_BATCH_FULL": "0",
                "OMP_NUM_THREADS": "1",
                "MPLCONFIGDIR": str(temp / "matplotlib"),
                "MPLBACKEND": "module://matplotlib_inline.backend_inline",
            },
        )
    exported = ROOT / ".lab-cache/second-batch-notebooks" / slug / "exports" / slug
    metrics = json.loads((exported / "metrics.json").read_text())
    if datetime.fromisoformat(metrics["runtime"]["utc"]) < started:
        raise ValueError("Stale local notebook output.")
    expected = notebook.metadata.release_source
    if (
        metrics["source"]["project_sha256"] != expected["project_sha256"]
        or metrics["source"]["common_sha256"] != expected["common_sha256"]
    ):
        raise ValueError("Executed source identity differs.")
    dest = ROOT / "reports/second-batch-notebooks" / slug
    dest.mkdir(parents=True, exist_ok=True)
    for artifact in exported.glob("*"):
        if artifact.suffix in {".json", ".png"}:
            shutil.copyfile(artifact, dest / artifact.name)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = [nbformat.from_dict(redact_paths(o)) for o in cell.outputs]
            for item in cell.outputs:
                if item.output_type == "stream" and item.name == "stderr":
                    item.text = re.sub(
                        r"/(?:Users|home)/[a-zA-Z0-9_-]+/", "<host-path>/", item.text
                    )
    nbformat.write(notebook, dest / "executed.ipynb")
    files = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in dest.iterdir()
        if p.name != "evidence.json"
    }
    (dest / "evidence.json").write_text(
        json.dumps(
            {
                "project": slug,
                "fresh_kernel": True,
                "profile": metrics["profile"],
                "source_notebook_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "files": files,
            },
            indent=2,
        )
        + "\n"
    )
    print("LOCAL_NOTEBOOK_PASSED", slug, metrics["profile"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", choices=SPECS, required=True)
    run(parser.parse_args().project)
