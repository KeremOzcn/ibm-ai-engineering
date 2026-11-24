"""Execute the three local companions and record measured results."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Run each notebook in this Python environment, then save outputs and evidence."""
    paths = sorted((ROOT / "companions").glob("*.ipynb"))
    if len(paths) != 3:
        raise SystemExit("Expected exactly three companion notebooks.")
    results = []
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        with tempfile.TemporaryDirectory(prefix="ibm-companion-") as directory:
            temp = Path(directory)
            kernel = temp / "kernels/python3"
            kernel.mkdir(parents=True)
            (kernel / "kernel.json").write_text(
                json.dumps(
                    {
                        "argv": [
                            sys.executable,
                            "-m",
                            "ipykernel_launcher",
                            "-f",
                            "{connection_file}",
                        ],
                        "display_name": "Companion verification",
                        "language": "python",
                    }
                ),
                encoding="utf-8",
            )
            manager = KernelManager(
                kernel_name="python3",
                kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(temp / "kernels")]),
            )
            client = NotebookClient(
                notebook,
                km=manager,
                timeout=180,
                allow_errors=False,
                record_timing=False,
                resources={"metadata": {"path": str(temp)}},
            )
            client.execute(
                cleanup_kc=True,
                env={
                    **os.environ,
                    "MPLCONFIGDIR": str(temp / "matplotlib"),
                    "MPLBACKEND": "module://matplotlib_inline.backend_inline",
                },
            )
        metrics = []
        images = 0
        for cell in notebook.cells:
            for output in cell.get("outputs", []):
                images += "image/png" in output.get("data", {})
                for line in output.get("text", "").splitlines():
                    if line.startswith("COMPANION_METRICS="):
                        metrics.append(json.loads(line.removeprefix("COMPANION_METRICS=")))
        if len(metrics) != 1:
            raise ValueError(f"Expected one metrics record in {path.name}; got {len(metrics)}")
        nbformat.validate(notebook)
        nbformat.write(notebook, path)
        results.append(
            {
                "notebook": path.relative_to(ROOT).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "executed_code_cells": sum(
                    cell.cell_type == "code" and bool(cell.source.strip())
                    for cell in notebook.cells
                ),
                "error_outputs": 0,
                "embedded_png_outputs": images,
                "metrics": metrics[0],
            }
        )
        print(f"PASS {path.name}: {metrics[0]}", flush=True)
    report = {
        "executed_at_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in [
                "numpy",
                "scipy",
                "scikit-learn",
                "matplotlib",
                "torch",
                "nbformat",
                "nbclient",
                "ipykernel",
                "ipython",
            ]
        },
        "scope": "Three companion examples only; no course reference notebook was executed.",
        "results": results,
    }
    (ROOT / "reports/companion-execution.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
