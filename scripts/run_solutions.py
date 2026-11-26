"""Run worked labs with explicitly recorded, bounded smoke-test adaptations."""

from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import importlib.metadata
import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import nbformat
from build_solutions import ROOT, python_source
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

# This is a verification profile, not the experiment used to claim model quality.
BOOTSTRAP = """import os, itertools, random
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
import numpy as np
np.random.seed(42)
random.seed(42)
import pandas as pd
_read_csv = pd.read_csv
def _bounded_csv(*args, **kwargs):
    data = _read_csv(*args, **kwargs)
    return data.sample(n=5000,random_state=42) if len(data)>5000 else data
pd.read_csv = _bounded_csv
from sklearn.model_selection import GridSearchCV
_grid_fit = GridSearchCV.fit
def _bounded_grid(self, *args, **kwargs):
    grids = self.param_grid if isinstance(self.param_grid,list) else [self.param_grid]
    self.param_grid = [{k:list(v)[:1] for k,v in grid.items()} for grid in grids[:1]]
    self.cv = 2
    self.n_jobs = 1
    self.verbose = 0
    return _grid_fit(self,*args,**kwargs)
GridSearchCV.fit = _bounded_grid
try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    torch.manual_seed(42)
    torch.set_num_threads(1)
    _loader_iter = torch.utils.data.DataLoader.__iter__
    def _bounded_loader(self):
        return itertools.islice(_loader_iter(self),12)
    torch.utils.data.DataLoader.__iter__ = _bounded_loader
"""


TENSORFLOW_BOOTSTRAP = """import tensorflow as tf
tf.random.set_seed(42)
for _dataset in [tf.keras.datasets.mnist,tf.keras.datasets.fashion_mnist,tf.keras.datasets.cifar10]:
    _original_load = _dataset.load_data
    def _small_load(*args,_load=_original_load,**kwargs):
        (x,y),(xt,yt) = _load(*args,**kwargs)
        return (x[:512],y[:512]),(xt[:256],yt[:256])
    _dataset.load_data = _small_load
_keras_fit = tf.keras.Model.fit
def _small_fit(self,*args,**kwargs):
    args = list(args)
    def _limit(value):
        if isinstance(value,(np.ndarray,pd.DataFrame,pd.Series,tf.Tensor,tf.RaggedTensor)):return value[:512]
        if isinstance(value,(tuple,list)) and value and all(isinstance(x,(np.ndarray,tf.Tensor,tf.RaggedTensor)) for x in value):return type(value)(x[:512] for x in value)
        if isinstance(value,tf.data.Dataset):return value.take(8)
        return value
    for i in range(min(2,len(args))):args[i]=_limit(args[i])
    for key in ['x','y','validation_data']:
        if key in kwargs:kwargs[key]=_limit(kwargs[key])
    kwargs['epochs']=min(2,kwargs.get('epochs',2))
    data=args[0] if args else kwargs.get('x')
    if hasattr(data,'class_indices'):
        kwargs['steps_per_epoch']=min(2,len(data))
        if hasattr(kwargs.get('validation_data'),'class_indices'):
            kwargs['validation_steps']=min(2,len(kwargs['validation_data']))
    return _keras_fit(self,*args,**kwargs)
tf.keras.Model.fit = _small_fit
"""


BUDGET_NAMES = {
    "epochs",
    "nepochs",
    "numepochs",
    "numberofepochs",
    "episodes",
    "numepisodes",
    "numtrainepochs",
}


def is_budget(name: str) -> bool:
    """Identify training budgets without confusing them with architecture dimensions."""
    return name.lower().replace("_", "") in BUDGET_NAMES


def training_signatures(sources: list[str]) -> dict[str, list[str]]:
    """Find notebook-local training function arguments, including positional budgets."""
    signatures = {}
    for source in sources:
        tree = ast.parse(python_source(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and any(
                word in node.name.lower() for word in ("train", "fit")
            ):
                signatures[node.name] = [arg.arg for arg in node.args.posonlyargs + node.args.args]
    return signatures


def bound_training(source: str, signatures: dict[str, list[str]] | None = None) -> str:
    """Limit executable budgets while preserving strings, model dimensions and other loops."""
    tree = ast.parse(python_source(source))
    signatures = signatures if signatures is not None else training_signatures([source])

    def capped(node: ast.expr) -> ast.expr:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, int | float) and not isinstance(node.value, bool):
                return ast.copy_location(ast.Constant(min(node.value, 2)), node)
            return node
        return ast.copy_location(
            ast.Call(ast.Name("min", ast.Load()), [ast.Constant(2), node], []), node
        )

    class BudgetTransformer(ast.NodeTransformer):
        def visit_Assign(self, node: ast.Assign) -> ast.Assign:
            self.generic_visit(node)
            if any(
                isinstance(target, ast.Name) and is_budget(target.id) for target in node.targets
            ):
                if isinstance(node.value, ast.Constant):
                    node.value = capped(node.value)
            return node

        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
            self.generic_visit(node)
            args = node.args.posonlyargs + node.args.args
            for index, arg in enumerate(args[len(args) - len(node.args.defaults) :]):
                if is_budget(arg.arg):
                    node.args.defaults[index] = capped(node.args.defaults[index])
            return node

        def visit_Call(self, node: ast.Call) -> ast.Call:
            self.generic_visit(node)
            for keyword in node.keywords:
                if keyword.arg and is_budget(keyword.arg):
                    keyword.value = capped(keyword.value)
            if isinstance(node.func, ast.Name) and node.func.id in signatures:
                for index, parameter in enumerate(signatures[node.func.id][: len(node.args)]):
                    if is_budget(parameter) or parameter.lower() in {
                        "iter",
                        "iterations",
                        "n_iter",
                        "steps",
                    }:
                        node.args[index] = capped(node.args[index])
            return node

        def visit_For(self, node: ast.For) -> ast.For:
            self.generic_visit(node)
            range_loop = (
                isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Name)
                and node.iter.func.id == "range"
            )
            if isinstance(node.target, ast.Name) and (
                node.target.id.lower() in {"epoch", "episode"}
                or (node.target.id == "e" and range_loop)
            ):
                node.iter = ast.Call(
                    ast.Attribute(ast.Name("itertools", ast.Load()), "islice", ast.Load()),
                    [node.iter, ast.Constant(2)],
                    [],
                )
            return node

    return ast.unparse(ast.fix_missing_locations(BudgetTransformer().visit(tree)))


def execution_code_digest(notebook: nbformat.NotebookNode) -> str:
    """Identify the actual verification code, including bootstrap and transformed cells."""
    sources = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    return hashlib.sha256(json.dumps(sources, ensure_ascii=False).encode("utf-8")).hexdigest()


def verified_previous(
    record: dict, source_hash: str, profile_hash: str, data_profile_hash: str | None = None
) -> bool:
    """Reuse evidence only when source, verification code and saved output all match."""
    artifact = ROOT / record.get("executed_notebook", "")
    return (
        record.get("status") == "smoke-passed"
        and record.get("solution_sha256") == source_hash
        and record.get("execution_code_sha256") == profile_hash
        and record.get("data_profile_sha256") == data_profile_hash
        and artifact.is_file()
        and hashlib.sha256(artifact.read_bytes()).hexdigest() == record.get("executed_sha256")
    )


def redact_paths(value):
    """Remove local host paths from saved output without changing executable code."""
    if isinstance(value, str):
        return value.replace(str(ROOT), "<repository>").replace(str(Path.home()), "<user-home>")
    if isinstance(value, list):
        return [redact_paths(item) for item in value]
    if isinstance(value, dict):
        return {key: redact_paths(item) for key, item in value.items()}
    return value


def main() -> None:
    """Run selected labs; record failures rather than claiming unexecuted work passes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--course", type=int, required=True, action="append")
    parser.add_argument("--match", default="*")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Exclude a notebook stem pattern (repeatable).",
    )
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument(
        "--skip-passed",
        action="store_true",
        help="Skip successful runs of the current worked source.",
    )
    args = parser.parse_args()
    report_path = ROOT / "reports/solution-execution.json"
    previous = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.exists()
        else {"results": []}
    )
    results = {row["solution"]: row for row in previous["results"]}
    overrides = json.loads((ROOT / "provenance/smoke-overrides.json").read_text(encoding="utf-8"))
    paths = sorted(
        path
        for course in args.course
        for path in (ROOT / "solutions").glob(f"{course:02}-*/notebooks/{args.match}.ipynb")
    )
    if not paths:
        parser.error("No matching worked notebooks.")
    failures = 0
    for path in paths:
        if any(fnmatch.fnmatchcase(path.stem, pattern) for pattern in args.exclude):
            continue
        relative = path.relative_to(ROOT).as_posix()
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        workspace = ROOT / ".lab-cache" / path.parent.parent.name / path.stem
        workspace.mkdir(parents=True, exist_ok=True)
        data_manifest = workspace / "smoke-data.json"
        data_profile = (
            json.loads(data_manifest.read_text(encoding="utf-8"))
            if data_manifest.exists()
            else None
        )
        data_profile_hash = (
            hashlib.sha256(data_manifest.read_bytes()).hexdigest()
            if data_manifest.exists()
            else None
        )
        notebook = nbformat.read(path, as_version=4)
        applied_overrides = []
        for override in overrides:
            if override["solution"] == relative:
                cell = notebook.cells[override["cell"] - 1]
                if (
                    hashlib.sha256(cell.source.encode("utf-8")).hexdigest()
                    != override["source_sha256"]
                ):
                    raise ValueError(
                        f"Smoke override no longer matches: {relative}:{override['cell']}"
                    )
                cell.source = override["source"]
                applied_overrides.append({"cell": override["cell"], "reason": override["reason"]})
        signatures = training_signatures(
            [cell.source for cell in notebook.cells if cell.cell_type == "code"]
        )
        needs_tensorflow = any(
            "tensorflow" in cell.source or "import keras" in cell.source
            for cell in notebook.cells
            if cell.cell_type == "code"
        )
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.source = bound_training(cell.source, signatures)
        notebook.cells.insert(
            1,
            nbformat.v4.new_code_cell(
                BOOTSTRAP + (TENSORFLOW_BOOTSTRAP if needs_tensorflow else "")
            ),
        )
        profile_hash = execution_code_digest(notebook)
        if args.skip_passed and verified_previous(
            results.get(relative, {}), source_hash, profile_hash, data_profile_hash
        ):
            continue
        notebook.cells[0].source += (
            "\n\n**BOUNDED SMOKE RUN:** seed 42; CSVs sampled to at most 5,000 rows; "
            "one grid setting with two folds; training budgets capped; DataLoaders limited "
            "to 12 batches. These outputs are execution checks, not full-run model evaluations."
        )
        if needs_tensorflow:
            notebook.cells[
                0
            ].source += " Bundled TensorFlow datasets use 512 training / 256 test examples; fit inputs use at most 512 examples or eight dataset batches."
        for override in applied_overrides:
            notebook.cells[0].source += " " + override["reason"]
        if data_profile:
            notebook.cells[0].source += " Dataset override: " + data_profile["description"]
        record = {
            "solution": relative,
            "solution_sha256": source_hash,
            "profile": "bounded-smoke",
            "executed_at_utc": datetime.now(UTC).isoformat(),
            "status": "failed",
            "python": sys.version.split()[0],
            "execution_code_sha256": profile_hash,
            "packages": {
                dist.metadata["Name"]: dist.version
                for dist in importlib.metadata.distributions()
                if dist.metadata.get("Name")
            },
            "output_paths_redacted": True,
            "data_profile": data_profile,
            "data_profile_sha256": data_profile_hash,
            "smoke_overrides": applied_overrides,
        }
        print(f"RUNNING {path.name}", flush=True)
        try:
            with tempfile.TemporaryDirectory(prefix="ibm-lab-kernel-") as folder:
                temp = Path(folder)
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
                            "display_name": "Lab verification",
                            "language": "python",
                        }
                    ),
                    encoding="utf-8",
                )
                manager = AsyncKernelManager(
                    kernel_name="python3",
                    kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(temp / "kernels")]),
                )
                client = NotebookClient(
                    notebook,
                    km=manager,
                    timeout=args.timeout,
                    allow_errors=False,
                    record_timing=False,
                    resources={"metadata": {"path": str(workspace)}},
                )
                client.execute(
                    cleanup_kc=True,
                    env={
                        **os.environ,
                        "MPLCONFIGDIR": str(temp / "matplotlib"),
                        "MPLBACKEND": "module://matplotlib_inline.backend_inline",
                    },
                )
            destination = ROOT / "reports/executed-labs" / path.relative_to(ROOT / "solutions")
            destination.parent.mkdir(parents=True, exist_ok=True)
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    cell.outputs = [
                        nbformat.from_dict(redact_paths(output)) for output in cell.outputs
                    ]
            nbformat.write(notebook, destination)
            record.update(
                {
                    "status": "smoke-passed",
                    "executed_notebook": destination.relative_to(ROOT).as_posix(),
                    "executed_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                }
            )
        except Exception as error:
            failures += 1
            message = redact_paths(str(error))
            record["error_type"] = type(error).__name__
            record["error"] = message[-5000:]
            nbformat.write(notebook, workspace / "failed-execution.ipynb")
        results[relative] = record
        temporary_report = report_path.with_suffix(".json.tmp")
        temporary_report.write_text(
            json.dumps(
                {"profile": "bounded-smoke", "results": list(results.values())},
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary_report.replace(report_path)
        print(
            f"{record['status'].upper()} {path.name}"
            + (f" ({record['error_type']})" if record.get("error_type") else ""),
            flush=True,
        )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
