"""Generate standalone review/Kaggle notebooks from the selected project modules."""

from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = {
    "satellite": {
        "title": "Satellite classification with a reserved source scene",
        "input": "data_root",
        "find": "images_dataSAT",
        "local": ".lab-cache/06-computer-vision-capstone/lab-vision-transformers-using-pytorch/images_dataSAT",
        "scope": "A fresh CNN versus a 16-patch spatial CNN-Transformer. The small pilot tests the pipeline; it is not a model ranking or generalization benchmark.",
    },
    "instruction_tuning": {
        "title": "Fresh base-versus-LoRA instruction tuning",
        "input": "data_file",
        "find": "CodeAlpaca-20k.json",
        "local": ".lab-cache/11-advanced-fine-tuning/instruction-tuning-with-llms/CodeAlpaca-20k.json",
        "scope": "Two real optimizer steps by default, response-only loss, group-separated prompts and a fresh selected-adapter reload. No supplied predictions are used.",
    },
    "aircraft": {
        "title": "Aircraft classification and caption reliability",
        "input": "data_root",
        "find": "aircraft_damage_dataset_v1",
        "local": ".lab-cache/02-deep-learning-with-keras/aircraft-damage-classification-and-captioning/aircraft_damage_dataset_v1",
        "scope": "Correct VGG16 preprocessing and complete evaluation batches; separate pretrained BLIP descriptions. Caption-quality review remains manual and unmeasured.",
    },
}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def definitions(source: str) -> str:
    """Remove module imports of embedded helpers and the CLI invocation only."""
    lines = source.splitlines()
    removed = set()
    for node in ast.parse(source).body:
        is_import = isinstance(node, ast.ImportFrom) and node.module in {
            "__future__",
            "projects.common",
        }
        is_entry = isinstance(node, ast.If) and ast.unparse(node.test) == "__name__ == '__main__'"
        if is_import or is_entry:
            removed.update(range(node.lineno - 1, node.end_lineno))
    return "\n".join(line for index, line in enumerate(lines) if index not in removed)


def build(*, check: bool = False) -> None:
    common = (ROOT / "projects/common.py").read_text()
    for slug, spec in PROJECTS.items():
        source = (ROOT / f"projects/{slug}/project.py").read_text()
        code = (
            common
            + f"\nCOMMON_SOURCE_SHA256 = {sha(common)!r}\nPROJECT_SOURCE_SHA256 = {sha(source)!r}\n\n"
            + definitions(source)
        )
        ast.parse(code)
        config = f"""# Keep this False for the small correctness pilot. Full training requires a separate run.
FULL_RUN = False
ALLOW_DOWNLOADS = False
SEED = 42

# Set this explicitly if multiple attached datasets match.
DATA_PATH = None
if DATA_PATH is None:
    if Path('/kaggle/input').is_dir():
        matches = sorted(Path('/kaggle/input').rglob({spec['find']!r}))
        if len(matches) != 1:
            raise ValueError('Set DATA_PATH to the intended attached dataset; expected one matching input.')
        DATA_PATH = matches[0]
    else:
        root = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p/'provenance/source-manifest.json').is_file()), None)
        if root is None:
            raise ValueError('Set DATA_PATH to the local data file/directory.')
        DATA_PATH = root / {spec['local']!r}
DATA_PATH = Path(DATA_PATH)
OUTPUT = (Path('/kaggle/working') if Path('/kaggle/working').is_dir() else root / '.lab-cache/selected-projects') / {slug!r}
"""
        # A manually supplied local input need not be inside this repository.
        config = config.replace(
            "OUTPUT = (", "root = locals().get('root') or Path.cwd()\nOUTPUT = ("
        )
        arguments = "full=FULL_RUN, seed=SEED"
        if slug != "satellite":
            arguments += ", allow_downloads=ALLOW_DOWNLOADS"
        execute = f"report = run(DATA_PATH, OUTPUT, {arguments})\nprint('Profile:', report['profile'])\nprint('Actual evaluated split sizes:', report['selected_split_counts'])"
        if slug == "satellite":
            summary = "for name, result in report['models'].items():\n    print(name, {key: result['test'][key] for key in ['samples', 'accuracy', 'macro_f1']})"
        elif slug == "instruction_tuning":
            summary = "print('Fresh adapter reload:', report['reload_verified'])\nprint('Response losses:', report['base_response_loss'], report['adapter_response_loss'])\nfor row in report['predictions']:\n    print({key: row[key] for key in ['instruction', 'reference', 'base', 'adapter']})"
        else:
            summary = "print('Classifier:', report['test'])\nprint('Caption quality:', report['caption_quality'])\nfor row in report['captions']:\n    print(row)"
        notebook = nbformat.v4.new_notebook(
            cells=[
                nbformat.v4.new_markdown_cell(
                    f"# {spec['title']}\n\n{spec['scope']}\n\nNew project code prepared with AI assistance from concepts studied in the [IBM AI Engineering program](https://www.coursera.org/professional-certificates/ai-engineer). This is a separate experiment, not an IBM assessment submission. Data/model ownership remains with the original providers; data and weights are not bundled.\n\nA local pilot is not Kaggle execution evidence or proof of a quality improvement."
                ),
                nbformat.v4.new_markdown_cell(
                    "## Configuration and dependencies\n\nAttach the intended data and cache model weights, or explicitly enable downloads with internet access. Use the project's documented dependency environment. All training defaults to the bounded pilot; set `FULL_RUN=True` only for a separately budgeted complete run. `Save & Run All` must pass in the chosen Kaggle environment before publication."
                ),
                nbformat.v4.new_code_cell(code),
                nbformat.v4.new_code_cell(config),
                nbformat.v4.new_code_cell(execute),
                nbformat.v4.new_code_cell(summary),
                nbformat.v4.new_markdown_cell(
                    "## Interpretation\n\nInspect sample counts, split provenance and reload evidence before reading model scores. Pilot scores are too small to rank models or claim robust improvement. Full training, near-duplicate/source overlap review and task-specific quality evaluation remain separate checks."
                ),
            ],
            metadata={
                "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                "selected_project": {
                    "slug": slug,
                    "project_sha256": sha(source),
                    "common_sha256": sha(common),
                },
            },
        )
        for index, cell in enumerate(notebook.cells):
            cell.id = f"{slug}-{index}"
            if cell.cell_type == "code":
                ast.parse(cell.source)
        text = nbformat.writes(notebook) + "\n"
        path = ROOT / f"projects/{slug}/notebook.ipynb"
        if check:
            if not path.exists() or path.read_text() != text:
                raise ValueError(f"Stale project notebook: {slug}")
        else:
            path.write_text(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    build(check=parser.parse_args().check)
    print("Three standalone project notebooks are current.")
