"""Build the three independently authored notebooks for actual Kaggle execution."""

from __future__ import annotations

import argparse
import ast
import json

import nbformat
from build_project_notebooks import PROJECTS, ROOT, definitions, sha

SPECS = {
    "satellite": {
        "title": "Satellite CNN vs Transformer Across Source Scenes",
        "slug": "satellite-cnn-vs-transformer-across-source-scenes",
        "url": "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/4Z1fwRR295-1O3PMQBH6Dg/images-dataSAT.tar",
        "archive": "images-dataSAT.tar",
        "sha256": "5e45d64a33052c2a7f2e8b8d4232fbba6b7419831d192de56cae87dbe9547206",
        "question": "How do a small CNN and a spatial CNN-Transformer generalize to an unseen source scene?",
        "method": "Reserve T32UPE for testing and T31UDQ for validation. Train both fresh models for three epochs with shared stem initialization and identical image order/augmentation. Restore the validation-selected checkpoints. The hybrid retains 16 spatial tokens plus CLS; parameter counts differ, so this is not an attention-only ablation.",
        "sources": "Data: [IBM Skills Network archive](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/4Z1fwRR295-1O3PMQBH6Dg/images-dataSAT.tar). The original image filenames identify Sentinel-2 scenes. Dataset redistribution permission and broader geographic overlap are not established; data are fetched from the provider, not republished here.",
        "limits": "Only four source-scene groups and one seed; the validation group has 93 images. Exact decoded-image duplicates are checked, but geographic overlap and near-duplicates are unverified. Three epochs do not establish architecture superiority.",
    },
    "instruction_tuning": {
        "title": "LoRA Instruction Tuning and Verified Reload",
        "slug": "lora-instruction-tuning-and-verified-reload",
        "url": "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/WzOT_CwDALWedTtXjwH7bA/CodeAlpaca-20k.json",
        "archive": "CodeAlpaca-20k.json",
        "sha256": "10941b3bcf9fa1c2c47dd43dd9ec69a4ceec19094fd15def63e23f6c5c48d218",
        "question": "What changes after a bounded, genuine LoRA update when base and adapter use identical held-out prompts?",
        "method": "Group normalized instructions before the 80/10/10 split; retain nonempty no-context examples. Train OPT-350M LoRA on 512 eligible training records for one epoch (256 optimizer steps), with 64 validation and 64 test records, sequence length 256. Select by validation response NLL. Reload the adapter onto a fresh base at the identical immutable revision. Generate paired outputs with greedy decoding and a 64-token cap. Mask prompt and padding labels, preserving genuine EOS only.",
        "sources": "Data: [CodeAlpaca by Sahil Chaudhary](https://github.com/sahil280114/codealpaca), downloaded through the IBM course mirror. The upstream repository describes synthetic text-davinci-003 answers; its data/model usage conditions remain applicable. Base: [Meta OPT-350M](https://huggingface.co/facebook/opt-350m), with its research/noncommercial license. No base weights or dataset archive are bundled with this notebook.",
        "limits": "This is a 512-record training budget, not training on the entire corpus. Response NLL measures reference likelihood, not generated-code correctness. Outputs may be truncated; no task tests or human quality benchmark are claimed. One seed and synthetic answers limit conclusions; no adapter improvement is assumed.",
    },
    "aircraft": {
        "title": "Aircraft Damage and Caption Reliability",
        "slug": "aircraft-damage-and-caption-reliability",
        "url": "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ZjXM4RKxlBK9__ZjHBLl5A/aircraft-damage-dataset-v1.tar",
        "archive": "aircraft_damage_dataset_v1.tar",
        "sha256": "e8cc631b5eb6f373f6e1d47b44cdedd5fb2dd7e6ce7fc28877a1daa4a8f22bc6",
        "question": "Can a crack/dent classifier work while a general image captioner still describes the wrong object?",
        "method": "Use all 300 training, 96 validation and 50 test images after exact-pixel duplicate/conflict checks. Freeze ImageNet VGG16, use its expected BGR/channel-centering preprocessing and train a fresh head for ten epochs. Restore the validation-selected head and include remainder batches. Run BLIP separately on one predetermined example per test class; leave human-review fields empty.",
        "sources": "Data: [Aircraft Damage Detection by Youssef Donia](https://universe.roboflow.com/youssef-donia-fhktl/aircraft-damage-detection-1j9qk), CC BY 4.0, adapted as crack/dent folders by IBM Skills Network. Models: [Keras VGG16](https://keras.io/api/applications/vgg/) and [Salesforce BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base). No source dataset or pretrained weights are bundled.",
        "limits": "This is a small teaching dataset with unverified source-photo/near-duplicate overlap. Classification and captioning are separate models, with no damage localization or joint training. Two captions support a qualitative failure analysis, not a caption benchmark or an aircraft safety system.",
    },
}

BOOTSTRAP = """import os
import subprocess
import sys
os.environ["USE_TF"] = "0"  # Transformers uses PyTorch; the aircraft module imports TF separately.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.45.2", "peft==0.11.1", "accelerate>=0.30,<2"], check=True)
from pathlib import Path
import json
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
WORK = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()/".lab-cache/kaggle-local"
WORK.mkdir(parents=True, exist_ok=True)
SCRATCH = Path("/kaggle/temp/ibm-portfolio") if Path("/kaggle/working").is_dir() else WORK/"scratch"
SCRATCH.mkdir(parents=True, exist_ok=True)
"""


def build(owner: str = "kerem0zcan", *, public: bool = False) -> None:
    """Keep generated uploads separate from immutable course and pilot artifacts."""
    common = (ROOT / "projects/common.py").read_text()
    support = (ROOT / "scripts/kaggle_release_support.py").read_text()
    for slug, spec in SPECS.items():
        if not 5 <= len(spec["title"]) <= 50:
            raise ValueError("Kaggle notebook titles must have 5 to 50 characters.")
        source = (ROOT / f"projects/{slug}/project.py").read_text()
        code = (
            common
            + f"\nCOMMON_SOURCE_SHA256 = {sha(common)!r}\nPROJECT_SOURCE_SHA256 = {sha(source)!r}\n"
            + definitions(source)
        )
        local = PROJECTS[slug]["local"]
        data = f"""# Fetch from the provider; reject changed archives by SHA-256.
DATA_DIR = SCRATCH / "inputs" / {slug!r}
DATA_PATH = DATA_DIR / {PROJECTS[slug]['find']!r}
USING_LOCAL_CACHE = False
if not Path("/kaggle/working").is_dir():
    local_root = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p/"provenance/source-manifest.json").exists()), None)
    if local_root is not None:
        candidate = local_root / {local!r}
        if candidate.exists():
            DATA_PATH = candidate
            USING_LOCAL_CACHE = True
if not USING_LOCAL_CACHE:
    archive = verified_download({spec['url']!r}, DATA_DIR/{spec['archive']!r}, {spec['sha256']!r})
    if archive.suffix == ".tar":
        unpack_regular_tar(archive, DATA_DIR)
    else:
        DATA_PATH = archive
if not DATA_PATH.exists():
    raise FileNotFoundError("Expected dataset layout was not produced.")
OUTPUT = WORK / {slug!r}
OUTPUT.mkdir(parents=True, exist_ok=True)
RUN_OUTPUT = SCRATCH / "runs" / {slug!r}
"""
        extra = ", allow_downloads=True" if slug != "satellite" else ""
        if slug == "instruction_tuning":
            extra += ", epochs=1, train_limit=512, evaluation_limit=64"
        execute = f"report = run(DATA_PATH, RUN_OUTPUT, full=True, seed=42{extra})\nfor filename in ['metrics.json', 'split-manifest.json']:\n    shutil.copyfile(RUN_OUTPUT/filename, OUTPUT/filename)\nprint('Executed profile:', report['profile'])\nprint('Evaluated split sizes:', report['selected_split_counts'])\nprint('Runtime:', report['runtime'])"
        if slug == "satellite":
            result = """fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (name, result) in zip(axes, report["models"].items()):
    matrix = result["test"]["confusion_matrix"]
    ax.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2): ax.text(j, i, str(matrix[i][j]), ha="center", va="center")
    ax.set(title=f"{name}: macro F1 {result['test']['macro_f1']:.3f}", xlabel="Predicted", ylabel="True", xticks=[0,1], yticks=[0,1])
    print(name, "parameters", result["parameters"], "test samples", result["test"]["samples"], "reload", result["exact_reload"])
fig.tight_layout(); fig.savefig(OUTPUT/"results.png", dpi=160); plt.show()
print("Labels: 0 = non-agricultural, 1 = agricultural")
"""
        elif slug == "instruction_tuning":
            result = """base = report["base_response_loss"]["nll"]
adapter = report["adapter_response_loss"]["nll"]
fig, ax = plt.subplots(figsize=(7,4))
ax.bar(["Base", "Reloaded LoRA"], [base, adapter], color=["#60758a", "#158578"])
ax.set(ylabel="Held-out response NLL (lower is better)", title="Reference likelihood is not code correctness")
fig.tight_layout(); fig.savefig(OUTPUT/"results.png", dpi=160); plt.show()
print("Steps:", report["optimizer_steps"], "trainable fraction:", report["trainable_parameters"]/report["total_parameters"])
print("Fresh reload max logit difference:", report["reload_max_abs_logit_difference"])
print("Truncated responses:", report["truncated_response_counts"])
for row in report["predictions"][:4]:
    print(json.dumps({key:row[key] for key in ["instruction", "reference", "base", "adapter"]}, indent=2))
print("All 64 paired examples are saved in metrics.json; they have not been scored for functional correctness.")
"""
        else:
            result = """from PIL import Image
fig, axes = plt.subplots(1, 2, figsize=(11,5))
for ax, row in zip(axes, report["captions"]):
    with Image.open(DATA_PATH/row["file"]) as image: ax.imshow(image.convert("RGB"))
    ax.set_title(f"True: {row['label']} | classifier: {row['classifier_prediction']}")
    ax.set_xlabel(row["caption"], wrap=True); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Inspect the image against the caption; human-review fields remain empty")
fig.tight_layout(); fig.savefig(OUTPUT/"results.png", dpi=160); plt.show()
print("Complete classifier test:", report["test"])
print("Classifier reload:", report["exact_reload"], "caption quality:", report["caption_quality"])
"""
        nb = nbformat.v4.new_notebook(
            cells=[
                nbformat.v4.new_markdown_cell(
                    f"# {spec['title']}\n\n**Research question:** {spec['question']}\n\nKerem Ozcan | IBM AI Engineering portfolio study | Prepared with AI assistance\n\nThis independently authored experiment applies course concepts. It is not an IBM assessment notebook. Scores below are produced by the current run; no supplied course predictions/checkpoints are used."
                ),
                nbformat.v4.new_markdown_cell(
                    f"## Experiment design\n\n{spec['method']}\n\n**Evaluation boundary:** {spec['limits']}"
                ),
                nbformat.v4.new_markdown_cell(
                    "## Runtime\n\nKaggle internet access is required for provider-hosted data and public model downloads. Retain Kaggle's preinstalled Torch/TensorFlow builds; the cell pins the Transformers/PEFT interfaces used here. CPU is supported; an enabled GPU accelerates the Torch projects. Actual package versions and source hashes are saved with the metrics."
                ),
                nbformat.v4.new_code_cell(BOOTSTRAP),
                nbformat.v4.new_markdown_cell(
                    "## Data provenance\n\n"
                    + spec["sources"]
                    + f"\n\nExpected download SHA-256: `{spec['sha256']}`. The downloader rejects changed bytes and unsafe tar members."
                ),
                nbformat.v4.new_code_cell(support),
                nbformat.v4.new_code_cell(data),
                nbformat.v4.new_markdown_cell(
                    "## Implementation\n\nExpand this cell to inspect the complete standalone implementation. The split manifests, actual processed sample counts and selected-checkpoint reload checks are part of the experiment."
                ),
                nbformat.v4.new_code_cell(code, metadata={"jupyter": {"source_hidden": True}}),
                nbformat.v4.new_code_cell(execute),
                nbformat.v4.new_markdown_cell(
                    "## Measured results\n\nRead the sample counts and stated limitations alongside these measurements."
                ),
                nbformat.v4.new_code_cell(result),
                nbformat.v4.new_markdown_cell(
                    "## What this experiment establishes\n\nThe saved metrics bind this execution to its source hashes, dataset partitions and runtime versions. Checkpoint reload is an implementation check. It does not establish task quality by itself.\n\n"
                    + spec["limits"]
                    + "\n\n## Attribution\n\n"
                    + spec["sources"]
                    + "\n\nConcepts studied through the [IBM AI Engineering Professional Certificate](https://www.coursera.org/professional-certificates/ai-engineer). Source/data/model licenses remain with their providers. No general license is applied to IBM course materials or third-party datasets."
                ),
            ],
            metadata={
                "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                "release_source": {"project_sha256": sha(source), "common_sha256": sha(common)},
            },
        )
        for i, cell in enumerate(nb.cells):
            cell.id = f"release-{slug}-{i}"
            if cell.cell_type == "code":
                ast.parse(cell.source)
        out = ROOT / "kaggle" / slug
        out.mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, out / "notebook.ipynb")
        metadata = {
            "id": f"{owner}/{spec['slug']}",
            "title": spec["title"],
            "code_file": "notebook.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": not public,
            "enable_gpu": slug == "instruction_tuning",
            "machine_shape": "NvidiaTeslaT4" if slug == "instruction_tuning" else "",
            "docker_image_pinning_type": "original",
            "enable_internet": True,
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
        (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Built three Kaggle execution bundles; public={public}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--public", action="store_true", help="Use only after private cloud validation."
    )
    args = parser.parse_args()
    build(public=args.public)
