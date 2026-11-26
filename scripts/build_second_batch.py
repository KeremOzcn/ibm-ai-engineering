"""Build independent notebooks for the authorized second Kaggle project batch."""

from __future__ import annotations

import argparse
import ast
import json

import nbformat
from build_project_notebooks import ROOT, definitions, sha

SPECS = {
    "translation": {
        "title": "German English Translation GRU vs Transformer",
        "path": "projects/translation/project.py",
        "question": "How do fresh context-vector GRU and causally masked encoder-decoder Transformer models compare under the same translation data/epoch budget?",
        "design": "Official Multi30k train/validation/test-2016 sentence files, pinned to repository revision a3d2e0d26b56f3846f66a952536ffed4e401d05a. Keep at most 48 tokens per side; remove repeated normalized German source sentences with held-out priority. Fit both vocabularies on retained training data only. Share initial embedding tensors, training order, batch size64, learning rate0.0005 and six epochs. Other architecture/parameter/compute differences remain. Validation token NLL selects checkpoints; generate the complete retained test set greedily and compute case-insensitive corpus SacreBLEU (13a), not a hand-picked sentence score.",
        "attribution": "[Multi30k dataset and research references](https://github.com/multi30k/dataset). Text is downloaded from that provider at a pinned revision; the dataset is not bundled or assigned a new license. Concepts studied in the IBM sequence-to-sequence/Transformer courses; no supplied course weights or assessment solutions are included.",
        "limits": "One seed, short-caption domain, capped vocabularies and six epochs. Parameter counts differ, so an isolated effect of attention is not established. BLEU is a limited automatic metric, not a human translation-quality judgment.",
        "dependencies": ["sacrebleu==2.5.1"],
        "inputs": ["translation/"],
        "gpu": True,
    },
    "fraud": {
        "title": "Fraud Detection Under a 100 Alert Review Budget",
        "path": "projects/fraud/project.py",
        "question": "How much fraud can simple models identify when only100 transactions can be reviewed, and what happens when alert rates drift?",
        "design": "Remove repeated events and conflicting labels, then use chronological60/20/20 cuts without changing the rare-event prevalence. Exclude Time from model inputs; fit the scaler/logistic regression and a depth6 tree exclusively on training records. Include a prevalence baseline. Set a fixed cutoff using validation scores and a100-review budget; ties cannot exceed that validation budget. On the later test period, report both the frozen-cutoff workload and a separate fixed top100 ranking. Test workload can differ under the frozen cutoff. Report average precision (the non-interpolated PR summary), ROC-AUC, precision/recall and actual false alerts. Unit costs are explicitly illustrative.",
        "attribution": "The [ULB/Worldline credit-card fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), fetched through the IBM Skills Network teaching mirror. Original dataset conditions remain applicable. No transaction dataset is republished as a notebook output.",
        "limits": "A two-day historical anonymized dataset. Customer grouping and upstream PCA fitting provenance are unknown. Review costs are illustrative units, not measured production savings. The fixed-cutoff policy cannot guarantee a future alert budget under distribution drift.",
        "dependencies": [],
        "inputs": ["fraud/"],
        "gpu": False,
    },
    "cartpole": {
        "title": "CartPole DQN Target Networks Across Three Seeds",
        "path": "projects/cartpole/project.py",
        "question": "Does a lagged target network stabilize this small DQN compared with updating against the online network?",
        "design": "CartPole-v1 with the same500-step limit and natural reward throughout. Compare online-target versus a target synchronized every500 environment steps. Match network, optimizer, replay capacity, warmup, exploration schedule and20,000 environment steps per training seed. Train seeds11/22/33; validation reset seeds1000-1004 choose checkpoints; only then evaluate greedy policies on20 reserved test seeds2000-2019 per training seed. End episodes on either termination or truncation, but bootstrap value targets on truncation. Show the random baseline, all per-seed results and standard deviation across training-seed means.",
        "attribution": "[Farama Gymnasium CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) and its [time-limit guidance](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/). This is original experiment code prepared with AI assistance from the course's deep Q-learning concepts.",
        "limits": "CartPole is a simple simulated task. Three training seeds and20 test episodes per seed are a bounded study, not general RL robustness or real-world control evidence. The online-target configuration may fail; retain that result.",
        "dependencies": ["gymnasium==1.1.1"],
        "inputs": [],
        "gpu": False,
    },
    "preference_alignment": {
        "title": "SFT vs DPO With Frozen Reference and Reload Proof",
        "path": "projects/instruction_tuning/preference_alignment.py",
        "question": "After genuine SFT and DPO updates, does held-out preference likelihood change against the exact frozen SFT reference?",
        "design": "An extension of the instruction-tuning project, using Meta OPT-350M and cleaned Argilla UltraFeedback preference pairs. Keep aligned single-turn pairs and group normalized prompts before80/10/10 splitting. Select256 eligible training/64 validation/64 test pairs. SFT on chosen answers for one pass, then one DPO pass (128 updates per stage, batch size2). Mask prompts/padding, reject identical truncated pairs, disable OPT/LoRA dropout consistently and cache frozen SFT reference log probabilities. Include the SFT weights in checkpoint selection: retain them if candidate DPO validation loss does not improve. Independently reload both SFT and selected-DPO adapters onto fresh bases at the same revision. Report raw and length-normalized pair likelihood rates, chosen-response NLL and reference-adjusted DPO loss, plus fresh paired examples.",
        "attribution": "[Argilla UltraFeedback Binarized Preferences Cleaned](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned), MIT per its dataset card, pinned to revision770076f077c4c5e298498fa32f804857f46d5134. Base: [Meta OPT-350M](https://huggingface.co/facebook/opt-350m), with its research/noncommercial terms. [DPO paper](https://arxiv.org/abs/2305.18290). No supplied course checkpoint, pretrained base weights or dataset archive is bundled.",
        "limits": "Small SFT/preference budgets, one seed, synthetic preference labels and truncated responses. Pairwise log-probability rates and implicit DPO margins are not generated-response win rates, task correctness or human-validated quality. This is a separate alignment experiment on the same base architecture, not reuse of the earlier CodeAlpaca adapter.",
        "dependencies": [
            "transformers==4.45.2",
            "peft==0.11.1",
            "accelerate==1.15.0",
            "pyarrow>=17,<27",
        ],
        "inputs": ["preference/"],
        "gpu": True,
    },
}


def build(*, public: bool = False, only: str | None = None) -> None:
    """Preserve first-batch artifacts and generate each second-batch source deterministically."""
    inputs = json.loads((ROOT / "provenance/second-batch-inputs.json").read_text())
    common = (ROOT / "projects/common.py").read_text()
    support = (ROOT / "scripts/kaggle_release_support.py").read_text()
    for slug, spec in SPECS.items():
        if only is not None and slug != only:
            continue
        if not 5 <= len(spec["title"]) <= 50:
            raise ValueError("Kaggle title length is outside5-50.")
        source = (ROOT / spec["path"]).read_text()
        embedded = (
            common
            + f"\nCOMMON_SOURCE_SHA256={sha(common)!r}\nPROJECT_SOURCE_SHA256={sha(source)!r}\n"
            + definitions(source)
        )
        files = [
            row
            for row in inputs
            if any(row["file"].startswith(prefix) for prefix in spec["inputs"])
        ]
        startup = f"""import os
import sys
import subprocess
import gzip
import shutil
from pathlib import Path
os.environ["USE_TF"]="0"
os.environ["TOKENIZERS_PARALLELISM"]="false"
os.environ["OMP_NUM_THREADS"]="1"
ON_KAGGLE=Path("/kaggle/working").is_dir()
if ON_KAGGLE and {bool(spec['dependencies'])!r}:
    subprocess.run([sys.executable,"-m","pip","install","-q",*{spec['dependencies']!r}],check=True)
import matplotlib.pyplot as plt
import numpy as np
import json
FULL_RUN=os.environ.get("SECOND_BATCH_FULL","1")=="1"
LOCAL_ROOT=next((p for p in [Path.cwd(),*Path.cwd().parents] if (p/"provenance/source-manifest.json").exists()),Path.cwd())
WORK=Path("/kaggle/working") if ON_KAGGLE else LOCAL_ROOT/".lab-cache/second-batch-notebooks"/{slug!r}/"exports"
SCRATCH=Path("/kaggle/temp/ibm-second-batch") if ON_KAGGLE else LOCAL_ROOT/".lab-cache/second-batch-notebooks"/{slug!r}/"scratch"
WORK.mkdir(parents=True,exist_ok=True);SCRATCH.mkdir(parents=True,exist_ok=True)
DATA_ROOT=SCRATCH/"inputs" if ON_KAGGLE else LOCAL_ROOT/".lab-cache/second-batch-data"
RUN_OUTPUT=SCRATCH/{slug!r};OUTPUT=WORK/{slug!r};OUTPUT.mkdir(parents=True,exist_ok=True)
"""
        acquire = f"""INPUTS={files!r}
for entry in INPUTS:
    path=verified_download(entry["url"],DATA_ROOT/entry["file"],entry["sha256"])
    if path.suffix==".gz":
        with gzip.open(path,"rb") as source,path.with_suffix("").open("wb") as target:
            shutil.copyfileobj(source,target)
"""
        argument = {
            "translation": "DATA_ROOT/'translation', RUN_OUTPUT, full=FULL_RUN",
            "fraud": "DATA_ROOT/'fraud/creditcard.csv', RUN_OUTPUT",
            "cartpole": "RUN_OUTPUT, full=FULL_RUN",
            "preference_alignment": "DATA_ROOT/'preference/cleaned.parquet', RUN_OUTPUT, full=FULL_RUN, allow_downloads=ON_KAGGLE",
        }[slug]
        execute = f"""report=run({argument})
for artifact in RUN_OUTPUT.glob("*.json"):
    shutil.copyfile(artifact,OUTPUT/artifact.name)
print("Executed profile:",report["profile"])
print("Actual runtime:",report["runtime"])
print("Source identity:",report["source"])
"""
        if slug == "fraud":
            plotting = """fig,axes=plt.subplots(1,2,figsize=(11,4))
for name,result in report["models"].items():
    curve=result["pr_curve"]
    axes[0].plot(curve["recall"],curve["precision"],label=f"{name} AP={result['average_precision']:.3f}")
names=list(report["models"])
axes[1].bar(names,[report["models"][n]["top_budget_test"]["true_alerts"] for n in names],color="#138477")
axes[0].set(xlabel="Recall",ylabel="Precision",title="Later-period test precision/recall");axes[0].legend(fontsize=8)
axes[1].set(ylabel="Fraud found in top100 reviews",title="Fixed batch review budget");axes[1].tick_params(axis="x",labelrotation=25)
fig.tight_layout();fig.savefig(OUTPUT/"results.png",dpi=160);plt.show()
print("Actual split counts:",report["split_counts"])
for name,result in report["models"].items():print(name,result["fixed_threshold_test"],result["top_budget_test"])
print(report["cost_assumptions"])
"""
        elif slug == "cartpole":
            plotting = """fig,axes=plt.subplots(1,2,figsize=(11,4))
for name,result in report["models"].items():
    curves=np.array([[v["validation_mean_return"] for v in r["history"]] for r in result["runs"]])
    steps=[v["environment_steps"] for v in result["runs"][0]["history"]]
    axes[0].plot(steps,curves.mean(0),label=name)
    axes[0].fill_between(steps,curves.mean(0)-curves.std(0),curves.mean(0)+curves.std(0),alpha=.15)
names=list(report["models"])
axes[1].bar(names,[report["models"][n]["mean_of_seed_means"] for n in names],yerr=[report["models"][n]["std_of_seed_means"] or 0 for n in names],color="#138477",capsize=4)
axes[1].axhline(np.mean(report["random_baseline_returns"]),color="#d36d42",label="Random baseline")
axes[0].set(xlabel="Environment steps per seed",ylabel="Validation return",title="Matched budgets; shaded seed variation");axes[0].legend()
axes[1].set(ylabel="Mean test episode return",title="Error bars: SD of training-seed means");axes[1].legend()
fig.tight_layout();fig.savefig(OUTPUT/"results.png",dpi=160);plt.show()
for name,result in report["models"].items():print(name,[(r["seed"],r["test_mean_return"],r["exact_reload"]) for r in result["runs"]])
print("Seed separation and episode cap:",report["manifest"])
"""
        elif slug == "translation":
            plotting = """fig,axes=plt.subplots(1,2,figsize=(11,4))
names=list(report["models"])
axes[0].bar(names,[report["models"][n]["corpus_bleu"] for n in names],color=["#60758a","#138477"])
axes[0].set(ylabel="Case-insensitive corpus BLEU",title="Complete retained held-out set")
for name,result in report["models"].items():
    axes[1].plot([h["epoch"] for h in result["history"]],[h["validation_nll"] for h in result["history"]],label=name)
    print(name,"parameters",result["parameters"],"BLEU signature",result["bleu_signature"],"test samples",result["test_samples"],"reload",result["exact_reload"])
    for example in result["examples"][:3]:print(json.dumps(example,ensure_ascii=False))
axes[1].set(xlabel="Epoch",ylabel="Validation token NLL",title="Checkpoint selection");axes[1].legend()
fig.tight_layout();fig.savefig(OUTPUT/"results.png",dpi=160);plt.show()
print("Actual selected counts:",report["selected_split_counts"])
print(report["comparison"])
"""
        else:
            plotting = """fig,axes=plt.subplots(1,2,figsize=(11,4))
names=list(report["models"])
axes[0].bar(names,[report["models"][n]["chosen_response_nll"] for n in names],color="#138477")
axes[1].bar(names,[report["models"][n]["length_normalized_preference_rate"] for n in names],color="#60758a")
axes[0].set(ylabel="Chosen response token NLL",title="Reference likelihood; lower is better")
axes[1].set(ylabel="Held-out pair likelihood preference rate",ylim=(0,1),title="Not a generated-response win rate")
fig.tight_layout();fig.savefig(OUTPUT/"results.png",dpi=160);plt.show()
print("Actual pairs:",report["selected_split_counts"],"SFT/DPO updates:",report["sft_optimizer_steps"],report["dpo_optimizer_steps"])
print("Validation-selected stage:",report["selected_stage"],report["validation_losses"])
print("Reload differences:",report["sft_reload_max_abs_difference"],report["dpo_reload_max_abs_difference"])
print("Response truncations:",report["truncated_response_counts"])
for row in report["examples"][:3]:print(json.dumps(row,ensure_ascii=False))
"""
        notebook = nbformat.v4.new_notebook(
            cells=[
                nbformat.v4.new_markdown_cell(
                    f"# {spec['title']}\n\n**Question:** {spec['question']}\n\nKerem Ozcan | IBM AI Engineering portfolio | Prepared with AI assistance\n\nOriginal experiment code developed from course concepts. This notebook contains no IBM assessment solutions or supplied course model predictions."
                ),
                nbformat.v4.new_markdown_cell(
                    "## Design and evaluation boundary\n\n"
                    + spec["design"]
                    + "\n\n"
                    + spec["limits"]
                ),
                nbformat.v4.new_markdown_cell(
                    "## Runtime\n\nKaggle internet is required for provider-hosted inputs/models. Local verification uses already installed isolated environments. `SECOND_BATCH_FULL=0` selects a small local correctness pilot for the neural projects; the published Kaggle version uses the documented full experiment budget. Data and weights stay in scratch storage; only JSON evidence, plots and notebook outputs are exported."
                ),
                nbformat.v4.new_code_cell(startup),
                nbformat.v4.new_markdown_cell(
                    "## Inputs and provenance\n\n"
                    + spec["attribution"]
                    + "\n\nProvider downloads must match the pinned SHA-256 values below. Model weights are fetched from the original provider; no submitted lab checkpoint is reused."
                ),
                nbformat.v4.new_code_cell(support),
                nbformat.v4.new_code_cell(acquire),
                nbformat.v4.new_markdown_cell(
                    "## Standalone implementation\n\nExpand this cell to inspect data partitions, actual sample accounting, training and selected-checkpoint reload checks."
                ),
                nbformat.v4.new_code_cell(embedded, metadata={"jupyter": {"source_hidden": True}}),
                nbformat.v4.new_code_cell(execute),
                nbformat.v4.new_markdown_cell(
                    "## Freshly measured results\n\nThese values are generated by this execution. Keep unsuccessful baselines and the stated evaluation limits visible."
                ),
                nbformat.v4.new_code_cell(plotting),
                nbformat.v4.new_markdown_cell(
                    "## Interpretation and attribution\n\n"
                    + spec["limits"]
                    + "\n\n"
                    + spec["attribution"]
                    + "\n\nConcepts studied through the [IBM AI Engineering Professional Certificate](https://www.coursera.org/professional-certificates/ai-engineer). Original provider conditions remain applicable. Mechanical execution and reload checks do not replace task-specific or human quality evaluation."
                ),
            ],
            metadata={
                "kernelspec": {"name": "python3", "display_name": "Python3", "language": "python"},
                "release_source": {"project_sha256": sha(source), "common_sha256": sha(common)},
            },
        )
        for i, cell in enumerate(notebook.cells):
            cell.id = f"second-{slug}-{i}"
            if cell.cell_type == "code":
                ast.parse(cell.source)
        destination = ROOT / "kaggle" / slug
        destination.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, destination / "notebook.ipynb")
        project_notebook = ROOT / (
            "projects/instruction_tuning/preference_alignment.ipynb"
            if slug == "preference_alignment"
            else f"projects/{slug}/notebook.ipynb"
        )
        nbformat.write(notebook, project_notebook)
        kernel_slug = spec["title"].lower().replace(" ", "-")
        metadata = {
            "id": f"kerem0zcan/{kernel_slug}",
            "title": spec["title"],
            "code_file": "notebook.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": not public,
            "enable_gpu": spec["gpu"],
            "machine_shape": "NvidiaTeslaT4" if spec["gpu"] else "",
            "enable_internet": True,
            "docker_image_pinning_type": "original",
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
        (destination / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Generated authorized second-batch notebooks; public=", public)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--project", choices=SPECS)
    args = parser.parse_args()
    build(public=args.public, only=args.project)
