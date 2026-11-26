"""Check new project source, actual notebook evidence and the independent second release ledger."""

from __future__ import annotations

import hashlib
import json

import nbformat
from build_second_batch import ROOT, SPECS, definitions, sha


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_metrics(slug, metrics, manifest, *, full=False):
    """Require actual processed denominators, separated holdouts and saved-model reload evidence."""
    if metrics["project"] != slug:
        raise ValueError("Wrong project evidence.")
    model_keys = {
        "translation": {"gru", "transformer"},
        "fraud": {"prevalence_baseline", "logistic_regression", "decision_tree"},
        "cartpole": {"online_target", "lagged_target"},
        "preference_alignment": {"base", "sft", "selected_dpo"},
    }
    if set(metrics["models"]) != model_keys[slug]:
        raise ValueError("Missing or unexpected model results.")
    if full:
        profiles = {
            "cartpole": "three-seed-step-budget",
            "translation": "complete-retained-splits",
            "preference_alignment": "bounded-sft-dpo",
            "fraud": "chronological-complete-splits",
        }
        if metrics["profile"] != profiles[slug]:
            raise ValueError("A bounded pilot cannot satisfy the public experiment scope.")
        if slug == "translation":
            if metrics["selected_split_counts"] != metrics["complete_split_counts"] or any(
                len(m["history"]) != 6
                or any(
                    h["train_samples"] != metrics["complete_split_counts"]["train"]
                    for h in m["history"]
                )
                for m in metrics["models"].values()
            ):
                raise ValueError("Translation full data/epoch budget is incomplete.")
        elif slug == "cartpole":
            if (
                manifest["train_seeds"] != [11, 22, 33]
                or manifest["validation_seeds"] != list(range(1000, 1005))
                or manifest["test_seeds"] != list(range(2000, 2020))
                or any(
                    r["environment_steps"] != 20000
                    for m in metrics["models"].values()
                    for r in m["runs"]
                )
            ):
                raise ValueError("RL full seed/step budget is incomplete.")
        elif slug == "preference_alignment":
            if (
                metrics["selected_split_counts"] != {"train": 256, "validation": 64, "test": 64}
                or metrics["sft_optimizer_steps"] != 128
                or metrics["dpo_optimizer_steps"] != 128
            ):
                raise ValueError("Alignment full pair/update budget is incomplete.")
    if slug == "cartpole":
        partitions = [set(manifest[k]) for k in ["train_seeds", "validation_seeds", "test_seeds"]]
        for value in metrics["models"].values():
            if (
                len(value["runs"]) != len(partitions[0])
                or {r["seed"] for r in value["runs"]} != partitions[0]
            ):
                raise ValueError("Missing RL training seeds.")
            if not all(
                r["exact_reload"]
                and len(r["test_returns"]) == len(partitions[2])
                and all(1 <= x <= 500 for x in r["test_returns"])
                for r in value["runs"]
            ):
                raise ValueError("Incomplete RL rollout/reload evidence.")
        if manifest["episode_limit"] != 500:
            raise ValueError("Inconsistent RL horizon.")
    else:
        partitions = []
        for role in ["train", "validation", "test"]:
            rows = manifest["partitions"][role]
            expected_count = (
                metrics["split_counts"][role]["samples"]
                if slug == "fraud"
                else metrics["selected_split_counts"][role]
            )
            if not rows or len(rows) != expected_count:
                raise ValueError("Manifest partition count differs from processed samples.")
            partitions.append({r["group"] if isinstance(r, dict) else r for r in rows})
        if slug == "translation":
            if not all(
                m["exact_reload"] and m["test_samples"] == metrics["selected_split_counts"]["test"]
                for m in metrics["models"].values()
            ):
                raise ValueError("Incomplete translation accounting/reload.")
        elif slug == "fraud":
            if (
                not metrics["split_counts"]["train"]["max_time"]
                < metrics["split_counts"]["validation"]["min_time"]
                or not metrics["split_counts"]["validation"]["max_time"]
                < metrics["split_counts"]["test"]["min_time"]
            ):
                raise ValueError("Chronological partitions overlap.")
            if not all(
                m["exact_reload"]
                and m["test_samples"] == metrics["split_counts"]["test"]["samples"]
                for m in metrics["models"].values()
            ):
                raise ValueError("Incomplete fraud accounting/reload.")
            if any(
                m["validation_alerts"] > metrics["review_budget"]
                or m["top_budget_test"]["alerts"] > metrics["review_budget"]
                for m in metrics["models"].values()
            ):
                raise ValueError("Review budget exceeded.")
        else:
            if not metrics["reload_verified"] or not metrics["dpo_training_changed_adapter"]:
                raise ValueError("DPO training/reload proof is missing.")
            if metrics["sft_optimizer_steps"] < 1 or metrics["dpo_optimizer_steps"] < 1:
                raise ValueError("No alignment optimization occurred.")
            if not all(
                m["pairs"] == metrics["selected_split_counts"]["test"]
                for m in metrics["models"].values()
            ):
                raise ValueError("Preference metrics use different test denominators.")
    if any(partitions[a] & partitions[b] for a, b in [(0, 1), (0, 2), (1, 2)]):
        raise ValueError(f"Holdout leakage: {slug}")


def check() -> dict:
    """Check published/evaluated artifacts without importing or training any model."""
    for slug, spec in SPECS.items():
        source = ROOT / "kaggle" / slug / "notebook.ipynb"
        notebook = nbformat.read(source, as_version=4)
        nbformat.validate(notebook)
        expected = {
            "project_sha256": digest(ROOT / spec["path"]),
            "common_sha256": digest(ROOT / "projects/common.py"),
        }
        common_text = (ROOT / "projects/common.py").read_text()
        project_text = (ROOT / spec["path"]).read_text()
        embedded = (
            common_text
            + f"\nCOMMON_SOURCE_SHA256={sha(common_text)!r}\nPROJECT_SOURCE_SHA256={sha(project_text)!r}\n"
            + definitions(project_text)
        )
        if notebook.cells[8].source != embedded:
            raise ValueError(f"Embedded implementation differs: {slug}")
        study_path = ROOT / (
            "projects/instruction_tuning/preference_alignment.ipynb"
            if slug == "preference_alignment"
            else f"projects/{slug}/notebook.ipynb"
        )
        study = nbformat.read(study_path, as_version=4)
        if [(c.cell_type, c.source) for c in study.cells] != [
            (c.cell_type, c.source) for c in notebook.cells
        ]:
            raise ValueError(f"Study and Kaggle notebooks differ: {slug}")
        if dict(notebook.metadata.release_source) != expected:
            raise ValueError(f"Stale second-batch source: {slug}")
        folder = ROOT / "reports/second-batch-notebooks" / slug
        evidence = json.loads((folder / "evidence.json").read_text())
        required_local = {"executed.ipynb", "metrics.json", "split-manifest.json", "results.png"}
        if not required_local.issubset(evidence["files"]):
            raise ValueError(f"Incomplete local notebook evidence: {slug}")
        if not evidence["fresh_kernel"] or evidence["source_notebook_sha256"] != digest(source):
            raise ValueError(f"Stale second-batch notebook execution: {slug}")
        for name, value in evidence["files"].items():
            if (
                not (folder / name).resolve().is_relative_to(folder.resolve())
                or digest(folder / name) != value
            ):
                raise ValueError(f"Changed local artifact: {slug}/{name}")
        executed = nbformat.read(folder / "executed.ipynb", as_version=4)
        if [(c.cell_type, c.source) for c in executed.cells] != [
            (c.cell_type, c.source) for c in notebook.cells
        ]:
            raise ValueError("Executed notebook source differs.")
        if any(
            c.execution_count is None or any(o.output_type == "error" for o in c.outputs)
            for c in executed.cells
            if c.cell_type == "code" and c.source.strip()
        ):
            raise ValueError("Incomplete local notebook.")
        metrics = json.loads((folder / "metrics.json").read_text())
        manifest = json.loads((folder / "split-manifest.json").read_text())
        if metrics["source"] != expected or metrics["split_manifest_sha256"] != digest(
            folder / "split-manifest.json"
        ):
            raise ValueError("Local evidence/source mismatch.")
        validate_metrics(slug, metrics, manifest)
    release = ROOT / "reports/kaggle-second-batch/release.json"
    if release.exists():
        rows = json.loads(release.read_text())
        if len(rows) != 4 or {r["project"] for r in rows} != set(SPECS):
            raise ValueError("Incomplete second-batch release ledger.")
        for row in rows:
            slug = row["project"]
            folder = release.parent / slug
            required = {
                "metrics.json",
                "split-manifest.json",
                "results.png",
                "source.ipynb",
                "kernel-metadata.json",
                "execution.log",
            }
            if slug == "translation":
                required |= {"gru-predictions.json", "transformer-predictions.json"}
            if slug == "preference_alignment":
                required |= {"reference-logps.json"}
            if row["status"] != "COMPLETE" or row["is_private"] or set(row["files"]) != required:
                raise ValueError("Second-batch publication is incomplete.")
            for name, value in row["files"].items():
                if digest(folder / name) != value:
                    raise ValueError(f"Changed cloud artifact: {slug}/{name}")
            metadata = json.loads((folder / "kernel-metadata.json").read_text())
            if (
                metadata["is_private"]
                or metadata["id"] != row["ref"]
                or metadata["id_no"] != row["kernel_id"]
            ):
                raise ValueError("Remote visibility/identity differs.")
            remote = nbformat.read(folder / "source.ipynb", as_version=4)
            authored = nbformat.read(ROOT / "kaggle" / slug / "notebook.ipynb", as_version=4)
            if [(c.cell_type, c.source) for c in remote.cells] != [
                (c.cell_type, c.source) for c in authored.cells
            ]:
                raise ValueError("Published source differs.")
            metrics = json.loads((folder / "metrics.json").read_text())
            manifest = json.loads((folder / "split-manifest.json").read_text())
            if metrics["source"]["project_sha256"] != digest(ROOT / SPECS[slug]["path"]) or metrics[
                "source"
            ]["common_sha256"] != digest(ROOT / "projects/common.py"):
                raise ValueError("Published model source is stale.")
            if metrics["split_manifest_sha256"] != digest(folder / "split-manifest.json"):
                raise ValueError("Cloud manifest differs.")
            validate_metrics(slug, metrics, manifest, full=True)
    return {"local_notebooks": 4, "public_versions": 4 if release.exists() else 0}


if __name__ == "__main__":
    print(json.dumps(check()))
