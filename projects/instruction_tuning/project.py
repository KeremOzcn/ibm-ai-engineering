"""Compare a fixed OPT base with a genuinely trained, saved and reloaded LoRA adapter."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import time
from pathlib import Path

import torch

from projects.common import file_sha256, runtime, source_identity, write_json


def prompt(record: dict) -> str:
    return f"### Instruction:\n{record['instruction'].strip()}\n\n### Response:\n"


def load_records(path: Path, seed: int) -> dict[str, list[dict]]:
    """Group normalized instructions before splitting; support JSON and JSONL sources."""
    text = path.read_text(encoding="utf-8")
    try:
        rows = json.loads(text)
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    groups = {}
    for row in rows:
        if (
            row.get("input", "").strip()
            or not row.get("instruction", "").strip()
            or not row.get("output", "").strip()
        ):
            continue
        key = " ".join(row["instruction"].split()).casefold()
        group_id = hashlib.sha256(key.encode()).hexdigest()
        record_id = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
        groups.setdefault(group_id, {})[record_id] = {**row, "id": record_id, "group": group_id}
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    if len(keys) < 10:
        raise ValueError("Need at least ten distinct instruction groups.")
    train_end, validation_end = int(0.8 * len(keys)), int(0.9 * len(keys))
    return {
        role: [row for key in selection for row in groups[key].values()]
        for role, selection in [
            ("train", keys[:train_end]),
            ("validation", keys[train_end:validation_end]),
            ("test", keys[validation_end:]),
        ]
    }


def encode_response(record: dict, tokenizer, max_length: int) -> dict:
    """Construct response labels explicitly; prompt and padded positions are excluded."""
    prefix = tokenizer.encode(prompt(record), add_special_tokens=True)
    answer = tokenizer.encode(record["output"].strip(), add_special_tokens=False)
    available = max_length - len(prefix)
    if available < 1 or not answer:
        raise ValueError("Prompt leaves no supervised response tokens.")
    truncated = len(answer) + 1 > available
    response = answer[:available] if truncated else answer + [tokenizer.eos_token_id]
    return {
        "input_ids": prefix + response,
        "labels": [-100] * len(prefix) + response,
        "response_truncated": truncated,
    }


def collate(examples: list[dict], pad_id: int) -> dict:
    """Mask padding by position, including when pad and EOS happen to share an ID."""
    width = max(len(row["input_ids"]) for row in examples)
    ids = torch.full((len(examples), width), pad_id, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    attention = torch.zeros_like(ids)
    for index, row in enumerate(examples):
        length = len(row["input_ids"])
        ids[index, :length] = torch.tensor(row["input_ids"])
        labels[index, :length] = torch.tensor(row["labels"])
        attention[index, :length] = 1
    return {"input_ids": ids, "attention_mask": attention, "labels": labels}


def response_nll(model, examples: list[dict], pad_id: int, device) -> dict:
    """Weight loss by actually supervised next-token targets."""
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(examples), 2):
            batch = {
                key: value.to(device)
                for key, value in collate(examples[start : start + 2], pad_id).items()
            }
            tokens = int((batch["labels"][:, 1:] != -100).sum())
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise ValueError("Non-finite response loss.")
            total += loss.item() * tokens
            count += tokens
    if not count:
        raise ValueError("No supervised evaluation tokens.")
    return {"nll": total / count, "tokens": count}


def generate_responses(
    model, tokenizer, records: list[dict], device, max_new_tokens: int
) -> list[str]:
    """Use the identical deterministic prompt/decoding contract for both models."""
    model.eval()
    results = []
    with torch.no_grad():
        for record in records:
            batch = tokenizer(prompt(record), return_tensors="pt")
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model.generate(
                **batch,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            results.append(
                tokenizer.decode(output[0, batch["input_ids"].shape[1] :], skip_special_tokens=True)
            )
    return results


def trainable_digest(model) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run(
    data_file: Path,
    output: Path,
    *,
    full: bool = False,
    allow_downloads: bool = False,
    seed: int = 42,
    epochs: int = 3,
    train_limit: int | None = None,
    evaluation_limit: int = 64,
) -> dict:
    """A pilot performs two real optimizer steps; no supplied predictions/logs are loaded."""
    if epochs < 1 or evaluation_limit < 1 or (train_limit is not None and train_limit < 1):
        raise ValueError("Epochs and record limits must be positive.")
    if not full and train_limit is not None:
        raise ValueError("Explicit training budgets require full=True; the pilot stays fixed.")
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "facebook/opt-350m"
    model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=not allow_downloads).to(
        device
    )
    revision = getattr(model.config, "_commit_hash", None)
    if not revision:
        raise ValueError("The base model must have a resolved immutable revision.")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, local_files_only=not allow_downloads
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    partitions = load_records(data_file, seed)
    manifest = {
        role: [{"id": r["id"], "group": r["group"]} for r in rows]
        for role, rows in partitions.items()
    }
    write_json(output / "split-manifest.json", manifest)
    max_length = 256 if full else 128
    selected, encoded, rejected = {}, {}, {}
    for role, records in partitions.items():
        selected[role], encoded[role], rejected[role] = [], [], 0
        limit = (
            (train_limit if role == "train" else evaluation_limit)
            if full
            else (32 if role == "train" else 4)
        )
        for record in records:
            try:
                example = encode_response(record, tokenizer, max_length)
            except ValueError:
                rejected[role] += 1
                continue
            selected[role].append(record)
            encoded[role].append(example)
            if limit is not None and len(selected[role]) >= limit:
                break
        if not selected[role]:
            raise ValueError(f"No eligible {role} records.")
    started = time.perf_counter()
    base_nll = response_nll(model, encoded["test"], tokenizer.pad_token_id, device)
    base_outputs = generate_responses(
        model, tokenizer, selected["test"], device, 64 if full else 16
    )
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, config)
    before = trainable_digest(model)
    total_parameters = sum(p.numel() for p in model.parameters())
    parameters = [p for p in model.parameters() if p.requires_grad]
    trainable_parameters = sum(p.numel() for p in parameters)
    optimizer = torch.optim.AdamW(parameters, lr=2e-4)
    generator = torch.Generator().manual_seed(seed)
    history, seen = [], []
    best_loss, steps = float("inf"), 0
    adapter_dir = output / "best-adapter"
    for epoch in range(epochs if full else 1):
        model.train()
        order = torch.randperm(len(encoded["train"]), generator=generator).tolist()
        for start in range(0, len(order), 2):
            indices = order[start : start + 2]
            batch = {
                key: value.to(device)
                for key, value in collate(
                    [encoded["train"][i] for i in indices], tokenizer.pad_token_id
                ).items()
            }
            optimizer.zero_grad()
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss.")
            loss.backward()
            optimizer.step()
            seen.extend(selected["train"][i]["id"] for i in indices)
            steps += 1
            if not full and steps >= 2:
                break
        score = response_nll(model, encoded["validation"], tokenizer.pad_token_id, device)
        history.append({"epoch": epoch + 1, "steps": steps, "validation_nll": score["nll"]})
        if score["nll"] < best_loss:
            best_loss = score["nll"]
            model.save_pretrained(adapter_dir, safe_serialization=True)
            tokenizer.save_pretrained(adapter_dir)
    changed = trainable_digest(model) != before
    if not changed:
        raise AssertionError("Optimizer steps did not change the adapter.")
    # Reload the validation-selected adapter, not the final epoch's parameters.
    model.load_adapter(adapter_dir, adapter_name="selected", is_trainable=False)
    model.set_adapter("selected")
    model.eval()
    probe = tokenizer(prompt(selected["validation"][0]), return_tensors="pt").to(device)
    with torch.no_grad():
        expected = model(**probe).logits[:, -1].cpu()
    del model, optimizer, parameters
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    fresh_base = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, local_files_only=not allow_downloads
    ).to(device)
    model = PeftModel.from_pretrained(fresh_base, adapter_dir).to(device).eval()
    with torch.no_grad():
        actual = model(**probe).logits[:, -1].cpu()
    torch.testing.assert_close(expected, actual, rtol=1e-5, atol=1e-5)
    adapter_nll = response_nll(model, encoded["test"], tokenizer.pad_token_id, device)
    adapter_outputs = generate_responses(
        model, tokenizer, selected["test"], device, 64 if full else 16
    )
    report = {
        "project": "instruction_tuning",
        "profile": ("budgeted-experiment" if train_limit is not None else "full")
        if full
        else "bounded-pilot",
        "budget": {
            "epochs": epochs if full else 1,
            "train_limit": train_limit,
            "evaluation_limit": evaluation_limit if full else 4,
        },
        "seed": seed,
        "base_model": model_id,
        "base_revision": revision,
        "data_sha256": file_sha256(data_file),
        "split_manifest_sha256": file_sha256(output / "split-manifest.json"),
        "complete_split_counts": {role: len(rows) for role, rows in partitions.items()},
        "selected_split_counts": {role: len(rows) for role, rows in selected.items()},
        "rejected_before_selection": rejected,
        "truncated_response_counts": {
            role: sum(row["response_truncated"] for row in examples)
            for role, examples in encoded.items()
        },
        "processed_training_ids": seen,
        "optimizer_steps": steps,
        "adapter_changed": changed,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "adapter_sha256": file_sha256(adapter_dir / "adapter_model.safetensors"),
        "reload_max_abs_logit_difference": float((expected - actual).abs().max()),
        "reload_verified": True,
        "base_response_loss": base_nll,
        "adapter_response_loss": adapter_nll,
        "quality_claim": "Response NLL and paired generations only; task correctness and robust improvement remain unmeasured.",
        "decoding": {"do_sample": False, "num_beams": 1, "max_new_tokens": 64 if full else 16},
        "predictions": [
            {
                "id": r["id"],
                "instruction": r["instruction"],
                "reference": r["output"],
                "base": b,
                "adapter": a,
            }
            for r, b, a in zip(selected["test"], base_outputs, adapter_outputs, strict=True)
        ],
        "history": history,
        "seconds": time.perf_counter() - started,
        "runtime": runtime(["torch", "transformers", "peft"]),
        "source": source_identity(
            globals().get("__file__"), globals().get("PROJECT_SOURCE_SHA256")
        ),
    }
    write_json(output / "metrics.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path(".lab-cache/selected-projects/instruction_tuning")
    )
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--allow-downloads", action="store_true")
    args = parser.parse_args()
    report = run(args.data_file, args.output, full=args.full, allow_downloads=args.allow_downloads)
    print({"steps": report["optimizer_steps"], "reload": report["reload_verified"]})


if __name__ == "__main__":
    main()
