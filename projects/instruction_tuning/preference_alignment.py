"""Extend instruction tuning with real SFT and DPO updates against a frozen SFT reference."""

from __future__ import annotations

import gc
import hashlib
import json
import random
import time
from pathlib import Path

import torch

from projects.common import file_sha256, runtime, source_identity, write_json


def instruction_prompt(text: str) -> str:
    return f"### Instruction:\n{text.strip()}\n\n### Response:\n"


def load_base(model_id: str, revision: str | None, allow_downloads: bool):
    """Pin the base revision and disable OPT dropout consistently for policy/reference scores."""
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(
        model_id, revision=revision, local_files_only=not allow_downloads
    )
    config.dropout = 0.0
    config.attention_dropout = 0.0
    resolved = getattr(config, "_commit_hash", None)
    if not resolved:
        raise ValueError("An immutable base revision is required.")
    return AutoModelForCausalLM.from_pretrained(
        model_id, revision=resolved, config=config, local_files_only=not allow_downloads
    )


def preference_partitions(path: Path, seed: int) -> dict:
    """Group normalized prompts before splitting; keep aligned single-turn response pairs."""
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["prompt", "chosen", "rejected"])
    groups = {}
    for row in table.to_pylist():
        chosen, rejected = row["chosen"], row["rejected"]
        if len(chosen) != 2 or len(rejected) != 2:
            continue
        if [r["role"] for r in chosen] != ["user", "assistant"] or [
            r["role"] for r in rejected
        ] != ["user", "assistant"]:
            continue
        prompt = row["prompt"].strip()
        if chosen[0]["content"].strip() != prompt or rejected[0]["content"].strip() != prompt:
            continue
        yes, no = chosen[-1]["content"].strip(), rejected[-1]["content"].strip()
        if not prompt or not yes or not no or yes == no:
            continue
        group = hashlib.sha256(" ".join(prompt.split()).casefold().encode()).hexdigest()
        identifier = hashlib.sha256(
            json.dumps([prompt, yes, no], ensure_ascii=False).encode()
        ).hexdigest()
        groups.setdefault(group, {})[identifier] = {
            "id": identifier,
            "group": group,
            "prompt": prompt,
            "chosen": yes,
            "rejected": no,
        }
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    a, b = int(len(keys) * 0.8), int(len(keys) * 0.9)
    return {
        role: [r for key in selection for r in groups[key].values()]
        for role, selection in [("train", keys[:a]), ("validation", keys[a:b]), ("test", keys[b:])]
    }


def encode_preference(row: dict, tokenizer, maximum: int = 256) -> dict:
    """Preserve the shared prompt, supervise responses only and reject identical truncations."""
    prefix = tokenizer.encode(instruction_prompt(row["prompt"]), add_special_tokens=True)
    if len(prefix) > maximum // 2:
        raise ValueError("Prompt exceeds the reserved prompt budget.")
    result = {}
    for side in ["chosen", "rejected"]:
        answer = tokenizer.encode(row[side], add_special_tokens=False) + [tokenizer.eos_token_id]
        limit = maximum - len(prefix)
        truncated = len(answer) > limit
        answer = answer[:limit]
        result[side] = {
            "input_ids": prefix + answer,
            "labels": [-100] * len(prefix) + answer,
            "truncated": truncated,
        }
    if result["chosen"]["input_ids"] == result["rejected"]["input_ids"]:
        raise ValueError("Responses become identical after truncation.")
    return {**result, "id": row["id"], "group": row["group"]}


def response_logps(model, examples: list[dict], pad_id: int, device):
    """Sum only supervised next-token log probabilities; retain token counts for NLL."""
    width = max(len(e["input_ids"]) for e in examples)
    ids = torch.full((len(examples), width), pad_id, dtype=torch.long, device=device)
    labels = torch.full_like(ids, -100)
    attention = torch.zeros_like(ids)
    for i, row in enumerate(examples):
        n = len(row["input_ids"])
        ids[i, :n] = torch.tensor(row["input_ids"], device=device)
        labels[i, :n] = torch.tensor(row["labels"], device=device)
        attention[i, :n] = 1
    logits = model(input_ids=ids, attention_mask=attention).logits[:, :-1]
    following = labels[:, 1:]
    mask = following.ne(-100)
    values = logits.log_softmax(-1).gather(-1, following.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    counts = mask.sum(1)
    if bool((counts == 0).any()):
        raise ValueError("Every response needs supervised next-token targets.")
    return (values * mask).sum(1), counts


def dpo_objective(chosen, rejected, reference_chosen, reference_rejected, beta: float = 0.1):
    """Canonical sigmoid DPO with fixed reference sequence log probabilities."""
    margin = beta * ((chosen - rejected) - (reference_chosen - reference_rejected))
    return -torch.nn.functional.logsigmoid(margin).mean()


def pair_scores(model, examples: list[dict], pad_id: int, device) -> list[dict]:
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(examples), 2):
            batch = examples[start : start + 2]
            flat = [r[side] for r in batch for side in ["chosen", "rejected"]]
            values, counts = response_logps(model, flat, pad_id, device)
            values = values.cpu().tolist()
            counts = counts.cpu().tolist()
            for i, row in enumerate(batch):
                result.append(
                    {
                        "id": row["id"],
                        "chosen": values[2 * i],
                        "rejected": values[2 * i + 1],
                        "chosen_tokens": counts[2 * i],
                        "rejected_tokens": counts[2 * i + 1],
                    }
                )
    return result


def summarize(scores: list[dict], reference: list[dict] | None = None) -> dict:
    nll = -sum(r["chosen"] for r in scores) / sum(r["chosen_tokens"] for r in scores)
    result = {
        "pairs": len(scores),
        "chosen_response_nll": nll,
        "raw_sequence_preference_rate": sum(r["chosen"] > r["rejected"] for r in scores)
        / len(scores),
        "length_normalized_preference_rate": sum(
            r["chosen"] / r["chosen_tokens"] > r["rejected"] / r["rejected_tokens"] for r in scores
        )
        / len(scores),
    }
    if reference is not None:
        if [r["id"] for r in scores] != [r["id"] for r in reference]:
            raise ValueError("Reference pairs are misaligned.")
        margins = [
            0.1 * ((r["chosen"] - r["rejected"]) - (s["chosen"] - s["rejected"]))
            for r, s in zip(scores, reference, strict=True)
        ]
        result["reference_adjusted_dpo_loss"] = sum(
            float(torch.nn.functional.softplus(torch.tensor(-m))) for m in margins
        ) / len(margins)
        result["reference_adjusted_margin_positive_rate"] = sum(m > 0 for m in margins) / len(
            margins
        )
    return result


def adapter_digest(model) -> str:
    """Check that optimization changes actual trainable tensor bytes."""
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        if value.requires_grad:
            digest.update(name.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def generate_examples(model, tokenizer, records: list[dict], device) -> list[str]:
    model.eval()
    outputs = []
    with torch.no_grad():
        for row in records[:6]:
            inputs = tokenizer(instruction_prompt(row["prompt"]), return_tensors="pt").to(device)
            ids = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
            )
            outputs.append(
                tokenizer.decode(ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            )
    return outputs


def run(
    data_file: Path,
    output: Path,
    *,
    full: bool = False,
    seed: int = 42,
    allow_downloads: bool = False,
) -> dict:
    """Use a fresh SFT baseline, frozen reference scores and independently reload selected DPO."""
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoTokenizer

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "facebook/opt-350m"
    model = load_base(model_id, None, allow_downloads).to(device)
    revision = getattr(model.config, "_commit_hash", None)
    if not revision:
        raise ValueError("An immutable base revision is required.")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, local_files_only=not allow_downloads
    )
    parts = preference_partitions(data_file, seed)
    selected, encoded, rejected = {}, {}, {}
    for role, rows in parts.items():
        selected[role], encoded[role], rejected[role] = [], [], 0
        limit = (256 if role == "train" else 64) if full else (16 if role == "train" else 8)
        for row in rows:
            try:
                example = encode_preference(row, tokenizer)
            except ValueError:
                rejected[role] += 1
                continue
            selected[role].append(row)
            encoded[role].append(example)
            if len(selected[role]) == limit:
                break
        if len(selected[role]) != limit:
            raise ValueError(f"Insufficient eligible {role} pairs.")
    manifest = {
        "partitions": {
            role: [{"id": r["id"], "group": r["group"]} for r in rows]
            for role, rows in selected.items()
        },
        "available_grouped_counts": {r: len(v) for r, v in parts.items()},
        "rejected_before_selection": rejected,
    }
    write_json(output / "split-manifest.json", manifest)
    del parts
    started = time.perf_counter()
    results = {
        "base": summarize(pair_scores(model, encoded["test"], tokenizer.pad_token_id, device))
    }
    generations = {"base": generate_examples(model, tokenizer, selected["test"], device)}
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0,
            task_type=TaskType.CAUSAL_LM,
        ),
    )
    initial = adapter_digest(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    order = list(range(len(encoded["train"])))
    random.Random(seed).shuffle(order)
    sft_steps = 0
    for start in range(0, len(order), 2):
        model.train()
        batch = [encoded["train"][i]["chosen"] for i in order[start : start + 2]]
        values, tokens = response_logps(model, batch, tokenizer.pad_token_id, device)
        loss = -values.sum() / tokens.sum()
        if not bool(torch.isfinite(loss)):
            raise ValueError("Non-finite SFT loss.")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sft_steps += 1
    if adapter_digest(model) == initial:
        raise AssertionError("SFT did not update the adapter.")
    sft_dir = output / "sft-adapter"
    model.save_pretrained(sft_dir, safe_serialization=True)
    reference = {
        role: pair_scores(model, examples, tokenizer.pad_token_id, device)
        for role, examples in encoded.items()
    }
    write_json(output / "reference-logps.json", reference)
    results["sft"] = summarize(reference["test"], reference["test"])
    generations["sft"] = generate_examples(model, tokenizer, selected["test"], device)
    probe = tokenizer(
        instruction_prompt(selected["validation"][0]["prompt"]), return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        expected = model(**probe).logits[:, -1].cpu()
    del model, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model = (
        PeftModel.from_pretrained(
            load_base(model_id, revision, allow_downloads),
            sft_dir,
            is_trainable=True,
        )
        .to(device)
        .eval()
    )
    with torch.no_grad():
        actual = model(**probe).logits[:, -1].cpu()
    torch.testing.assert_close(expected, actual, rtol=1e-5, atol=1e-5)
    sft_reload = float((expected - actual).abs().max())
    before_dpo = adapter_digest(model)
    best_dir = output / "selected-dpo-adapter"
    model.save_pretrained(best_dir, safe_serialization=True)
    initial_val = summarize(reference["validation"], reference["validation"])[
        "reference_adjusted_dpo_loss"
    ]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-5)
    dpo_steps = 0
    for start in range(0, len(order), 2):
        model.train()
        indices = order[start : start + 2]
        flat = [encoded["train"][i][side] for i in indices for side in ["chosen", "rejected"]]
        values, _ = response_logps(model, flat, tokenizer.pad_token_id, device)
        refs = torch.tensor(
            [[reference["train"][i][side] for side in ["chosen", "rejected"]] for i in indices],
            device=device,
        )
        loss = dpo_objective(values[::2], values[1::2], refs[:, 0], refs[:, 1])
        if not bool(torch.isfinite(loss)):
            raise ValueError("Non-finite DPO loss.")
        optimizer.zero_grad()
        loss.backward()
        nn_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not bool(torch.isfinite(nn_norm)):
            raise ValueError("Non-finite DPO gradients.")
        optimizer.step()
        dpo_steps += 1
    changed = adapter_digest(model) != before_dpo
    if not changed:
        raise AssertionError("DPO did not update the adapter.")
    candidate_val = summarize(
        pair_scores(model, encoded["validation"], tokenizer.pad_token_id, device),
        reference["validation"],
    )["reference_adjusted_dpo_loss"]
    selected_stage = "dpo" if candidate_val < initial_val else "sft-retained"
    if selected_stage == "dpo":
        model.save_pretrained(best_dir, safe_serialization=True)
    model.load_adapter(best_dir, adapter_name="selected", is_trainable=False)
    model.set_adapter("selected")
    model.eval()
    with torch.no_grad():
        expected = model(**probe).logits[:, -1].cpu()
    del model, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model = (
        PeftModel.from_pretrained(
            load_base(model_id, revision, allow_downloads),
            best_dir,
        )
        .to(device)
        .eval()
    )
    with torch.no_grad():
        actual = model(**probe).logits[:, -1].cpu()
    torch.testing.assert_close(expected, actual, rtol=1e-5, atol=1e-5)
    results["selected_dpo"] = summarize(
        pair_scores(model, encoded["test"], tokenizer.pad_token_id, device), reference["test"]
    )
    generations["selected_dpo"] = generate_examples(model, tokenizer, selected["test"], device)
    report = {
        "project": "preference_alignment",
        "profile": "bounded-sft-dpo" if full else "bounded-pilot",
        "seed": seed,
        "base_model": model_id,
        "base_revision": revision,
        "device": str(device),
        "data_sha256": file_sha256(data_file),
        "split_manifest_sha256": file_sha256(output / "split-manifest.json"),
        "selected_split_counts": {r: len(v) for r, v in selected.items()},
        "sft_optimizer_steps": sft_steps,
        "dpo_optimizer_steps": dpo_steps,
        "sft_adapter_sha256": file_sha256(sft_dir / "adapter_model.safetensors"),
        "selected_adapter_sha256": file_sha256(best_dir / "adapter_model.safetensors"),
        "reference_logps_sha256": file_sha256(output / "reference-logps.json"),
        "selected_stage": selected_stage,
        "sft_reload_max_abs_difference": sft_reload,
        "dpo_reload_max_abs_difference": float((expected - actual).abs().max()),
        "reload_verified": True,
        "dpo_training_changed_adapter": changed,
        "validation_losses": {"sft_reference": initial_val, "dpo_candidate": candidate_val},
        "models": results,
        "truncated_response_counts": {
            r: sum(e[side]["truncated"] for e in rows for side in ["chosen", "rejected"])
            for r, rows in encoded.items()
        },
        "examples": [
            {"prompt": row["prompt"], **{stage: values[i] for stage, values in generations.items()}}
            for i, row in enumerate(selected["test"][:6])
        ],
        "limitations": "Small preference/SFT budgets, one seed and synthetic preference labels. Pairwise reference likelihood and DPO margins are not generated-response win rates or human-validated quality. Long responses are truncated; identical truncated pairs are excluded. The SFT checkpoint is retained when validation does not favor DPO.",
        "seconds": time.perf_counter() - started,
        "runtime": runtime(["torch", "transformers", "peft", "pyarrow"]),
        "source": source_identity(
            globals().get("__file__"), globals().get("PROJECT_SOURCE_SHA256")
        ),
    }
    write_json(output / "metrics.json", report)
    return report
