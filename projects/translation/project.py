"""Compare fresh German-to-English models using the same paired data and epoch budget."""

from __future__ import annotations

import collections
import hashlib
import math
import random
import re
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from projects.common import file_sha256, runtime, source_identity, write_json

PAD, BOS, EOS, UNK = 0, 1, 2, 3


def tokenize(text: str) -> list[str]:
    """Use the same deterministic Unicode word/punctuation tokenizer in both models."""
    return re.findall(r"\w+|[^\w\s]", text.casefold())


def load_pairs(root: Path, maximum_tokens: int = 48) -> tuple[dict, dict]:
    """Reserve official splits and exclude repeated normalized source sentences across roles."""
    parts, seen, audit = {}, set(), {"excluded": collections.Counter()}
    for role, stem in [("test", "test_2016_flickr"), ("validation", "val"), ("train", "train")]:
        german = (root / f"{stem}.de").read_text(encoding="utf-8").splitlines()
        english = (root / f"{stem}.en").read_text(encoding="utf-8").splitlines()
        if len(german) != len(english):
            raise ValueError("Parallel files have different lengths.")
        rows = []
        for source, reference in zip(german, english, strict=True):
            src, tgt = tokenize(source), tokenize(reference)
            group = hashlib.sha256(" ".join(src).encode()).hexdigest()
            if not src or not tgt or max(len(src), len(tgt)) > maximum_tokens:
                audit["excluded"]["empty_or_over_length"] += 1
                continue
            if group in seen:
                audit["excluded"]["repeated_source_sentence"] += 1
                continue
            seen.add(group)
            rows.append(
                {"id": group, "source": source, "reference": reference, "src": src, "tgt": tgt}
            )
        if not rows:
            raise ValueError(f"No retained {role} pairs.")
        parts[role] = rows
    audit["excluded"] = dict(audit["excluded"])
    audit["input_files"] = {p.name: file_sha256(p) for p in root.glob("*.de")} | {
        p.name: file_sha256(p) for p in root.glob("*.en")
    }
    return parts, audit


def vocabulary(rows: list[dict], key: str, cap: int = 8000) -> tuple[dict, list]:
    """Fit token indices exclusively on the training records used in this run."""
    counts = collections.Counter(token for row in rows for token in row[key])
    tokens = ["<pad>", "<bos>", "<eos>", "<unk>"] + [
        word for word, _ in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[: cap - 4]
    ]
    return {word: i for i, word in enumerate(tokens)}, tokens


def batchify(rows: list[dict], source_vocab: dict, target_vocab: dict):
    arrays = []
    for key, vocab in [("src", source_vocab), ("tgt", target_vocab)]:
        values = [
            torch.tensor([BOS] + [vocab.get(t, UNK) for t in row[key]] + [EOS]) for row in rows
        ]
        arrays.append(nn.utils.rnn.pad_sequence(values, batch_first=True, padding_value=PAD))
    return arrays[0], arrays[1], rows


class RecurrentTranslator(nn.Module):
    """A context-vector GRU encoder/decoder with parallel teacher-forced training."""

    def __init__(self, source_size: int, target_size: int, width: int = 128):
        super().__init__()
        self.source_embedding = nn.Embedding(source_size, width, padding_idx=PAD)
        self.target_embedding = nn.Embedding(target_size, width, padding_idx=PAD)
        self.encoder = nn.GRU(width, width, batch_first=True)
        self.decoder = nn.GRU(width * 2, width, batch_first=True)
        self.head = nn.Linear(width, target_size)

    def forward(self, source, target):
        lengths = source.ne(PAD).sum(1).cpu()
        embedded = nn.utils.rnn.pack_padded_sequence(
            self.source_embedding(source), lengths, batch_first=True, enforce_sorted=False
        )
        _, context = self.encoder(embedded)
        target_emb = self.target_embedding(target)
        repeated = context[-1].unsqueeze(1).expand(-1, target.shape[1], -1)
        output, _ = self.decoder(torch.cat([target_emb, repeated], -1), context)
        return self.head(output)


class TransformerTranslator(nn.Module):
    """A causal two-layer encoder/decoder; padding is masked in attention."""

    def __init__(self, source_size: int, target_size: int, width: int = 128):
        super().__init__()
        self.source_embedding = nn.Embedding(source_size, width, padding_idx=PAD)
        self.target_embedding = nn.Embedding(target_size, width, padding_idx=PAD)
        self.transformer = nn.Transformer(
            d_model=width,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
        )
        positions = torch.arange(64)[:, None]
        scale = torch.exp(torch.arange(0, width, 2) * (-math.log(10000.0) / width))
        encoding = torch.zeros(64, width)
        encoding[:, 0::2] = torch.sin(positions * scale)
        encoding[:, 1::2] = torch.cos(positions * scale)
        self.register_buffer("positions", encoding)
        self.head = nn.Linear(width, target_size)
        self.width = width

    def forward(self, source, target):
        mask = torch.ones(
            target.shape[1], target.shape[1], device=target.device, dtype=torch.bool
        ).triu(1)
        src = (
            self.source_embedding(source) * math.sqrt(self.width)
            + self.positions[: source.shape[1]]
        )
        tgt = (
            self.target_embedding(target) * math.sqrt(self.width)
            + self.positions[: target.shape[1]]
        )
        output = self.transformer(
            src,
            tgt,
            tgt_mask=mask,
            src_key_padding_mask=source.eq(PAD),
            tgt_key_padding_mask=target.eq(PAD),
            memory_key_padding_mask=source.eq(PAD),
        )
        return self.head(output)


def token_loss(model, loader, device) -> dict:
    """Weight reference loss by actual non-padding next-token targets."""
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for src, tgt, _ in loader:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src, tgt[:, :-1])
            total += float(
                nn.functional.cross_entropy(
                    output.reshape(-1, output.shape[-1]),
                    tgt[:, 1:].reshape(-1),
                    ignore_index=PAD,
                    reduction="sum",
                )
            )
            count += int(tgt[:, 1:].ne(PAD).sum())
    return {"nll": total / count, "tokens": count}


def translations(model, loader, device, words: list[str]) -> tuple[list[str], list[dict]]:
    """Decode freely, with identical greedy settings and a 49-token generation cap."""
    model.eval()
    hypotheses = []
    records = []
    with torch.no_grad():
        for src, _, rows in loader:
            src = src.to(device)
            generated = torch.full((len(src), 1), BOS, device=device, dtype=torch.long)
            finished = torch.zeros(len(src), dtype=torch.bool, device=device)
            for _ in range(49):
                output = model(src, generated)[:, -1].argmax(-1)
                output = torch.where(finished, torch.full_like(output, EOS), output)
                generated = torch.cat([generated, output[:, None]], 1)
                finished |= output.eq(EOS)
                if bool(finished.all()):
                    break
            for row, ids in zip(rows, generated[:, 1:].cpu().tolist(), strict=True):
                if EOS in ids:
                    ids = ids[: ids.index(EOS)]
                text = " ".join(words[i] for i in ids if i not in [PAD, BOS])
                hypotheses.append(text)
                records.append({"id": row["id"], "prediction": text})
    return hypotheses, records


def run(
    data_root: Path, output: Path, *, full: bool = False, seed: int = 42, epochs: int = 6
) -> dict:
    """Use fresh matched-budget models and reserve test generation until checkpoint selection."""
    from functools import partial

    from sacrebleu.metrics import BLEU

    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parts, audit = load_pairs(data_root)
    selected = {}
    for role, rows in parts.items():
        selected[role] = list(rows)
        random.Random(seed).shuffle(selected[role])
        if not full:
            selected[role] = selected[role][: 512 if role == "train" else 64]
    manifest = {
        "partitions": {role: [row["id"] for row in rows] for role, rows in selected.items()},
        "audit": audit,
    }
    write_json(output / "split-manifest.json", manifest)
    source_vocab, source_words = vocabulary(selected["train"], "src")
    target_vocab, target_words = vocabulary(selected["train"], "tgt")
    collator = partial(batchify, source_vocab=source_vocab, target_vocab=target_vocab)
    validation = DataLoader(selected["validation"], batch_size=64, collate_fn=collator)
    test = DataLoader(selected["test"], batch_size=32, collate_fn=collator)
    torch.manual_seed(seed)
    shared_src = nn.Embedding(len(source_vocab), 128, padding_idx=PAD).state_dict()
    shared_tgt = nn.Embedding(len(target_vocab), 128, padding_idx=PAD).state_dict()
    started = time.perf_counter()
    results = {}
    for name, factory in [("gru", RecurrentTranslator), ("transformer", TransformerTranslator)]:
        torch.manual_seed(seed)
        model = factory(len(source_vocab), len(target_vocab)).to(device)
        model.source_embedding.load_state_dict(shared_src)
        model.target_embedding.load_state_dict(shared_tgt)
        train = DataLoader(
            selected["train"],
            batch_size=64,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
            collate_fn=collator,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
        path = output / name / "best.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        best = float("inf")
        history = []
        for epoch in range(epochs if full else 2):
            model.train()
            processed = 0
            for src, tgt, rows in train:
                src, tgt = src.to(device), tgt.to(device)
                logits = model(src, tgt[:, :-1])
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), tgt[:, 1:].reshape(-1), ignore_index=PAD
                )
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                processed += len(rows)
            score = token_loss(model, validation, device)
            history.append(
                {"epoch": epoch + 1, "train_samples": processed, "validation_nll": score["nll"]}
            )
            print(name, "epoch", epoch + 1, "validation_nll", score["nll"], flush=True)
            if score["nll"] < best:
                best = score["nll"]
                torch.save(model.state_dict(), path)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        fresh = factory(len(source_vocab), len(target_vocab)).to(device)
        fresh.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        fresh.eval()
        probe = next(iter(validation))
        src, tgt = probe[0][:2].to(device), probe[1][:2, :3].to(device)
        with torch.no_grad():
            torch.testing.assert_close(model(src, tgt), fresh(src, tgt), rtol=0, atol=0)
        del fresh
        hypotheses, predictions = translations(model, test, device, target_words)
        bleu = BLEU(lowercase=True, tokenize="13a")
        score = bleu.corpus_score(hypotheses, [[r["reference"] for r in selected["test"]]])
        write_json(output / f"{name}-predictions.json", predictions)
        results[name] = {
            "test_samples": len(hypotheses),
            "corpus_bleu": float(score.score),
            "bleu_signature": str(bleu.get_signature()),
            "test_response_loss": token_loss(model, test, device),
            "parameters": sum(p.numel() for p in model.parameters()),
            "history": history,
            "checkpoint_sha256": file_sha256(path),
            "exact_reload": True,
            "examples": [
                {"source": r["source"], "reference": r["reference"], "prediction": h}
                for r, h in list(zip(selected["test"], hypotheses, strict=True))[:6]
            ],
        }
    report = {
        "project": "translation",
        "profile": "complete-retained-splits" if full else "bounded-pilot",
        "seed": seed,
        "device": str(device),
        "complete_split_counts": {r: len(v) for r, v in parts.items()},
        "selected_split_counts": {r: len(v) for r, v in selected.items()},
        "split_manifest_sha256": file_sha256(output / "split-manifest.json"),
        "models": results,
        "vocabulary_sizes": {"source": len(source_words), "target": len(target_words)},
        "comparison": "Same retained pairs, initial embeddings, example order, learning rate and epochs; model capacities and computation differ. Not an isolated causal test of attention.",
        "limitations": "One seed, short-caption domain, capped vocabulary and six-epoch budget. Case-insensitive corpus BLEU is not a general translation quality or human preference benchmark.",
        "seconds": time.perf_counter() - started,
        "runtime": runtime(["torch", "sacrebleu", "numpy"]),
        "source": source_identity(
            globals().get("__file__"), globals().get("PROJECT_SOURCE_SHA256")
        ),
    }
    write_json(output / "metrics.json", report)
    return report
