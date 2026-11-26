"""Train a fresh CNN and a genuinely spatial CNN-Transformer on scene-separated images."""

from __future__ import annotations

import argparse
import copy
import hashlib
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset

from projects.common import (
    balanced_subset,
    classification_metrics,
    clean_image_splits,
    file_sha256,
    index_images,
    runtime,
    source_identity,
    split_counts,
    write_json,
)


def scene_id(filename: str) -> str:
    """Use the complete source scene, excluding the final tile number."""
    return re.sub(r"_\d+\.[^.]+$", "", Path(filename).name)


def assign_scene_splits(
    rows: list[dict], test_scene: str = "T32UPE", validation_scene: str = "T31UDQ"
) -> list[dict]:
    """Reserve whole source-scene groups; never fall back silently to a random split."""
    if test_scene == validation_scene:
        raise ValueError("Validation and test scenes must differ.")
    result = []
    for row in rows:
        scene = scene_id(row["file"])
        role = (
            "test"
            if test_scene in scene
            else "validation"
            if validation_scene in scene
            else "train"
        )
        result.append({**row, "scene": scene, "role": role})
    for role in ["train", "validation", "test"]:
        if {row["label"] for row in result if row["role"] == role} != {0, 1}:
            raise ValueError(f"{role} must contain both classes; inspect the chosen scenes.")
    return result


class SatelliteDataset(Dataset):
    """Deterministic augmentation per image/epoch, independent of model dropout RNG."""

    def __init__(self, root: Path, rows: list[dict], training: bool, seed: int):
        self.root, self.rows, self.training, self.seed = root, rows, training, seed
        self.epoch = 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row["file"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB").resize((64, 64))
        token = f"{self.seed}:{self.epoch}:{row['file']}".encode()
        if self.training and hashlib.sha256(token).digest()[0] < 128:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        values = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(values.transpose(2, 0, 1).copy()), row["label"], row["file"]


class Stem(nn.Sequential):
    """Preserve an 8x8 feature map for a 64x64 image."""

    def __init__(self):
        layers = []
        for before, after in [(3, 16), (16, 32), (32, 64)]:
            layers.extend(
                [
                    nn.Conv2d(before, after, 3, stride=2, padding=1),
                    nn.GroupNorm(4, after),
                    nn.GELU(),
                ]
            )
        super().__init__(*layers)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = Stem()
        self.head = nn.Linear(64, 2)

    def forward(self, images):
        return self.head(self.stem(images).mean(dim=(-1, -2)))


class SpatialHybrid(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = Stem()
        self.patch = nn.Conv2d(64, 64, 2, stride=2)
        self.cls = nn.Parameter(torch.zeros(1, 1, 64))
        self.position = nn.Parameter(torch.zeros(1, 17, 64))
        layer = nn.TransformerEncoderLayer(
            64, 4, dim_feedforward=128, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.head = nn.Linear(64, 2)

    def tokens(self, images):
        patches = self.patch(self.stem(images)).flatten(2).transpose(1, 2)
        if patches.shape[1] <= 1:
            raise ValueError("Spatial attention requires multiple image patches.")
        return torch.cat([self.cls.expand(len(images), -1, -1), patches], dim=1) + self.position

    def forward(self, images):
        return self.head(self.encoder(self.tokens(images))[:, 0])


def evaluate(model, loader, device) -> dict:
    """Compute weighted loss/metrics over actual evaluated examples."""
    model.eval()
    labels, probabilities, ids = [], [], []
    loss_sum = 0.0
    with torch.no_grad():
        for images, target, filenames in loader:
            logits = model(images.to(device))
            loss_sum += nn.functional.cross_entropy(
                logits, target.to(device), reduction="sum"
            ).item()
            labels.extend(target.tolist())
            probabilities.extend(logits.softmax(-1)[:, 1].cpu().tolist())
            ids.extend(filenames)
    result = classification_metrics(labels, probabilities)
    return {**result, "loss": loss_sum / len(labels), "ids": ids}


def restore(model, path: Path, device):
    """Restore all weights strictly before final evaluation."""
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
    return model.eval()


def run(
    data_root: Path, output: Path, *, full: bool = False, seed: int = 42, epochs: int = 3
) -> dict:
    """Run a bounded pilot by default; a full run keeps every retained image."""
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = assign_scene_splits(index_images(data_root, ["class_0_non_agri", "class_1_agri"]))
    rows, removed = clean_image_splits(rows)
    write_json(output / "split-manifest.json", {"records": rows, "removed": removed})
    groups = {
        role: [row for row in rows if row["role"] == role]
        for role in ["train", "validation", "test"]
    }
    for role in groups:
        if {r["label"] for r in groups[role]} != {0, 1}:
            raise ValueError(f"Both classes must remain in {role} after cleaning.")
    selected = {
        role: balanced_subset(group, None if full else (16 if role == "train" else 8), seed)
        for role, group in groups.items()
    }
    validation = DataLoader(
        SatelliteDataset(data_root, selected["validation"], False, seed), batch_size=8
    )
    test = DataLoader(SatelliteDataset(data_root, selected["test"], False, seed), batch_size=8)
    torch.manual_seed(seed)
    initial_stem = copy.deepcopy(Stem().state_dict())
    results = {}
    for name, factory in [("cnn", CNN), ("spatial_hybrid", SpatialHybrid)]:
        torch.manual_seed(seed)
        model = factory().to(device)
        model.stem.load_state_dict(initial_stem)
        dataset = SatelliteDataset(data_root, selected["train"], True, seed)
        train = DataLoader(
            dataset, batch_size=8, shuffle=True, generator=torch.Generator().manual_seed(seed)
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        history, visited = [], []
        best_f1 = -1.0
        checkpoint = output / name / "best.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        for epoch in range(epochs if full else 1):
            dataset.epoch = epoch
            model.train()
            total_loss, processed = 0.0, 0
            for images, labels, filenames in train:
                optimizer.zero_grad()
                loss = nn.functional.cross_entropy(model(images.to(device)), labels.to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(labels)
                processed += len(labels)
                visited.extend(filenames)
            score = evaluate(model, validation, device)
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": total_loss / processed,
                    "validation_macro_f1": score["macro_f1"],
                    "train_samples": processed,
                }
            )
            if score["macro_f1"] > best_f1:
                best_f1 = score["macro_f1"]
                torch.save(model.state_dict(), checkpoint)
        restore(model, checkpoint, device)
        probe = next(iter(validation))[0].to(device)
        with torch.no_grad():
            expected = model(probe).cpu()
            actual = restore(factory().to(device), checkpoint, device)(probe).cpu()
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        results[name] = {
            "parameters": sum(p.numel() for p in model.parameters()),
            "spatial_tokens": 16 if name == "spatial_hybrid" else None,
            "seconds": time.perf_counter() - started,
            "history": history,
            "checkpoint_sha256": file_sha256(checkpoint),
            "exact_reload": True,
            "training_ids": visited,
            "test": evaluate(model, test, device),
        }
    if results["cnn"]["training_ids"] != results["spatial_hybrid"]["training_ids"]:
        raise AssertionError("Compared models saw different sample orders.")
    report = {
        "project": "satellite",
        "profile": "full" if full else "bounded-pilot",
        "seed": seed,
        "device": str(device),
        "complete_split_counts": split_counts(rows),
        "selected_split_counts": split_counts([r for group in selected.values() for r in group]),
        "split_manifest_sha256": file_sha256(output / "split-manifest.json"),
        "comparison": "Different parameter counts; shared stem initialization and sample order. Not an isolated attention-only ablation.",
        "near_duplicate_or_geographic_overlap_audited": False,
        "models": results,
        "source": source_identity(
            globals().get("__file__"), globals().get("PROJECT_SOURCE_SHA256")
        ),
        "runtime": runtime(["torch", "numpy", "scikit-learn", "Pillow"]),
    }
    write_json(output / "metrics.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path(".lab-cache/selected-projects/satellite")
    )
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    report = run(args.data_root, args.output, full=args.full)
    print({name: row["test"]["macro_f1"] for name, row in report["models"].items()})


if __name__ == "__main__":
    main()
