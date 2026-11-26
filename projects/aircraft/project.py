"""Train a VGG16 feature classifier and inspect separate BLIP descriptions."""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from pathlib import Path

import numpy as np

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

VGG_NOTOP_MD5 = "6d6bbae143d832006294945121d1f1fc"


def vgg_weight_source(weights: Path, allow_downloads: bool) -> str:
    """Use a verified explicit file path so offline loading cannot invoke get_file."""
    if not weights.is_file():
        if allow_downloads:
            return "imagenet"
        raise FileNotFoundError("Cache the ImageNet VGG16 weights or explicitly allow downloads.")
    digest = hashlib.md5(usedforsecurity=False)
    with weights.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != VGG_NOTOP_MD5:
        raise ValueError("Cached VGG16 weights do not match the published Keras checksum.")
    return str(weights)


def rgb_array(path: Path, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Return displayable, unscaled RGB; preprocessing is a separate model operation."""
    from PIL import Image, ImageOps

    with Image.open(path) as image:
        return np.asarray(
            ImageOps.exif_transpose(image).convert("RGB").resize(size), dtype=np.float32
        )


def preprocess_rgb(images: np.ndarray) -> np.ndarray:
    """Apply the pretrained VGG16 contract while preserving RGB display inputs."""
    from tensorflow.keras.applications.vgg16 import preprocess_input

    return preprocess_input(np.asarray(images, dtype=np.float32).copy())


def extract_features(
    root: Path, rows: list[dict], backbone, preprocess, batch_size: int = 8
) -> np.ndarray:
    """Include the final incomplete batch, preserving record order."""
    features = []
    for start in range(0, len(rows), batch_size):
        batch = np.stack(
            [rgb_array(root / row["file"]) for row in rows[start : start + batch_size]]
        )
        features.append(np.asarray(backbone(preprocess(batch.copy()), training=False)))
    if not features:
        raise ValueError("Cannot extract an empty split.")
    return np.concatenate(features)


def caption_image(processor, model, image, max_new_tokens: int = 32) -> str:
    """Run the captioner directly; failures propagate instead of becoming fake captions."""
    import torch

    model.eval()
    inputs = processor(images=image, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    caption = processor.decode(output[0], skip_special_tokens=True).strip()
    if not caption:
        raise ValueError("Captioner returned an empty description.")
    return caption


def run(
    data_root: Path,
    output: Path,
    *,
    full: bool = False,
    allow_downloads: bool = False,
    seed: int = 42,
    epochs: int = 10,
) -> dict:
    """Use the existing source splits after exact-pixel leakage/label checks."""
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    import tensorflow as tf
    from PIL import Image, ImageOps
    from transformers import BlipForConditionalGeneration, BlipProcessor

    tf.keras.utils.set_random_seed(seed)
    rows = index_images(
        data_root, ["crack", "dent"], {"train": "train", "validation": "valid", "test": "test"}
    )
    rows, removed = clean_image_splits(rows)
    write_json(output / "split-manifest.json", {"records": rows, "removed": removed})
    groups = {
        role: [row for row in rows if row["role"] == role]
        for role in ["train", "validation", "test"]
    }
    if any({r["label"] for r in group} != {0, 1} for group in groups.values()):
        raise ValueError("Each cleaned split must contain both crack and dent images.")
    selected = {
        role: balanced_subset(group, None if full else (8 if role == "train" else 4), seed)
        for role, group in groups.items()
    }
    weights = (
        Path(os.environ.get("KERAS_HOME", str(Path.home() / ".keras")))
        / "models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
    )
    weight_source = vgg_weight_source(weights, allow_downloads)
    started = time.perf_counter()
    backbone = tf.keras.applications.VGG16(include_top=False, weights=weight_source, pooling="avg")
    backbone.trainable = False
    feature_arrays = {
        role: extract_features(data_root, group, backbone, preprocess_rgb)
        for role, group in selected.items()
    }
    labels = {
        role: np.asarray([r["label"] for r in group], dtype=np.float32)
        for role, group in selected.items()
    }
    head = tf.keras.Sequential(
        [
            tf.keras.layers.Input((512,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    head.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    )
    history, best_loss = [], float("inf")
    checkpoint = output / "best-classifier.keras"
    rng = np.random.default_rng(seed)
    for epoch in range(epochs if full else 2):
        order = rng.permutation(len(selected["train"]))
        for start in range(0, len(order), 8):
            indices = order[start : start + 8]
            head.train_on_batch(feature_arrays["train"][indices], labels["train"][indices, None])
        logits = head(feature_arrays["validation"], training=False)
        loss = float(
            tf.keras.losses.binary_crossentropy(
                labels["validation"][:, None], logits, from_logits=True
            )
            .numpy()
            .mean()
        )
        history.append({"epoch": epoch + 1, "validation_loss": loss})
        if loss < best_loss:
            best_loss = loss
            head.save(checkpoint)
    head = tf.keras.models.load_model(checkpoint)
    expected = np.asarray(head(feature_arrays["validation"], training=False))
    reloaded = tf.keras.models.load_model(checkpoint)
    np.testing.assert_array_equal(
        expected, np.asarray(reloaded(feature_arrays["validation"], training=False))
    )
    probabilities = (
        tf.sigmoid(head(feature_arrays["test"], training=False)).numpy().ravel().tolist()
    )
    metrics = classification_metrics(labels["test"].astype(int).tolist(), probabilities)
    model_id = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_id, local_files_only=not allow_downloads)
    captioner = BlipForConditionalGeneration.from_pretrained(
        model_id, local_files_only=not allow_downloads
    )
    captions = []
    for label in [0, 1]:
        index = next(i for i, row in enumerate(selected["test"]) if row["label"] == label)
        row = selected["test"][index]
        with Image.open(data_root / row["file"]) as image:
            text = caption_image(
                processor, captioner, ImageOps.exif_transpose(image).convert("RGB")
            )
        captions.append(
            {
                "file": row["file"],
                "label": ["crack", "dent"][label],
                "classifier_prediction": ["crack", "dent"][int(probabilities[index] >= 0.5)],
                "caption": text,
                "visible_defect_supported": None,
                "hallucination_review": None,
            }
        )
    report = {
        "project": "aircraft",
        "profile": "full" if full else "bounded-pilot",
        "seed": seed,
        "class_names": ["crack", "dent"],
        "complete_split_counts": split_counts(rows),
        "selected_split_counts": split_counts([r for group in selected.values() for r in group]),
        "split_manifest_sha256": file_sha256(output / "split-manifest.json"),
        "vgg_weights_sha256": file_sha256(weights),
        "preprocessing": "RGB 0..255 -> VGG BGR channel centering; no 1/255 rescale",
        "checkpoint_sha256": file_sha256(checkpoint),
        "exact_reload": True,
        "test": {**metrics, "ids": [r["file"] for r in selected["test"]]},
        "history": history,
        "caption_model": model_id,
        "caption_revision": getattr(captioner.config, "_commit_hash", None),
        "captions": captions,
        "caption_quality": "Not evaluated; human review fields remain null.",
        "near_duplicate_or_source_group_leakage_audited": False,
        "seconds": time.perf_counter() - started,
        "source": source_identity(
            globals().get("__file__"), globals().get("PROJECT_SOURCE_SHA256")
        ),
        "runtime": runtime(["tensorflow", "transformers", "torch", "numpy", "Pillow"]),
    }
    write_json(output / "metrics.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path(".lab-cache/selected-projects/aircraft")
    )
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--allow-downloads", action="store_true")
    args = parser.parse_args()
    report = run(args.data_root, args.output, full=args.full, allow_downloads=args.allow_downloads)
    print(report["test"])


if __name__ == "__main__":
    main()
