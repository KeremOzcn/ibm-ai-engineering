"""Behavior regressions for the selected portfolio experiments."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from projects.common import clean_image_splits  # noqa: E402

TORCH = importlib.util.find_spec("torch") is not None
TF = importlib.util.find_spec("tensorflow") is not None
PIL = importlib.util.find_spec("PIL") is not None
PEFT = importlib.util.find_spec("peft") is not None


class SelectedDataTests(unittest.TestCase):
    def test_duplicate_cleanup_preserves_test_and_excludes_label_conflicts(self):
        rows = [
            {"file": "train/a", "role": "train", "label": 0, "pixel_sha256": "same"},
            {"file": "test/a", "role": "test", "label": 0, "pixel_sha256": "same"},
            {"file": "train/b", "role": "train", "label": 0, "pixel_sha256": "conflict"},
            {"file": "test/b", "role": "test", "label": 1, "pixel_sha256": "conflict"},
        ]
        kept, removed = clean_image_splits(rows)
        self.assertEqual([row["file"] for row in kept], ["test/a"])
        self.assertEqual(len(removed), 3)

    @unittest.skipUnless(TORCH, "PyTorch is not installed")
    def test_normalized_instruction_groups_never_cross_partitions(self):
        from projects.instruction_tuning.project import load_records

        rows = [{"instruction": f"Task {i}", "input": "", "output": "answer"} for i in range(20)]
        rows += [{"instruction": "  TASK   1 ", "input": "", "output": "different answer"}]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows))
            parts = load_records(path, 42)
        groups = {role: {r["group"] for r in rows} for role, rows in parts.items()}
        self.assertFalse(groups["train"] & groups["validation"])
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["validation"] & groups["test"])
        self.assertEqual(sum(map(len, parts.values())), 21)


@unittest.skipUnless(TORCH, "PyTorch is not installed")
class SelectedTorchTests(unittest.TestCase):
    def test_offline_vgg_uses_verified_explicit_path_and_rejects_corruption(self):
        import hashlib

        from projects.aircraft import project

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "weights.h5"
            path.write_bytes(b"fixture weights")
            expected = hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()
            with patch.object(project, "VGG_NOTOP_MD5", expected):
                self.assertEqual(project.vgg_weight_source(path, False), str(path))
                path.write_bytes(b"corrupt")
                with self.assertRaisesRegex(ValueError, "checksum"):
                    project.vgg_weight_source(path, False)
            with self.assertRaises(FileNotFoundError):
                project.vgg_weight_source(Path(folder) / "missing.h5", False)

    def test_scene_splits_keep_all_tiles_from_a_scene_together(self):
        from projects.satellite.project import assign_scene_splits

        rows = [
            {"file": f"class{label}/S2_{scene}.SAFE_{label}.jpg", "label": label}
            for scene in ["T32UPE", "T31UDQ", "T32UQB"]
            for label in [0, 1]
        ]
        split = assign_scene_splits(rows)
        self.assertEqual({r["role"] for r in split if "T32UPE" in r["file"]}, {"test"})
        self.assertEqual({r["role"] for r in split if "T31UDQ" in r["file"]}, {"validation"})
        with self.assertRaises(ValueError):
            assign_scene_splits(rows, test_scene="missing")

    def test_hybrid_retains_sixteen_spatial_tokens_and_backpropagates(self):
        import torch

        from projects.satellite.project import SpatialHybrid

        torch.set_num_threads(1)
        model = SpatialHybrid()
        images = torch.randn(2, 3, 64, 64)
        self.assertEqual(tuple(model.tokens(images).shape), (2, 17, 64))
        loss = torch.nn.functional.cross_entropy(model(images), torch.tensor([0, 1]))
        loss.backward()
        self.assertIsNotNone(model.stem[0].weight.grad)
        self.assertIsNotNone(model.encoder.layers[0].self_attn.in_proj_weight.grad)

    def test_metrics_count_processed_samples_not_declared_dataset_size(self):
        import torch

        from projects.satellite.project import evaluate

        class Loader:
            dataset = range(100)

            def __iter__(self):
                yield torch.tensor([[20.0, -20.0], [-20.0, 20.0]]), torch.tensor([0, 1]), ["a", "b"]

        result = evaluate(torch.nn.Identity(), Loader(), "cpu")
        self.assertEqual(result["samples"], 2)
        self.assertEqual(result["accuracy"], 1.0)

    def test_checkpoint_restore_replaces_a_worse_final_epoch(self):
        import torch

        from projects.satellite.project import restore

        model = torch.nn.Linear(2, 2)
        with torch.no_grad():
            model.weight.fill_(1)
        expected = model.weight.detach().clone()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "selected.pt"
            torch.save(model.state_dict(), path)
            with torch.no_grad():
                model.weight.fill_(99)
            restore(model, path, "cpu")
        torch.testing.assert_close(model.weight, expected, rtol=0, atol=0)

    def test_response_mask_keeps_real_eos_even_when_pad_equals_eos(self):
        from projects.instruction_tuning.project import collate, encode_response

        class Tokenizer:
            eos_token_id = 0

            def encode(self, text, add_special_tokens):
                return [1, 2, 3] if add_special_tokens else [4, 5]

        row = encode_response({"instruction": "task", "output": "answer"}, Tokenizer(), 10)
        batch = collate([row, {"input_ids": [1, 4, 0], "labels": [-100, 4, 0]}], 0)
        self.assertEqual(batch["labels"][0].tolist(), [-100, -100, -100, 4, 5, 0])
        self.assertEqual(batch["labels"][1].tolist(), [-100, 4, 0, -100, -100, -100])
        self.assertEqual(batch["attention_mask"][1].tolist(), [1, 1, 1, 0, 0, 0])

    def test_truncated_response_does_not_teach_an_artificial_eos(self):
        from projects.instruction_tuning.project import encode_response

        class Tokenizer:
            eos_token_id = 0

            def encode(self, text, add_special_tokens):
                return [1, 2, 3] if add_special_tokens else [4, 5, 6, 7]

        row = encode_response({"instruction": "task", "output": "answer"}, Tokenizer(), 5)
        self.assertEqual(row["input_ids"], [1, 2, 3, 4, 5])
        self.assertTrue(row["response_truncated"])

    def test_response_loss_is_weighted_by_supervised_tokens(self):
        import torch

        from projects.instruction_tuning.project import response_nll

        class Model(torch.nn.Module):
            def forward(self, input_ids, **kwargs):
                return SimpleNamespace(loss=torch.tensor(2.0 if len(input_ids) == 2 else 10.0))

        examples = [
            {"input_ids": [0, 1], "labels": [-100, 1]},
            {"input_ids": [0, 1, 2, 3], "labels": [-100, 1, 2, 3]},
            {"input_ids": [0, 1], "labels": [-100, 1]},
        ]
        result = response_nll(Model(), examples, 0, "cpu")
        self.assertAlmostEqual(result["nll"], 3.6)
        self.assertEqual(result["tokens"], 5)

    def test_caption_generation_errors_propagate(self):
        import torch

        from projects.aircraft.project import caption_image

        class BrokenModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))

            def generate(self, **kwargs):
                raise RuntimeError("model failed")

        def processor(**kwargs):
            return {"pixel_values": torch.zeros(1, 3, 4, 4)}

        with self.assertRaisesRegex(RuntimeError, "model failed"):
            caption_image(processor, BrokenModel(), object())


@unittest.skipUnless(TF, "TensorFlow is not installed")
class SelectedVGGTests(unittest.TestCase):
    def test_vgg_preprocessing_centers_bgr_and_preserves_display_rgb(self):
        import numpy as np

        from projects.aircraft.project import preprocess_rgb

        image = np.array([[[[10.0, 20.0, 30.0]]]], dtype=np.float32)
        result = preprocess_rgb(image)
        np.testing.assert_allclose(
            result.ravel(), [30 - 103.939, 20 - 116.779, 10 - 123.68], atol=1e-5
        )
        np.testing.assert_array_equal(image.ravel(), [10, 20, 30])


@unittest.skipUnless(PIL and TORCH, "Image/PyTorch dependencies are not installed")
class SelectedImageTests(unittest.TestCase):
    def test_feature_extraction_includes_the_final_incomplete_batch(self):
        from PIL import Image

        from projects.aircraft.project import extract_features

        sizes = []

        def backbone(batch, training):
            sizes.append(len(batch))
            return batch[:, :1, :1, :1].reshape(len(batch), 1)

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = []
            for i in range(17):
                Image.new("RGB", (2, 2), (i, i, i)).save(root / f"{i}.png")
                rows.append({"file": f"{i}.png"})
            result = extract_features(root, rows, backbone, lambda value: value, 8)
        self.assertEqual(sizes, [8, 8, 1])
        self.assertEqual(result.ravel().tolist(), list(range(17)))


@unittest.skipUnless(PEFT and TORCH, "PEFT is not installed")
class SelectedPeftTests(unittest.TestCase):
    def test_real_lora_update_and_fresh_checkpoint_reload(self):
        import copy

        import torch
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model
        from transformers import OPTConfig, OPTForCausalLM

        from projects.instruction_tuning.project import trainable_digest

        torch.manual_seed(42)
        config = OPTConfig(
            vocab_size=32,
            hidden_size=16,
            word_embed_proj_dim=16,
            ffn_dim=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=32,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
        base = OPTForCausalLM(config)
        initial = copy.deepcopy(base.state_dict())
        model = get_peft_model(
            base,
            LoraConfig(
                r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], task_type=TaskType.CAUSAL_LM
            ),
        )
        before = trainable_digest(model)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
        inputs = torch.tensor([[1, 3, 4, 5, 2]])
        labels = torch.tensor([[-100, -100, 4, 5, 2]])
        loss = model(input_ids=inputs, labels=labels).loss
        loss.backward()
        optimizer.step()
        self.assertNotEqual(trainable_digest(model), before)
        with tempfile.TemporaryDirectory() as folder:
            model.save_pretrained(folder)
            fresh = OPTForCausalLM(config)
            fresh.load_state_dict(initial)
            restored = PeftModel.from_pretrained(fresh, folder).eval()
            model.eval()
            with torch.no_grad():
                torch.testing.assert_close(
                    model(inputs).logits, restored(inputs).logits, rtol=1e-5, atol=1e-5
                )


if __name__ == "__main__":
    unittest.main()
