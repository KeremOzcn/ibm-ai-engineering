"""Exercise behavior regressions; optional framework dependencies are explicit."""

from __future__ import annotations

import ast
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def answer(filename: str, cell: int) -> str:
    """Read the actual guarded answer used to build the worked notebook."""
    patches = json.loads((ROOT / "provenance/solution-patches.json").read_text(encoding="utf-8"))
    matches = [
        row["solution"]
        for row in patches
        if Path(row["reference"]).name == filename and row["cell"] == cell
    ]
    if len(matches) != 1:
        raise AssertionError(f"Expected one answer: {filename}:{cell}")
    return matches[0]


class ExerciseAnswerTests(unittest.TestCase):
    """Test observable answer behavior rather than searching for expected strings."""

    @unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow is not installed")
    def test_text_transformer_cannot_read_future_tokens(self) -> None:
        import numpy as np
        import tensorflow as tf

        namespace = {"tf": tf, "np": np}
        exec(answer("lab-implementing-transformers-for-text-generation.ipynb", 16), namespace)
        model = namespace["TransformerModel"](7, 4, 1, 8, 1, 4)
        first = model(tf.constant([[1, 2, 3, 4]]), training=False).numpy()
        changed_future = model(tf.constant([[1, 2, 6, 5]]), training=False).numpy()
        np.testing.assert_allclose(first[:, :2], changed_future[:, :2], atol=1e-6)

    @unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow is not installed")
    def test_text_generation_samples_the_last_position(self) -> None:
        import numpy as np
        import tensorflow as tf

        class Vectorizer:
            def __call__(self, value):
                return tf.constant([[1, 2, 3]])

            def get_vocabulary(self):
                return ["pad", "first", "last", "other"]

        def model(inputs, training=False):
            return tf.constant(
                [[[0.0, 10000.0, 0.0, 0.0], [0.0, 0.0, 0.0, 10000.0], [0.0, 0.0, 10000.0, 0.0]]]
            )

        node = next(
            node
            for node in ast.parse(
                answer("lab-implementing-transformers-for-text-generation.ipynb", 24)
            ).body
            if isinstance(node, ast.FunctionDef) and node.name == "generate_text"
        )
        namespace = {"np": np, "tf": tf, "vectorizer": Vectorizer(), "seq_length": 3}
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), "<actual-generation-answer>", "exec"),
            namespace,
        )
        self.assertEqual(namespace["generate_text"](model, "seed", num_generate=1), "seed last")

    @unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow is not installed")
    def test_keras_checkpoint_answer_uses_supported_callback_attributes(self) -> None:
        from tensorflow.keras.callbacks import ModelCheckpoint

        namespace = {"ModelCheckpoint": ModelCheckpoint, "model_name": "test-model.keras"}
        exec(answer("lab-train-and-evaluate-a-keras-based-classifier.ipynb", 77), namespace)
        self.assertEqual(namespace["checkpoint_cb"].monitor, "val_accuracy")
        self.assertTrue(namespace["checkpoint_cb"].save_best_only)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Matplotlib is not installed")
    def test_vision_comparison_plots_accept_different_actual_history_lengths(self) -> None:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        namespace = {
            "plt": plt,
            "epochs": 50,
            "te_loss_all": [0.8, 0.5],
            "te_loss_all_test": [0.7, 0.4, 0.3],
        }
        exec(answer("lab-vision-transformers-using-pytorch.ipynb", 66), namespace)
        lines = plt.gca().lines
        self.assertEqual([len(line.get_xdata()) for line in lines], [2, 3])
        plt.close("all")

    @unittest.skipUnless(importlib.util.find_spec("torchvision"), "torchvision is not installed")
    def test_fashion_transform_defines_the_display_and_classifier_size(self) -> None:
        from PIL import Image
        from torchvision import transforms

        namespace = {"transforms": transforms}
        exec(answer("fashion-mnist-project.ipynb", 20), namespace)
        transformed = namespace["composed"](Image.new("L", (28, 28)))
        self.assertEqual(namespace["IMAGE_SIZE"], 16)
        self.assertEqual(tuple(transformed.shape), (1, 16, 16))

    @unittest.skipUnless(
        importlib.util.find_spec("pdfplumber") and importlib.util.find_spec("fpdf"),
        "PDF dependencies are not installed",
    )
    def test_pdf_translation_retains_all_pages_and_sentences(self) -> None:
        import pdfplumber
        from fpdf import FPDF

        calls = []

        def translator(model: object, text: str) -> str:
            calls.append(text)
            return "EN " + text

        namespace = {"translate": translator}
        exec(answer("a-transformer-model-for-language-translation.ipynb", 149), namespace)
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "input.pdf"
            target = Path(folder) / "output.pdf"
            pdf = FPDF()
            for text in ["Erster Satz. Zweiter Satz.", "Andere Seite."]:
                pdf.add_page()
                pdf.set_font("Courier", size=10)
                pdf.cell(0, 10, text)
            pdf.output(source)
            namespace["translate_pdf"](source, object(), target)
            with pdfplumber.open(target) as result:
                self.assertEqual(len(result.pages), 2)
                self.assertIn("Erster", result.pages[0].extract_text())
                self.assertIn("Andere", result.pages[1].extract_text())
            self.assertEqual(len(calls), 3)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
    def test_fine_tuning_unfreezes_only_requested_layers(self) -> None:
        import torch
        from torch import nn

        class Fixture(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.transformer_encoder = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(100, 2), 2
                )
                self.classifier = nn.Linear(100, 4)

        model = Fixture()
        for parameter in model.parameters():
            parameter.requires_grad = False
        namespace = {"model_finetune_some_layers": model, "nn": nn, "device": torch.device("cpu")}
        exec(answer("pre-training-and-fine-tuning-with-pytorch.ipynb", 123), namespace)
        self.assertEqual(model.classifier.out_features, 3)
        for name, parameter in model.named_parameters():
            self.assertEqual(
                parameter.requires_grad, "linear2" in name or name.startswith("classifier.")
            )

    def test_policy_retrieval_does_not_replace_movie_corpus(self) -> None:
        from types import SimpleNamespace

        class Store:
            records = []

            @classmethod
            def from_documents(cls, documents, embedding, collection_name):
                cls.records.append((list(documents), collection_name))
                return cls()

            def as_retriever(self, search_kwargs):
                return SimpleNamespace(
                    invoke=lambda query: [SimpleNamespace(page_content="policy", metadata={})]
                )

        movie = SimpleNamespace(page_content="dreams", metadata={"director": "Christopher Nolan"})
        namespace = {
            "Chroma": Store,
            "chunks_txt": ["policy chunk"],
            "watsonx_embedding": lambda: object(),
            "movie_docs": [movie],
        }
        exec(
            answer("develop-a-retriever-to-fetch-document-segments-based-on-queries.ipynb", 114),
            namespace,
        )
        self.assertEqual(namespace["movie_docs"], [movie])
        self.assertEqual(Store.records[0], (["policy chunk"], "policy_exercise"))


if __name__ == "__main__":
    unittest.main()
