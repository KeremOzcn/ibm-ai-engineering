"""Behavior checks for the solution audit and bounded verification profile."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nbformat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import check_solutions  # noqa: E402
import run_solutions  # noqa: E402
from build_solutions import normalize_setup, unresolved  # noqa: E402
from prepare_vision_smoke import prepare  # noqa: E402
from run_solutions import bound_training, training_signatures, verified_previous  # noqa: E402


class SolutionAuditTests(unittest.TestCase):
    """A syntax pass must not accidentally certify a remaining exercise stub."""

    def test_ellipsis_assignment_is_unfinished(self) -> None:
        self.assertTrue(unresolved("prediction = ..."))

    def test_tensor_ellipsis_index_is_valid(self) -> None:
        self.assertEqual(unresolved("values = tensor[..., :3]"), [])

    def test_capture_does_not_hide_python_syntax_error(self) -> None:
        self.assertTrue(unresolved("%%capture\nvalue ="))

    def test_comment_only_answer_is_unfinished(self) -> None:
        self.assertTrue(unresolved("# Write your response here"))

    def test_prompt_text_is_not_a_code_placeholder(self) -> None:
        self.assertEqual(unresolved('prompt = "Explain your code and print ..."'), [])

    def test_multiline_package_install_preserves_following_import(self) -> None:
        actual, changed = normalize_setup(
            "%pip install torch \\\n+    --index-url https://download.pytorch.org/whl/cpu\nimport torch"
        )
        self.assertTrue(changed)
        self.assertIn("import torch", actual)
        self.assertNotIn("--index-url", actual)

    def test_training_cap_does_not_change_model_depth(self) -> None:
        actual = bound_training("model.fit(X, y, epochs=100)\ndepth = 12\nnum_epochs = 50")
        import ast

        tree = ast.parse(actual)
        self.assertEqual(tree.body[0].value.keywords[0].value.value, 2)
        self.assertEqual(tree.body[1].value.value, 12)
        self.assertEqual(tree.body[2].value.value, 2)

    def test_positional_training_budget_is_limited(self) -> None:
        namespace = {}
        exec(
            bound_training(
                "def train_logic_gate(target, lr, epochs):\n    return epochs\nobserved = train_logic_gate([], 0.01, 1000000)"
            ),
            namespace,
        )
        self.assertEqual(namespace["observed"], 2)

    def test_training_budget_text_is_not_rewritten(self) -> None:
        namespace = {}
        exec(bound_training('description = "Use epochs=100 for the full run"'), namespace)
        self.assertEqual(namespace["description"], "Use epochs=100 for the full run")

    def test_literal_epoch_loop_is_bounded_without_changing_feature_loop(self) -> None:
        namespace = {}
        exec(
            bound_training(
                "import itertools\nseen = []\nfor epoch in range(1, 10):\n    seen.append(epoch)\nfeatures = [i for i in range(12)]"
            ),
            namespace,
        )
        self.assertEqual(namespace["seen"], [1, 2])
        self.assertEqual(len(namespace["features"]), 12)

    def test_short_display_variable_does_not_trigger_training_cap(self) -> None:
        namespace = {}
        exec(
            bound_training(
                "seen = []\nfor e in ['one', 'two', 'three', 'four']:\n    seen.append(e)"
            ),
            namespace,
        )
        self.assertEqual(namespace["seen"], ["one", "two", "three", "four"])

    def test_cross_cell_training_default_and_progress_wrapper_are_bounded(self) -> None:
        definition = "def train(number_of_epochs=100):\n    return [epoch for epoch in range(number_of_epochs)]"
        call = "observed = train(1000)"
        signatures = training_signatures([definition, call])
        namespace = {}
        exec(bound_training(definition, signatures), namespace)
        exec(bound_training(call, signatures), namespace)
        self.assertEqual(namespace["observed"], [0, 1])
        exec(
            bound_training(
                "import itertools\ntqdm = lambda x: x\nseen = []\nfor epoch in tqdm(range(10)):\n    seen.append(epoch)"
            ),
            namespace,
        )
        self.assertEqual(namespace["seen"], [0, 1])

    def test_success_is_reused_only_with_matching_profile_and_artifact(self) -> None:
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(run_solutions, "ROOT", Path(folder)),
        ):
            artifact = Path(folder) / "executed.ipynb"
            artifact.write_bytes(b"actual evidence")
            record = {
                "status": "smoke-passed",
                "solution_sha256": "source",
                "execution_code_sha256": "profile",
                "executed_notebook": artifact.name,
                "executed_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }
            self.assertTrue(verified_previous(record, "source", "profile"))
            self.assertFalse(verified_previous(record, "source", "changed-profile"))
            artifact.write_bytes(b"changed evidence")
            self.assertFalse(verified_previous(record, "source", "profile"))
            artifact.unlink()
            self.assertFalse(verified_previous(record, "source", "profile"))


class ExecutionEvidenceTests(unittest.TestCase):
    """Historical outputs cannot certify a modified worked notebook."""

    def test_stale_source_is_counted_separately_and_corrupt_proof_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            solution = root / "solutions/course/notebooks/example.ipynb"
            artifact = root / "reports/executed-labs/course/notebooks/example.ipynb"
            for path in [solution, artifact]:
                path.parent.mkdir(parents=True, exist_ok=True)
            notebook = nbformat.v4.new_notebook(
                cells=[nbformat.v4.new_code_cell("answer = 1", execution_count=1)]
            )
            nbformat.write(notebook, artifact)
            solution.write_text("changed worked source")
            (root / "provenance").mkdir()
            for name in ["verification-plan.json", "smoke-overrides.json"]:
                (root / "provenance" / name).write_text("[]")
            record = {
                "solution": solution.relative_to(root).as_posix(),
                "solution_sha256": "old-source",
                "status": "smoke-passed",
                "executed_notebook": artifact.relative_to(root).as_posix(),
                "executed_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }
            ledger = root / "reports/solution-execution.json"
            ledger.write_text(json.dumps({"results": [record]}))
            progress = {
                "summary": {"syntax_or_stub_issues": 0},
                "notebooks": [{"solution": record["solution"]}],
            }
            with (
                patch.object(check_solutions, "ROOT", root),
                patch.object(check_solutions, "build", return_value=progress),
            ):
                result = check_solutions.check()
                self.assertEqual(result["saved_smoke_runs"], 1)
                self.assertEqual(result["current_smoke_runs"], 0)
                record["solution_sha256"] = hashlib.sha256(solution.read_bytes()).hexdigest()
                ledger.write_text(json.dumps({"results": [record]}))
                self.assertEqual(check_solutions.check()["current_smoke_runs"], 1)
                artifact.write_text("corrupt evidence")
                with self.assertRaisesRegex(ValueError, "Executed notebook changed"):
                    check_solutions.check()


class TensorFlowProfileTests(unittest.TestCase):
    """Tensor inputs must obey the same sample budget as NumPy inputs."""

    @unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow is not installed")
    def test_fit_caps_tensor_inputs_and_epoch_budget(self) -> None:
        import numpy as np
        import pandas as pd
        import tensorflow as tf

        node = next(
            n
            for n in ast.parse(run_solutions.TENSORFLOW_BOOTSTRAP).body
            if isinstance(n, ast.FunctionDef) and n.name == "_small_fit"
        )
        namespace = {
            "tf": tf,
            "np": np,
            "pd": pd,
            "_keras_fit": lambda model, *args, **kwargs: (args, kwargs),
        }
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), "<actual-fit-profile>", "exec"),
            namespace,
        )
        args, kwargs = namespace["_small_fit"](
            object(), tf.ones((2048, 2)), tf.ones((2048, 1)), epochs=100
        )
        self.assertEqual(len(args[0]), 512)
        self.assertEqual(len(args[1]), 512)
        self.assertEqual(kwargs["epochs"], 2)


class VisionSubsetTests(unittest.TestCase):
    """Only intended regular images can reach the local smoke dataset."""

    def test_subset_is_deterministic_and_rejects_archive_links_and_escape_paths(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / "images.tar"
            with tarfile.open(archive, "w") as target:
                for label in ["class_0_non_agri", "class_1_agri"]:
                    for filename in ["b.jpg", "a.jpg"]:
                        data = filename.encode()
                        member = tarfile.TarInfo(f"images_dataSAT/{label}/{filename}")
                        member.size = len(data)
                        target.addfile(member, io.BytesIO(data))
                for filename in ["../escape.jpg", "images_dataSAT/class_0_non_agri/..\\escape.jpg"]:
                    member = tarfile.TarInfo(filename)
                    member.size = 3
                    target.addfile(member, io.BytesIO(b"bad"))
                link = tarfile.TarInfo("images_dataSAT/class_0_non_agri/0-link.jpg")
                link.type = tarfile.SYMTYPE
                link.linkname = "../../escape.jpg"
                target.addfile(link)
            workspace = root / "cache"
            manifest = prepare(archive, workspace, 1)
            self.assertEqual(
                [row["file"] for row in manifest["selected_images"]],
                ["class_0_non_agri/a.jpg", "class_1_agri/a.jpg"],
            )
            self.assertFalse((root / "escape.jpg").exists())
            with self.assertRaises(FileExistsError):
                prepare(archive, workspace, 1)


if __name__ == "__main__":
    unittest.main()
