"""Regression checks for source preservation and recoverable collection import."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nbformat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import import_labs  # noqa: E402


class ImportRegressionTests(unittest.TestCase):
    """Import real small source files, never the owner's original archive."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)
        self.source = self.base / "source"
        self.destination = self.base / "destination"
        self.source.mkdir()
        (self.destination / "provenance").mkdir(parents=True)
        (self.destination / "provenance/courses.json").write_text(
            json.dumps(
                [{"number": 1, "slug": "01-example", "title": "Gradients θ"}], ensure_ascii=False
            ),
            encoding="utf-8",
        )
        notebook = {
            "nbformat": 4,
            "nbformat_minor": 4,
            "metadata": {},
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["Source notes ©"],
                    "outputs": [],
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": ["value = 1"],
                    "execution_count": 1,
                    "outputs": [{"output_type": "stream", "name": "stdout", "text": "1"}],
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": ["Copyright © IBM Corporation. All rights reserved."],
                    "execution_count": None,
                    "outputs": [],
                },
            ],
        }
        rows = []
        for name in ["lab.ipynb", "lab-copy.ipynb", "guide.md"]:
            path = self.source / name
            content = (
                json.dumps(notebook, ensure_ascii=False)
                if path.suffix == ".ipynb"
                else "# Guide ©\n"
            )
            path.write_text(content, encoding="utf-8")
            rows.append(
                {
                    "course": 1,
                    "title": "Example",
                    "file": name,
                    "sha256": import_labs.sha256(path.read_bytes()),
                    "source_download": "private-location",
                }
            )
        (self.source / "download_manifest.json").write_text(json.dumps(rows), encoding="utf-8")
        self.original_bytes = {path.name: path.read_bytes() for path in self.source.iterdir()}

    def run_import(self) -> None:
        """Run the public command using fixture paths."""
        with (
            patch.object(import_labs, "ROOT", self.destination),
            patch.object(sys, "argv", ["import_labs.py", str(self.source)]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            import_labs.main()

    def test_import_preserves_sources_and_consolidates_identical_cells(self) -> None:
        self.run_import()
        manifest = json.loads(
            (self.destination / "provenance/source-manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(manifest), 3)
        self.assertEqual(len({row["target"] for row in manifest}), 2)
        self.assertEqual(sum(row["duplicate"] for row in manifest), 1)
        self.assertTrue(all("source_download" not in row for row in manifest))
        self.assertEqual(
            self.original_bytes, {path.name: path.read_bytes() for path in self.source.iterdir()}
        )
        target = next(row["target"] for row in manifest if row["target"].endswith(".ipynb"))
        notebook = nbformat.read(self.destination / target, as_version=4)
        nbformat.validate(notebook)
        self.assertEqual(notebook.cells[-1].cell_type, "markdown")
        self.assertEqual(notebook.cells[2].source, "value = 1")
        self.assertIsNone(notebook.cells[2].execution_count)
        self.assertEqual(notebook.cells[2].outputs, [])

    def test_failed_manifest_publication_rolls_back_and_can_retry(self) -> None:
        original_replace = Path.replace

        def fail_manifest(path: Path, target: Path) -> Path:
            if target == self.destination / "provenance/source-manifest.json":
                raise OSError("simulated publication failure")
            return original_replace(path, target)

        with (
            patch.object(Path, "replace", fail_manifest),
            self.assertRaisesRegex(OSError, "simulated publication failure"),
        ):
            self.run_import()
        self.assertFalse((self.destination / "courses").exists())
        self.assertFalse((self.destination / "provenance/source-manifest.json").exists())
        self.run_import()
        self.assertTrue((self.destination / "provenance/source-manifest.json").is_file())

    def test_changed_source_fails_before_publication(self) -> None:
        (self.source / "lab.ipynb").write_text("changed source", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Source hash mismatch"):
            self.run_import()
        self.assertFalse((self.destination / "courses").exists())

    def test_outside_source_path_is_rejected(self) -> None:
        outside = self.base / "outside.md"
        outside.write_text("private file", encoding="utf-8")
        (self.source / "download_manifest.json").write_text(
            json.dumps(
                [
                    {
                        "course": 1,
                        "title": "Outside",
                        "file": "../outside.md",
                        "sha256": import_labs.sha256(outside.read_bytes()),
                    }
                ]
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "Invalid source path"):
            self.run_import()
        self.assertFalse((self.destination / "courses").exists())

    def test_existing_collection_is_not_overwritten(self) -> None:
        self.run_import()
        before = (self.destination / "provenance/source-manifest.json").read_bytes()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.run_import()
        self.assertEqual(
            before, (self.destination / "provenance/source-manifest.json").read_bytes()
        )


if __name__ == "__main__":
    unittest.main()
