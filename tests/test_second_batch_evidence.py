"""Ensure actual bounded neural pilots cannot satisfy the larger publication scope."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_second_batch import validate_metrics  # noqa: E402


class PublicationScopeTests(unittest.TestCase):
    def test_empty_manifests_and_missing_models_cannot_pass_evidence_checks(self):
        for slug in ["translation", "fraud", "preference_alignment"]:
            folder = ROOT / "reports/second-batch-notebooks" / slug
            metrics = json.loads((folder / "metrics.json").read_text())
            manifest = json.loads((folder / "split-manifest.json").read_text())
            broken = copy.deepcopy(manifest)
            broken["partitions"] = {role: [] for role in ["train", "validation", "test"]}
            with self.subTest(project=slug), self.assertRaisesRegex(ValueError, "partition count"):
                validate_metrics(slug, metrics, broken)
            metrics["models"] = {}
            with self.subTest(project=slug), self.assertRaisesRegex(ValueError, "model results"):
                validate_metrics(slug, metrics, manifest)

    def test_saved_neural_pilots_are_rejected_as_full_public_experiments(self):
        for slug in ["translation", "cartpole", "preference_alignment"]:
            with self.subTest(project=slug):
                folder = ROOT / "reports/second-batch-notebooks" / slug
                metrics = json.loads((folder / "metrics.json").read_text())
                manifest = json.loads((folder / "split-manifest.json").read_text())
                validate_metrics(slug, metrics, manifest)
                with self.assertRaisesRegex(ValueError, "bounded pilot"):
                    validate_metrics(slug, metrics, manifest, full=True)

    def test_fraud_local_profile_uses_the_complete_experiment_scope(self):
        folder = ROOT / "reports/second-batch-notebooks/fraud"
        validate_metrics(
            "fraud",
            json.loads((folder / "metrics.json").read_text()),
            json.loads((folder / "split-manifest.json").read_text()),
            full=True,
        )


if __name__ == "__main__":
    unittest.main()
