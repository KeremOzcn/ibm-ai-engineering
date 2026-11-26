"""Behavior regressions for the added portfolio experiments."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
TORCH = importlib.util.find_spec("torch") is not None
PANDAS = importlib.util.find_spec("pandas") is not None


@unittest.skipUnless(PANDAS, "pandas is not installed")
class FraudBoundaryTests(unittest.TestCase):
    def test_validation_score_ties_cannot_exceed_alert_budget(self):
        import numpy as np

        from projects.fraud.project import threshold_for_budget

        for scores in [[0.8] * 10, [0.1, 0.2, 0.9, 0.9], [0.0, 0.5, 0.6, 1.0]]:
            cutoff = threshold_for_budget(scores, 1)
            self.assertLessEqual(int((np.array(scores) >= cutoff).sum()), 1)
            self.assertTrue(np.isfinite(cutoff))

    def test_chronological_partitions_preserve_time_order_and_remove_repeated_events(self):
        import pandas as pd

        from projects.fraud.project import chronological_splits

        rows = []
        for i in range(40):
            rows.append(
                {
                    "Time": float(i),
                    "Amount": float(i + 1),
                    "Class": i % 2,
                    **{f"V{j}": float(i * j) for j in range(1, 29)},
                }
            )
        rows.append(rows[0].copy())
        parts, audit = chronological_splits(pd.DataFrame(rows))
        self.assertEqual(len(audit["repeated_event_ids"]), 1)
        self.assertLess(parts["train"].Time.max(), parts["validation"].Time.min())
        self.assertLess(parts["validation"].Time.max(), parts["test"].Time.min())
        self.assertEqual(sum(len(v) for v in parts.values()), 40)


@unittest.skipUnless(TORCH, "Torch is not installed")
class CartPoleBoundaryTests(unittest.TestCase):
    def test_time_limit_transition_bootstraps_but_terminal_transition_does_not(self):
        import torch

        from projects.cartpole.project import bellman_target

        result = bellman_target(
            torch.tensor([1.0, 1.0]), torch.tensor([True, False]), torch.tensor([10.0, 10.0])
        )
        torch.testing.assert_close(result, torch.tensor([1.0, 10.9]))


@unittest.skipUnless(TORCH, "Torch is not installed")
class TranslationBoundaryTests(unittest.TestCase):
    def test_future_target_tokens_do_not_change_earlier_decoder_logits(self):
        import torch

        from projects.translation.project import TransformerTranslator

        torch.manual_seed(42)
        model = TransformerTranslator(16, 16, width=32).eval()
        src = torch.tensor([[1, 4, 5, 2, 0]])
        with torch.no_grad():
            first = model(src, torch.tensor([[1, 5, 6, 2]]))
            changed = model(src, torch.tensor([[1, 5, 9, 8]]))
        torch.testing.assert_close(first[:, :2], changed[:, :2], rtol=1e-5, atol=1e-5)

    def test_repeated_source_sentence_stays_in_a_single_partition(self):
        from projects.translation.project import load_pairs

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for stem, rows in [
                ("train", ["ein hund", "ein zug"]),
                ("val", ["ein hund", "eine katze"]),
                ("test_2016_flickr", ["ein baum"]),
            ]:
                (root / f"{stem}.de").write_text("\n".join(rows))
                (root / f"{stem}.en").write_text("\n".join("a translation" for _ in rows))
            parts, audit = load_pairs(root)
        self.assertEqual(audit["excluded"]["repeated_source_sentence"], 1)
        self.assertEqual([r["source"] for r in parts["train"]], ["ein zug"])


@unittest.skipUnless(TORCH, "Torch is not installed")
class PreferenceBoundaryTests(unittest.TestCase):
    def test_reference_policy_has_log_two_loss_and_chosen_margin_gradient(self):
        import math

        import torch

        from projects.instruction_tuning.preference_alignment import dpo_objective

        chosen = torch.tensor([-2.0], requires_grad=True)
        rejected = torch.tensor([-3.0], requires_grad=True)
        loss = dpo_objective(chosen, rejected, chosen.detach(), rejected.detach())
        self.assertAlmostEqual(float(loss.detach()), math.log(2), places=6)
        loss.backward()
        self.assertLess(float(chosen.grad), 0)
        self.assertGreater(float(rejected.grad), 0)

    def test_response_logps_mask_prompt_padding_and_keep_real_eos(self):
        import math
        from types import SimpleNamespace

        import torch

        from projects.instruction_tuning.preference_alignment import response_logps

        class Model:
            def __init__(self, changed=False):
                self.changed = changed

            def __call__(self, input_ids, attention_mask):
                logits = torch.zeros((*input_ids.shape, 8))
                if self.changed:
                    logits[0, 0, 7] = 1000
                    logits[1, 2, 7] = 1000
                return SimpleNamespace(logits=logits)

        examples = [
            {"input_ids": [1, 3, 4, 2], "labels": [-100, -100, 4, 2]},
            {"input_ids": [1, 5, 2], "labels": [-100, 5, 2]},
        ]
        values, counts = response_logps(Model(), examples, 2, "cpu")
        altered, _ = response_logps(Model(True), examples, 2, "cpu")
        self.assertEqual(counts.tolist(), [2, 2])
        torch.testing.assert_close(values, torch.full((2,), -2 * math.log(8)))
        torch.testing.assert_close(values, altered)


if __name__ == "__main__":
    unittest.main()
