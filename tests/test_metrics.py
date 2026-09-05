from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from soc_threat import LABELS  # noqa: E402
from soc_threat.metrics import evaluate_predictions  # noqa: E402


class CompetitionMetricTests(unittest.TestCase):
    def test_perfect_predictions_score_one(self) -> None:
        actual = np.array(["benign", "malicious", "suspicious"], dtype=object)
        metrics = evaluate_predictions(actual, actual, labels=LABELS)
        self.assertEqual(metrics["competition_score"], 1.0)
        self.assertEqual(
            metrics["competition_metrics"]["soft_label_score"], 1.0
        )

    def test_weighted_components_match_known_confusion(self) -> None:
        actual = np.array(
            ["benign", "malicious", "suspicious", "suspicious"],
            dtype=object,
        )
        predicted = np.array(
            ["malicious", "suspicious", "malicious", "benign"],
            dtype=object,
        )
        metrics = evaluate_predictions(actual, predicted, labels=LABELS)
        competition = metrics["competition_metrics"]
        self.assertAlmostEqual(competition["threat_binary_f1"], 2.0 / 3.0)
        self.assertAlmostEqual(
            competition["threat_binary_recall"], 2.0 / 3.0
        )
        self.assertEqual(competition["threat_recall"], 0.0)
        self.assertEqual(competition["soft_label_score"], 0.25)
        self.assertAlmostEqual(
            metrics["competition_score"],
            0.40 * (2.0 / 3.0) + 0.25 * (2.0 / 3.0) + 0.05 * 0.25,
        )


if __name__ == "__main__":
    unittest.main()
