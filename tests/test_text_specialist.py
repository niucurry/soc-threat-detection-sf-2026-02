from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from soc_threat.text_specialist import (  # noqa: E402
    best_binary_macro_f1_threshold,
    best_competition_threshold,
    specialist_route,
    strong_malicious_rules,
)


class TextSpecialistTests(unittest.TestCase):
    def test_specialist_route(self) -> None:
        frame = pd.DataFrame(
            {
                "pipeline": ["syslog", "syslog", "network_flows"],
                "product_name": [None, "Duo", None],
            }
        )
        np.testing.assert_array_equal(
            specialist_route(frame), np.array([True, False, False])
        )

    def test_strong_malicious_rules(self) -> None:
        messages = pd.Series(
            [
                "2 account eni src dst REJECT OK",
                'prefix {"code":"4625","outcome":"failure"}',
                "ORG-1780 ::: fqdn=HOST-1 ::: action=BLOCKED",
                "ordinary successful backup",
            ]
        )
        np.testing.assert_array_equal(
            strong_malicious_rules(messages),
            np.array([True, True, True, False]),
        )

    def test_best_threshold_separates_classes(self) -> None:
        probabilities = np.array([0.01, 0.2, 0.8, 0.99])
        actual = np.array(
            ["benign", "benign", "malicious", "malicious"], dtype=object
        )
        score, threshold = best_binary_macro_f1_threshold(
            probabilities, actual
        )
        self.assertEqual(score, 1.0)
        self.assertGreater(threshold, 0.2)
        self.assertLess(threshold, 0.8)

    def test_competition_threshold_separates_classes(self) -> None:
        probabilities = np.array([0.01, 0.2, 0.8, 0.99])
        actual = np.array(
            ["benign", "benign", "malicious", "malicious"], dtype=object
        )
        fixed = np.zeros((3, 3), dtype=np.int64)
        fixed[2, 2] = 10
        score, threshold = best_competition_threshold(
            probabilities, actual, fixed
        )
        self.assertEqual(score, 1.0)
        self.assertGreater(threshold, 0.2)
        self.assertLess(threshold, 0.8)


if __name__ == "__main__":
    unittest.main()
