from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from soc_threat.text_specialist import (  # noqa: E402
    adaptive_probability_gap_threshold,
    best_binary_macro_f1_threshold,
    best_competition_threshold,
    specialist_route,
    strong_malicious_rules,
)


class TextSpecialistTests(unittest.TestCase):
    def test_adaptive_gap_prefers_lower_competitive_boundary(self) -> None:
        probabilities = np.array([0.01, 0.02, 0.03, 0.10, 0.30])
        threshold, details = adaptive_probability_gap_threshold(
            probabilities,
            min_positive_fraction=0.1,
            max_positive_fraction=0.8,
            relative_gap=0.3,
        )

        self.assertAlmostEqual(threshold, 0.065)
        self.assertEqual(details["predicted_positive_rows"], 2)

    def test_adaptive_gap_excludes_rule_rows(self) -> None:
        probabilities = np.array([0.01, 0.02, 0.03, 0.10, 0.99])
        excluded = np.array([False, False, False, False, True])
        threshold, details = adaptive_probability_gap_threshold(
            probabilities,
            excluded=excluded,
            min_positive_fraction=0.1,
            max_positive_fraction=0.8,
        )

        self.assertAlmostEqual(threshold, 0.065)
        self.assertEqual(details["rows"], 4)

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
                "1,TRAFFIC,drop,1,src,dst,tcp,deny,HOST-1-deny",
                "1,THREAT,url,1,src,dst,tcp,block-url,malware",
                "1,TRAFFIC,allow,1,src,dst,tcp,allow,drop_count=0",
                "1,THREAT,url,1,src,dst,tcp,allow,benign",
                "ordinary successful backup",
            ]
        )
        np.testing.assert_array_equal(
            strong_malicious_rules(messages, profile="expanded"),
            np.array([True, True, True, True, True, False, False, False]),
        )

    def test_rule_profiles_keep_version_specific_scope(self) -> None:
        messages = pd.Series(["1,TRAFFIC,drop,1", "1,THREAT,url,block-url", "REJECT OK"])
        np.testing.assert_array_equal(
            strong_malicious_rules(messages, profile="basic"), [False, False, True]
        )
        np.testing.assert_array_equal(
            strong_malicious_rules(messages, profile="expanded"), [True, True, True]
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
