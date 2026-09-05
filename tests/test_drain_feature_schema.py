from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from soc_threat.feature_schema import (  # noqa: E402
    CATEGORICAL_FEATURES as TABULAR_CATEGORICAL,
)
from soc_threat.feature_schema import NUMERIC_FEATURES as TABULAR_NUMERIC  # noqa: E402
from soc_threat.drain_feature_schema import (  # noqa: E402
    CATEGORICAL_FEATURES as DRAIN_CATEGORICAL,
)
from soc_threat.drain_feature_schema import NUMERIC_FEATURES as DRAIN_NUMERIC  # noqa: E402


class DrainFeatureSchemaTests(unittest.TestCase):
    def test_v1_1_extends_v1_0_without_duplicates(self) -> None:
        self.assertTrue(set(TABULAR_CATEGORICAL).issubset(DRAIN_CATEGORICAL))
        self.assertTrue(set(TABULAR_NUMERIC).issubset(DRAIN_NUMERIC))
        self.assertEqual(len(DRAIN_CATEGORICAL), len(set(DRAIN_CATEGORICAL)))
        self.assertEqual(len(DRAIN_NUMERIC), len(set(DRAIN_NUMERIC)))
        self.assertIn("template_id", DRAIN_CATEGORICAL)
        self.assertIn("dst_port_number", DRAIN_NUMERIC)


if __name__ == "__main__":
    unittest.main()
