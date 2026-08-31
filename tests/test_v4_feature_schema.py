from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from soc_threat.feature_schema import (  # noqa: E402
    CATEGORICAL_FEATURES as V1_CATEGORICAL,
)
from soc_threat.feature_schema import NUMERIC_FEATURES as V1_NUMERIC  # noqa: E402
from soc_threat.v4_feature_schema import (  # noqa: E402
    CATEGORICAL_FEATURES as V4_CATEGORICAL,
)
from soc_threat.v4_feature_schema import NUMERIC_FEATURES as V4_NUMERIC  # noqa: E402


class V4FeatureSchemaTests(unittest.TestCase):
    def test_v4_extends_v1_without_duplicates(self) -> None:
        self.assertTrue(set(V1_CATEGORICAL).issubset(V4_CATEGORICAL))
        self.assertTrue(set(V1_NUMERIC).issubset(V4_NUMERIC))
        self.assertEqual(len(V4_CATEGORICAL), len(set(V4_CATEGORICAL)))
        self.assertEqual(len(V4_NUMERIC), len(set(V4_NUMERIC)))
        self.assertIn("template_id", V4_CATEGORICAL)
        self.assertIn("dst_port_number", V4_NUMERIC)


if __name__ == "__main__":
    unittest.main()
