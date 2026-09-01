from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from soc_threat.v4_feature_schema import (  # noqa: E402
    CATEGORICAL_FEATURES as V4_CATEGORICAL,
)
from soc_threat.v4_feature_schema import NUMERIC_FEATURES as V4_NUMERIC  # noqa: E402
from soc_threat.v5_feature_schema import (  # noqa: E402
    CATEGORICAL_FEATURES as V5_CATEGORICAL,
)
from soc_threat.v5_feature_schema import NUMERIC_FEATURES as V5_NUMERIC  # noqa: E402


class V5FeatureSchemaTests(unittest.TestCase):
    def test_v5_extends_v4_without_duplicates(self) -> None:
        self.assertTrue(set(V4_CATEGORICAL).issubset(V5_CATEGORICAL))
        self.assertTrue(set(V4_NUMERIC).issubset(V5_NUMERIC))
        self.assertEqual(len(V5_CATEGORICAL), len(set(V5_CATEGORICAL)))
        self.assertEqual(len(V5_NUMERIC), len(set(V5_NUMERIC)))
        self.assertIn("schema_id", V5_CATEGORICAL)
        self.assertIn("semantic_template_id", V5_CATEGORICAL)
        self.assertIn("payload_parse_success", V5_NUMERIC)
        self.assertIn("malware_present", V5_NUMERIC)


if __name__ == "__main__":
    unittest.main()
