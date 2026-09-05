from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from soc_threat.drain_feature_schema import (  # noqa: E402
    CATEGORICAL_FEATURES as DRAIN_CATEGORICAL,
)
from soc_threat.drain_feature_schema import NUMERIC_FEATURES as DRAIN_NUMERIC  # noqa: E402
from soc_threat.structured_feature_schema import (  # noqa: E402
    CATEGORICAL_FEATURES as STRUCTURED_CATEGORICAL,
)
from soc_threat.structured_feature_schema import NUMERIC_FEATURES as STRUCTURED_NUMERIC  # noqa: E402


class StructuredFeatureSchemaTests(unittest.TestCase):
    def test_v1_2_extends_v1_1_without_duplicates(self) -> None:
        self.assertTrue(set(DRAIN_CATEGORICAL).issubset(STRUCTURED_CATEGORICAL))
        self.assertTrue(set(DRAIN_NUMERIC).issubset(STRUCTURED_NUMERIC))
        self.assertEqual(len(STRUCTURED_CATEGORICAL), len(set(STRUCTURED_CATEGORICAL)))
        self.assertEqual(len(STRUCTURED_NUMERIC), len(set(STRUCTURED_NUMERIC)))
        self.assertIn("schema_id", STRUCTURED_CATEGORICAL)
        self.assertIn("semantic_template_id", STRUCTURED_CATEGORICAL)
        self.assertIn("payload_parse_success", STRUCTURED_NUMERIC)
        self.assertIn("malware_present", STRUCTURED_NUMERIC)


if __name__ == "__main__":
    unittest.main()
