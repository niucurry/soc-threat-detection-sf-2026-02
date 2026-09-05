from __future__ import annotations

import sys
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
from soc_threat.structured_feature_schema import (  # noqa: E402
    NUMERIC_FEATURES as STRUCTURED_NUMERIC,
)


def test_standard_log_names_every_v1_family_feature() -> None:
    log = (PROJECT_ROOT / "docs" / "DEVELOPMENT_LOG_STANDARD.md").read_text("utf-8")
    for name in [
        *DRAIN_CATEGORICAL,
        *DRAIN_NUMERIC,
        *STRUCTURED_CATEGORICAL,
        *STRUCTURED_NUMERIC,
    ]:
        assert f"`{name}`" in log, name
