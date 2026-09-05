from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from compare_experiments import compact, parse_named_path  # noqa: E402


def test_parse_named_path_keeps_name_and_path() -> None:
    assert parse_named_path("v4.0-exp02=result/metrics.json") == (
        "v4.0-exp02",
        Path("result/metrics.json"),
    )


def test_compact_calculates_errors_for_any_square_matrix(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "competition_score": 0.9,
                "macro_f1": 0.8,
                "confusion_matrix": {"counts": [[8, 1], [2, 9]]},
                "per_class": {},
            }
        ),
        encoding="utf-8",
    )

    row = compact("test", metrics_path)

    assert row["errors"] == 3
    assert row["competition_score"] == 0.9
