from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_content_audit_supports_metadata_predictions_without_content_candidate(
    tmp_path: Path,
) -> None:
    predictions = tmp_path / "predictions.parquet"
    features = tmp_path / "features.parquet"
    output = tmp_path / "analysis"
    pd.DataFrame(
        {
            "event_id": ["threat", "benign"],
            "true_label": ["malicious", "benign"],
            "pred_label": ["suspicious", "benign"],
            "anchor_pred_label": ["benign", "benign"],
            "anchor_threat_probability": [0.4, 0.2],
            "metadata_threat_probability": [0.01, 0.02],
            "content_threat_probability": [0.95, 0.91],
            "threat_probability": [0.7, 0.2],
            "conflict_candidate": [1, 1],
            # Metadata-only predictions have no content candidate column.
            "metadata_reliability_candidate": [0, 0],
            "trust_score": [0.0, 0.0],
            "delta_margin": [0.0, 0.0],
        }
    ).to_parquet(predictions, index=False)
    pd.DataFrame(
        {
            "event_id": ["threat", "benign"],
            "pipeline": ["syslog", "syslog"],
            "vendor_name": ["__MISSING__", "__MISSING__"],
            "product_name": ["__MISSING__", "__MISSING__"],
            "content_family": ["json", "json"],
            "content_action": ["fail", "success"],
            "content_event_code": ["__MISSING__", "__MISSING__"],
            "message_length_bucket": ["301-1000", "301-1000"],
        }
    ).to_parquet(features, index=False)

    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "src" / "analyze_anchored_residual.py"),
            "--predictions",
            str(predictions),
            "--features",
            str(features),
            "--output-dir",
            str(output),
            "--evidence-source",
            "content",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(
        (output / "residual_summary.json").read_text(encoding="utf-8")
    )
    assert summary["evidence_source"] == "content"
    assert summary["content_reliability_candidates"] == 2
    assert summary["content_candidate_true_threat"] == 1
    assert summary["content_candidate_true_benign"] == 1
    assert summary["evidence_reliability_candidates"] == 2
    assert summary["conflict_true_threat"] == 1
    assert summary["conflict_true_benign"] == 1
    assert summary["anchor_threat_errors"] == 1
    assert summary["final_threat_errors"] == 0
    assert summary["fixed_anchor_errors"] == 0
    assert summary["fixed_anchor_threat_errors"] == 1
    assert summary["conflict_groups"][0]["avg_content_threat"] is not None
