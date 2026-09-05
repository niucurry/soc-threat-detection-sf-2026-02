from __future__ import annotations

from soc_threat.drain_feature_schema import (
    CATEGORICAL_FEATURES as DRAIN_CATEGORICAL_FEATURES,
)
from soc_threat.drain_feature_schema import NUMERIC_FEATURES as DRAIN_NUMERIC_FEATURES


# The ``*_v5`` storage columns are retained so historical Parquet files and
# checkpoints remain readable.  V5.1 is a legacy alias; the model version is v1.2.
CATEGORICAL_FEATURES = [
    *DRAIN_CATEGORICAL_FEATURES,
    "structured_parser",
    "payload_parse_status",
    "schema_id",
    "semantic_template_id",
    "event_category_v5",
    "event_type_v5",
    "event_action_v5",
    "event_outcome_v5",
    "event_reason_v5",
    "authentication_factor",
    "service_name_v5",
    "application_name_v5",
    "rule_name_v5",
    "threat_category_v5",
]

NUMERIC_FEATURES = [
    *DRAIN_NUMERIC_FEATURES,
    "structured_field_count",
    "security_field_count",
    "payload_parse_success",
    "payload_parse_error",
    "schema_seen_train",
    "schema_frequency_log1p",
    "semantic_template_seen_train",
    "semantic_template_frequency_log1p",
    "source_ip_in_message",
    "destination_ip_in_message",
    "event_severity_number",
    "malware_present",
    "detection_present",
    "authentication_present",
    "rule_name_present",
    "user_present_in_payload",
    "process_present_in_payload",
]


def validate_feature_lists() -> None:
    for name, values in (
        ("categorical", CATEGORICAL_FEATURES),
        ("numeric", NUMERIC_FEATURES),
    ):
        duplicates = sorted({value for value in values if values.count(value) > 1})
        if duplicates:
            raise ValueError(f"Duplicate {name} v1.2 structured features: {duplicates}")


validate_feature_lists()
