from __future__ import annotations

from soc_threat.feature_schema import (
    CATEGORICAL_FEATURES as TABULAR_CATEGORICAL_FEATURES,
)
from soc_threat.feature_schema import NUMERIC_FEATURES as TABULAR_NUMERIC_FEATURES


CATEGORICAL_FEATURES = [
    *TABULAR_CATEGORICAL_FEATURES,
    "vendor_name",
    "parser_group",
    "message_format",
    "parser_type",
    "template_id",
    "semantic_action",
    "network_protocol",
    "event_code",
    "event_name",
    "dst_port_bucket",
    "http_method",
    "http_status_bucket",
    "source_zone",
    "destination_zone",
]

NUMERIC_FEATURES = [
    *TABULAR_NUMERIC_FEATURES,
    "utc_hour",
    "utc_weekday",
    "src_port_from_message",
    "dst_port_number",
    "dst_port_missing",
    "event_code_present",
    "semantic_action_present",
    "network_protocol_present",
    "semantic_field_count",
    "parse_success",
    "template_seen_train",
    "template_frequency_log1p",
    "template_wildcard_count",
    "message_token_count",
    "is_auth_failure",
    "is_network_denied",
    "is_process_creation",
    "is_privileged_logon",
]


def validate_feature_lists() -> None:
    for name, values in (
        ("categorical", CATEGORICAL_FEATURES),
        ("numeric", NUMERIC_FEATURES),
    ):
        duplicates = sorted({value for value in values if values.count(value) > 1})
        if duplicates:
            raise ValueError(f"Duplicate {name} v1.1 Drain features: {duplicates}")


validate_feature_lists()
