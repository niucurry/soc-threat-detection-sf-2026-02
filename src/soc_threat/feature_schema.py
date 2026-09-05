from __future__ import annotations


CATEGORICAL_FEATURES = [
    "pipeline",
    "product_name",
    "product_group",
    "src_ip_kind",
    "port_bucket",
    "message_length_bucket",
    "structure_combo",
    "network_missing_pattern",
]

NUMERIC_FEATURES = [
    "src_port_number",
    "src_ip_missing",
    "dst_ip_missing",
    "src_port_missing",
    "src_host_missing",
    "dst_host_missing",
    "username_missing",
    "product_missing",
    "message_missing",
    "network_present_count",
    "message_length",
    "src_ip_length",
    "dst_ip_length",
    "src_host_length",
    "dst_host_length",
    "username_length",
    "message_has_deny",
    "message_has_allow",
    "message_has_accepted",
    "message_has_failed",
    "message_has_blocked",
    "message_starts_angle",
    "message_contains_json",
]
