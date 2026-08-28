from __future__ import annotations

import re
from typing import Any


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

PROMPT_FIELDS = {
    "事件时间": "timestamp",
    "日志管道": "pipeline",
    "来源 IP": "src_ip",
    "目标 IP": "dst_ip",
    "来源端口": "src_port",
    "来源主机": "src_host",
    "目标主机": "dst_host",
    "相关账号": "username",
    "安全产品": "product_name",
    "产品厂商": "vendor_name",
}

MISSING_TEXT = {"", "未知", "unknown", "null", "none", "nan", "<na>"}


def normalize_raw_value(value: str | None) -> str:
    if value is None:
        return ""
    stripped = value.strip()
    return "" if stripped.lower() in MISSING_TEXT else stripped


def parse_soc_prompt(prompt: str) -> dict[str, str]:
    """Parse the fixed Chinese prompt template back into raw SOC fields."""

    message_marker = "日志正文："
    marker_position = prompt.find(message_marker)
    if marker_position < 0:
        # Accept an ASCII colon in case a later converter changes punctuation.
        message_marker = "日志正文:"
        marker_position = prompt.find(message_marker)
    if marker_position < 0:
        raise ValueError("prompt does not contain 日志正文 marker")

    header_text = prompt[:marker_position]
    message = normalize_raw_value(prompt[marker_position + len(message_marker) :])
    parsed = {field: "" for field in PROMPT_FIELDS.values()}
    parsed["message_sanitized"] = message

    title_to_field = PROMPT_FIELDS
    for line in header_text.splitlines():
        stripped = line.strip()
        matched = False
        for title, field in title_to_field.items():
            for separator in ("：", ":"):
                prefix = title + separator
                if stripped.startswith(prefix):
                    parsed[field] = normalize_raw_value(stripped[len(prefix) :])
                    matched = True
                    break
            if matched:
                break
    return parsed


def product_group(product_name: str) -> str:
    if not product_name:
        return "missing"
    if product_name == "ASA Firewall":
        return "asa"
    if product_name == "AWS VPC Security":
        return "aws_vpc"
    if product_name in {"Precinct", "Falcon"}:
        return "other_suspicious_products"
    return "other"


def message_length_bucket(message: str) -> str:
    length = len(message)
    if length == 0:
        return "missing"
    if length <= 120:
        return "001-120"
    if length <= 180:
        return "121-180"
    if length <= 300:
        return "181-300"
    if length <= 1000:
        return "301-1000"
    return "1001+"


def source_ip_kind(src_ip: str) -> str:
    if not src_ip:
        return "missing"
    if re.fullmatch(r"[0-9]{1,3}(\.[0-9]{1,3}){3}", src_ip):
        return "ipv4_shape"
    if src_ip.lower().startswith("host-"):
        return "host_token"
    return "other"


def parse_port(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return -1
    return parsed if 0 <= parsed <= 65535 else -1


def port_bucket(port: int) -> str:
    if port < 0:
        return "missing"
    if port <= 1023:
        return "00000-01023"
    if port <= 49151:
        return "01024-49151"
    return "49152-65535"


def engineer_structured_features(raw: dict[str, str]) -> dict[str, Any]:
    pipeline = raw["pipeline"] or "__MISSING__"
    product_name = raw["product_name"] or "__MISSING__"
    src_ip = raw["src_ip"]
    dst_ip = raw["dst_ip"]
    src_host = raw["src_host"]
    dst_host = raw["dst_host"]
    username = raw["username"]
    message = raw["message_sanitized"]
    message_lower = message.lower()
    src_port = parse_port(raw["src_port"])

    product_group_value = product_group(raw["product_name"])
    message_bucket = message_length_bucket(message)
    deny = int("deny" in message_lower)
    src_ip_missing = int(not src_ip)
    dst_ip_missing = int(not dst_ip)
    src_port_missing = int(src_port < 0)
    return {
        "pipeline": pipeline,
        "product_name": product_name,
        "product_group": product_group_value,
        "src_ip_kind": source_ip_kind(src_ip),
        "port_bucket": port_bucket(src_port),
        "message_length_bucket": message_bucket,
        "structure_combo": (
            f"{pipeline}|{product_group_value}|{message_bucket}|"
            f"{'deny' if deny else 'not_deny'}"
        ),
        "network_missing_pattern": (
            f"{src_ip_missing}{dst_ip_missing}{src_port_missing}"
        ),
        "src_port_number": src_port,
        "src_ip_missing": src_ip_missing,
        "dst_ip_missing": dst_ip_missing,
        "src_port_missing": src_port_missing,
        "src_host_missing": int(not src_host),
        "dst_host_missing": int(not dst_host),
        "username_missing": int(not username),
        "product_missing": int(not raw["product_name"]),
        "message_missing": int(not message),
        "network_present_count": (
            int(bool(src_ip)) + int(bool(dst_ip)) + int(src_port >= 0)
        ),
        "message_length": len(message),
        "src_ip_length": len(src_ip),
        "dst_ip_length": len(dst_ip),
        "src_host_length": len(src_host),
        "dst_host_length": len(dst_host),
        "username_length": len(username),
        "message_has_deny": deny,
        "message_has_allow": int("allow" in message_lower),
        "message_has_accepted": int("accepted" in message_lower),
        "message_has_failed": int("failed" in message_lower),
        "message_has_blocked": int("blocked" in message_lower),
        "message_starts_angle": int(message.lstrip().startswith("<")),
        "message_contains_json": int("{" in message),
    }
