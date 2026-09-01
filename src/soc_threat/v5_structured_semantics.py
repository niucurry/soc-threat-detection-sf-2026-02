from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from soc_threat.log_semantics import (
    DrainSettings,
    MISSING_CATEGORY,
    ParsedLog,
    TemplateMatch,
    _text,
    _valid_port,
    http_status_bucket,
    normalize_for_template,
    parse_log,
    port_bucket,
    stable_template_key,
)


V5_TEMPLATE_MODEL_VERSION = 2
MAX_STRUCTURED_FIELDS = 256
MAX_STRUCTURED_CHARS = 32_000

SANITIZER_IN_KEY_PATTERN = re.compile(r"(?i)(?:USER|ORG|CRED|HOST)(?:-[A-Za-z0-9]+)+")
SANITIZER_VALUE_PATTERN = re.compile(
    r"(?i)\b(?:USER|ORG|CRED|HOST)(?:-[A-Za-z0-9]+)+\b"
)
IPV4_VALUE_PATTERN = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")

ACTION_ALIASES = {
    "allow": "allow",
    "allowed": "allow",
    "permit": "allow",
    "permitted": "allow",
    "accept": "accept",
    "accepted": "accept",
    "deny": "deny",
    "denied": "deny",
    "reject": "reject",
    "rejected": "reject",
    "drop": "drop",
    "dropped": "drop",
    "block": "block",
    "blocked": "block",
    "block-url": "block",
    "fail": "fail",
    "failed": "fail",
    "failure": "fail",
    "success": "success",
    "successful": "success",
}

OUTCOME_ALIASES = {
    "allow": "success",
    "allowed": "success",
    "accept": "success",
    "accepted": "success",
    "ok": "success",
    "success": "success",
    "successful": "success",
    "succeeded": "success",
    "deny": "failure",
    "denied": "failure",
    "reject": "failure",
    "rejected": "failure",
    "fail": "failure",
    "failed": "failure",
    "failure": "failure",
    "error": "failure",
}

AUTH_FACTOR_VALUES = {
    "duo_push": "duo_push",
    "push": "push",
    "phone": "phone",
    "phone_call": "phone_call",
    "sms": "sms",
    "passcode": "passcode",
    "password": "password",
    "webauthn": "webauthn",
    "u2f": "u2f",
    "totp": "totp",
}

REASON_PATTERNS = (
    (re.compile(r"(?i)invalid[_ -]?passcode"), "invalid_passcode"),
    (re.compile(r"(?i)auth(?:entication)?[_ -]?failure"), "authentication_failure"),
    (re.compile(r"(?i)bad[_ -]?password|invalid[_ -]?password"), "invalid_password"),
    (re.compile(r"(?i)account[_ -]?locked|locked[_ -]?account"), "account_locked"),
    (re.compile(r"(?i)user[_ -]?denied|denied[_ -]?by[_ -]?user"), "user_denied"),
    (re.compile(r"(?i)malware"), "malware"),
    (re.compile(r"(?i)geo[_ -]?ip[_ -]?block"), "geo_ip_block"),
)


@dataclass(frozen=True)
class V5ParsedLog:
    base: ParsedLog
    structured_parser: str
    payload_parse_status: str
    schema_id: str
    schema_signature: str
    semantic_template_id: str
    semantic_template: str
    event_category: str
    event_type: str
    event_action: str
    event_outcome: str
    event_reason: str
    authentication_factor: str
    service_name: str
    application_name: str
    rule_name: str
    threat_category: str
    event_severity: int
    event_code: str
    network_protocol: str
    http_method: str
    http_status: int
    source_ip_present: int
    destination_ip_present: int
    source_port: int
    destination_port: int
    structured_field_count: int
    security_field_count: int
    payload_parse_success: int
    payload_parse_error: int
    malware_present: int
    detection_present: int
    authentication_present: int
    rule_name_present: int
    user_present_in_payload: int
    process_present_in_payload: int
    drain_message: str


def _stable_key(prefix: str, parser_group: str, value: str) -> str:
    digest = hashlib.sha1(
        (parser_group + "\x00" + value).encode("utf-8", errors="replace")
    ).hexdigest()[:20]
    return f"{prefix}_{digest}"


def _normalize_key_part(value: str) -> str:
    replaced = SANITIZER_IN_KEY_PATTERN.sub("<TOKEN>", value.strip())
    replaced = re.sub(r"[^A-Za-z0-9_.<>-]+", "_", replaced).strip("_")
    return replaced.lower() or "unknown"


def _normalized_leaf(path: str) -> str:
    return _normalize_key_part(path.rsplit(".", 1)[-1])


def _bounded_category(value: Any, *, max_chars: int = 96) -> str:
    if value is None:
        return MISSING_CATEGORY
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).strip()
    if not text:
        return MISSING_CATEGORY
    text = SANITIZER_VALUE_PATTERN.sub("<TOKEN>", text)
    text = IPV4_VALUE_PATTERN.sub("<IP>", text)
    text = re.sub(
        r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        "<UUID>",
        text,
    )
    text = re.sub(r"\s+", " ", text).strip().lower()
    if len(text) > max_chars:
        text = normalize_for_template(text, max_chars=max_chars).lower()
    text = text[:max_chars]
    if text in {"<token>", "<ip>", "<uuid>"}:
        return MISSING_CATEGORY
    return text or MISSING_CATEGORY


def _flatten_value(
    value: Any,
    *,
    prefix: str,
    output: dict[str, list[Any]],
    depth: int = 0,
) -> None:
    if depth > 8 or len(output) >= MAX_STRUCTURED_FIELDS:
        return
    if isinstance(value, dict):
        for raw_key, child in value.items():
            key = _normalize_key_part(str(raw_key))
            path = f"{prefix}.{key}" if prefix else key
            _flatten_value(child, prefix=path, output=output, depth=depth + 1)
            if len(output) >= MAX_STRUCTURED_FIELDS:
                break
        return
    if isinstance(value, list):
        output.setdefault(prefix, []).append(f"__LIST_LEN__:{len(value)}")
        for child in value[:32]:
            if isinstance(child, (dict, list)):
                _flatten_value(child, prefix=prefix, output=output, depth=depth + 1)
            else:
                output.setdefault(prefix, []).append(child)
        return
    output.setdefault(prefix, []).append(value)


def _extract_json_fields(message: str) -> tuple[dict[str, list[Any]], bool]:
    fields: dict[str, list[Any]] = {}
    decoder = json.JSONDecoder()
    text = message[:MAX_STRUCTURED_CHARS]
    cursor = 0
    payload_index = 0
    attempts = 0
    while cursor < len(text) and attempts < 16 and len(fields) < MAX_STRUCTURED_FIELDS:
        brace = text.find("{", cursor)
        bracket = text.find("[", cursor)
        positions = [position for position in (brace, bracket) if position >= 0]
        if not positions:
            break
        start = min(positions)
        attempts += 1
        try:
            value, consumed = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            cursor = start + 1
            continue
        if isinstance(value, (dict, list)):
            _flatten_value(
                value,
                prefix=f"payload{payload_index}",
                output=fields,
            )
            payload_index += 1
        cursor = start + max(consumed, 1)
    return fields, payload_index > 0


def _extract_envelope_fields(message: str) -> dict[str, list[Any]]:
    fields: dict[str, list[Any]] = {}
    for segment in message[:MAX_STRUCTURED_CHARS].split(":::"):
        if "=" not in segment:
            continue
        raw_key, raw_value = segment.split("=", 1)
        key = _normalize_key_part(raw_key)
        value = raw_value.strip()
        if not key or not value or value.startswith(("{", "[")):
            continue
        fields.setdefault(f"envelope.{key}", []).append(value)
    return fields


def _strip_xml_namespace(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].split(":")[-1]


def _extract_xml_fields(message: str) -> tuple[dict[str, list[Any]], bool]:
    fields: dict[str, list[Any]] = {}
    start = message.find("<")
    if start < 0:
        return fields, False
    xml_text = message[start:MAX_STRUCTURED_CHARS]
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        # Sanitization can replace tag/attribute names and occasionally leave a
        # malformed opening tag.  Leaf values are still recoverable without
        # pretending that the whole document was valid XML.
        for match in re.finditer(
            r"<([A-Za-z_][\w:.-]*)(?:\s[^<>]*)?>([^<>]{1,4096})</",
            xml_text,
        ):
            tag = _normalize_key_part(_strip_xml_namespace(match.group(1)))
            value = match.group(2).strip()
            if value:
                fields.setdefault(f"xml_fallback.{tag}", []).append(value)
            if len(fields) >= MAX_STRUCTURED_FIELDS:
                break
        for tag_match in re.finditer(r"<([A-Za-z_][\w:.-]*)([^<>]{0,4096})>", xml_text):
            tag = _normalize_key_part(_strip_xml_namespace(tag_match.group(1)))
            attributes = tag_match.group(2)
            for attribute in re.finditer(
                r"([A-Za-z_][\w:.-]*)\s*=\s*(['\"])(.*?)\2", attributes
            ):
                key = _normalize_key_part(attribute.group(1))
                fields.setdefault(f"xml_fallback.{tag}.@{key}", []).append(
                    attribute.group(3)
                )
                if len(fields) >= MAX_STRUCTURED_FIELDS:
                    break
            if len(fields) >= MAX_STRUCTURED_FIELDS:
                break
        return fields, bool(fields)

    def visit(node: ET.Element, prefix: str, depth: int) -> None:
        if depth > 10 or len(fields) >= MAX_STRUCTURED_FIELDS:
            return
        tag = _normalize_key_part(_strip_xml_namespace(node.tag))
        path = f"{prefix}.{tag}" if prefix else tag
        text = (node.text or "").strip()
        if text:
            fields.setdefault(path, []).append(text)
        for key, value in node.attrib.items():
            attribute_path = f"{path}.@{_normalize_key_part(key)}"
            fields.setdefault(attribute_path, []).append(value)
        for child in list(node):
            visit(child, path, depth + 1)

    visit(root, "", 0)
    return fields, True


def _split_unescaped(value: str, separator: str, maxsplit: int) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    escaped = False
    splits = 0
    for character in value:
        if escaped:
            current.append(character)
            escaped = False
            continue
        if character == "\\":
            escaped = True
            current.append(character)
            continue
        if character == separator and splits < maxsplit:
            parts.append("".join(current))
            current = []
            splits += 1
            continue
        current.append(character)
    parts.append("".join(current))
    return parts


def _parse_cef_extension(extension: str) -> dict[str, list[Any]]:
    fields: dict[str, list[Any]] = {}
    matches = list(
        re.finditer(r"(?:(?<=^)|(?<=\s))([A-Za-z_][A-Za-z0-9_.-]*)=", extension)
    )
    for index, match in enumerate(matches):
        value_start = match.end()
        value_end = (
            matches[index + 1].start() if index + 1 < len(matches) else len(extension)
        )
        key = _normalize_key_part(match.group(1))
        value = extension[value_start:value_end].strip()
        fields.setdefault(f"cef.extension.{key}", []).append(value)
    return fields


def _extract_cef_fields(message: str) -> tuple[dict[str, list[Any]], bool, str]:
    fields: dict[str, list[Any]] = {}
    start = message.lower().find("cef:")
    if start < 0:
        return fields, False, ""
    cef = message[start:].strip()
    parts = _split_unescaped(cef, "|", maxsplit=7)
    if len(parts) < 8:
        return fields, False, ""
    header_names = (
        "version",
        "device_vendor",
        "device_product",
        "device_version",
        "signature_id",
        "name",
        "severity",
    )
    first = parts[0]
    header_values = [first.split(":", 1)[1], *parts[1:7]]
    for key, value in zip(header_names, header_values):
        fields[f"cef.header.{key}"] = [value.strip()]
    extension_fields = _parse_cef_extension(parts[7])
    fields.update(extension_fields)
    message_values = _values_for_aliases(extension_fields, {"msg", "message"})
    event_name = header_values[5].strip()
    residual = " ".join(
        value
        for value in [event_name, *[str(item) for item in message_values]]
        if value
    )
    return fields, True, residual


def _extract_vpc_fields(message: str) -> tuple[dict[str, list[Any]], bool]:
    fields: dict[str, list[Any]] = {}
    tokens = message.strip().split()
    start = -1
    for index, token in enumerate(tokens):
        if token.isdigit() and 2 <= int(token) <= 8 and len(tokens) - index >= 14:
            if tokens[index + 3].count(".") == 3 or tokens[index + 3] == "-":
                start = index
                break
    if start < 0:
        return fields, False
    values = tokens[start : start + 14]
    names = (
        "version",
        "account_id",
        "interface_id",
        "srcaddr",
        "dstaddr",
        "srcport",
        "dstport",
        "protocol",
        "packets",
        "bytes",
        "start",
        "end",
        "action",
        "log_status",
    )
    for name, value in zip(names, values):
        fields[f"vpc.{name}"] = [value]
    return fields, True


def _values_for_aliases(fields: dict[str, list[Any]], aliases: set[str]) -> list[Any]:
    values: list[Any] = []
    for path, path_values in fields.items():
        leaf = _normalized_leaf(path)
        if leaf in aliases or any(leaf.endswith(f"_{alias}") for alias in aliases):
            values.extend(path_values)
    return values


def _all_scalar_text(fields: dict[str, list[Any]], *, max_values: int = 512) -> str:
    values: list[str] = []
    for path_values in fields.values():
        for value in path_values:
            if isinstance(value, (str, int, float, bool)):
                values.append(str(value))
            if len(values) >= max_values:
                return " ".join(values)
    return " ".join(values)


def _first_category(fields: dict[str, list[Any]], aliases: set[str]) -> str:
    for value in _values_for_aliases(fields, aliases):
        if isinstance(value, str) and value.startswith("__LIST_LEN__:"):
            continue
        category = _bounded_category(value)
        if category != MISSING_CATEGORY:
            return category
    return MISSING_CATEGORY


def _first_category_at_paths(
    fields: dict[str, list[Any]], suffixes: tuple[str, ...]
) -> str:
    """Return a value only when its complete structured path has the right context."""

    normalized_suffixes = tuple(_normalize_key_part(value) for value in suffixes)
    for path, values in fields.items():
        normalized_path = _normalize_key_part(path)
        if not any(
            normalized_path == suffix or normalized_path.endswith(f".{suffix}")
            for suffix in normalized_suffixes
        ):
            continue
        for value in values:
            if isinstance(value, str) and value.startswith("__LIST_LEN__:"):
                continue
            category = _bounded_category(value)
            if category != MISSING_CATEGORY:
                return category
    return MISSING_CATEGORY


def _canonical_action(base_action: str, fields: dict[str, list[Any]]) -> str:
    if base_action != MISSING_CATEGORY:
        return base_action
    for value in _values_for_aliases(
        fields, {"action", "act", "decision", "disposition", "actiontaken"}
    ):
        lowered = str(value).strip().lower()
        comparable = re.sub(r"[_-]+", " ", lowered)
        for token, canonical in ACTION_ALIASES.items():
            if re.search(rf"\b{re.escape(token)}\b", comparable):
                return canonical
    return MISSING_CATEGORY


def _canonical_outcome(fields: dict[str, list[Any]], action: str) -> str:
    values = _values_for_aliases(fields, {"outcome", "result", "status"})
    for value in values:
        category = _bounded_category(value)
        if category == MISSING_CATEGORY:
            continue
        lowered = category
        comparable = re.sub(r"[_-]+", " ", lowered)
        for token, canonical in OUTCOME_ALIASES.items():
            if re.search(rf"\b{re.escape(token)}\b", comparable):
                return canonical
    for value in values:
        category = _bounded_category(value)
        if category != MISSING_CATEGORY:
            return category
    if action in {"allow", "accept", "success"}:
        return "success"
    return MISSING_CATEGORY


def _canonical_reason(fields: dict[str, list[Any]], message: str) -> str:
    reason_values = _values_for_aliases(fields, {"reason", "failure_reason", "message"})
    combined = " ".join(str(value) for value in reason_values[:16])
    for pattern, reason in REASON_PATTERNS:
        if pattern.search(combined) or pattern.search(message):
            return reason
    return MISSING_CATEGORY


def _authentication_factor(fields: dict[str, list[Any]]) -> str:
    candidates = _values_for_aliases(
        fields, {"factor", "auth_factor", "authentication_factor", "method"}
    )
    candidates.extend(
        value
        for path_values in fields.values()
        for value in path_values
        if isinstance(value, str)
    )
    for value in candidates:
        lowered = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        for token, canonical in AUTH_FACTOR_VALUES.items():
            if token in lowered:
                return canonical
    return MISSING_CATEGORY


def _event_type(fields: dict[str, list[Any]]) -> str:
    known_values = {
        "authentication",
        "process",
        "network",
        "connection",
        "malware",
        "alert",
        "detection",
        "dns",
        "file",
        "registry",
        "web",
        "email",
        "cloud",
    }
    for value in _values_for_aliases(fields, {"event_type", "type", "kind"}):
        category = _bounded_category(value)
        if category in known_values:
            return category
    return MISSING_CATEGORY


def _event_category(
    fields: dict[str, list[Any]],
    *,
    event_type: str,
    event_name: str,
    protocol: str,
    http_method: str,
    combined_text: str,
) -> str:
    lowered = combined_text.lower()
    if "malware" in lowered or "threat" in lowered:
        return "malware"
    if (
        "authentication" in lowered
        or "duo_push" in lowered
        or "invalid_passcode" in lowered
        or event_name in {"logon_failure", "logon_success", "credential_validation"}
    ):
        return "authentication"
    if event_name in {"process_creation", "process_exit"} or "process" in event_type:
        return "process"
    if http_method != MISSING_CATEGORY:
        return "web"
    if protocol != MISSING_CATEGORY or any(
        token in lowered
        for token in ("srcaddr", "dstaddr", "source.ip", "destination.ip")
    ):
        return "network"
    if event_type != MISSING_CATEGORY:
        return event_type
    return MISSING_CATEGORY


def _numeric_value(fields: dict[str, list[Any]], aliases: set[str]) -> int:
    for value in _values_for_aliases(fields, aliases):
        parsed = _valid_port(value)
        if parsed >= 0:
            return parsed
    return -1


def _event_severity(fields: dict[str, list[Any]]) -> int:
    for value in _values_for_aliases(fields, {"severity", "severitycode", "level"}):
        match = re.fullmatch(r"\s*(\d{1,3})(?:\.0+)?\s*", str(value))
        if match and 0 <= int(match.group(1)) <= 100:
            return int(match.group(1))
    return -1


def _structured_event_code(base: ParsedLog, fields: dict[str, list[Any]]) -> str:
    if base.event_code != MISSING_CATEGORY:
        return base.event_code
    for value in _values_for_aliases(
        fields, {"event_code", "eventcode", "eventid", "signature_id", "code"}
    ):
        category = _bounded_category(value, max_chars=48)
        if category != MISSING_CATEGORY:
            return category
    return MISSING_CATEGORY


def _structured_protocol(base: ParsedLog, fields: dict[str, list[Any]]) -> str:
    if base.network_protocol != MISSING_CATEGORY:
        return base.network_protocol
    protocol_numbers = {"1": "icmp", "6": "tcp", "17": "udp", "58": "icmpv6"}
    known = {"tcp", "udp", "icmp", "icmpv6", "gre", "esp", "sctp"}
    for value in _values_for_aliases(fields, {"protocol", "proto"}):
        category = _bounded_category(value, max_chars=24)
        if category in protocol_numbers:
            return protocol_numbers[category]
        if category in known:
            return category
    return MISSING_CATEGORY


def _structured_http_method(base: ParsedLog, fields: dict[str, list[Any]]) -> str:
    if base.http_method != MISSING_CATEGORY:
        return base.http_method
    known = {
        "GET",
        "POST",
        "PUT",
        "DELETE",
        "PATCH",
        "HEAD",
        "OPTIONS",
        "CONNECT",
        "TRACE",
    }
    for value in _values_for_aliases(
        fields, {"http_method", "request_method", "requestmethod", "method"}
    ):
        method = str(value).strip().upper()
        if method in known:
            return method
    return MISSING_CATEGORY


def _structured_http_status(base: ParsedLog, fields: dict[str, list[Any]]) -> int:
    if base.http_status >= 0:
        return base.http_status
    for value in _values_for_aliases(
        fields, {"http_status", "status_code", "statuscode", "response_code"}
    ):
        match = re.fullmatch(r"\s*(\d{3})\s*", str(value))
        if match and 100 <= int(match.group(1)) <= 599:
            return int(match.group(1))
    return -1


def _ip_present(fields: dict[str, list[Any]], aliases: set[str]) -> int:
    return int(
        any(
            IPV4_VALUE_PATTERN.search(str(value)) is not None
            for value in _values_for_aliases(fields, aliases)
        )
    )


def _semantic_template(
    base: ParsedLog,
    *,
    schema_id: str,
    structured_parser: str,
    event_category: str,
    event_type: str,
    event_action: str,
    event_outcome: str,
    event_reason: str,
    authentication_factor: str,
    threat_category: str,
    event_code: str,
    network_protocol: str,
    http_method: str,
) -> str:
    parts = [f"format={base.message_format}", f"parser={structured_parser}"]
    for name, value in (
        ("schema", schema_id),
        ("event_code", event_code),
        ("event_name", base.event_name),
        ("category", event_category),
        ("event_type", event_type),
        ("action", event_action),
        ("outcome", event_outcome),
        ("reason", event_reason),
        ("protocol", network_protocol),
        ("http_method", http_method),
        ("auth_factor", authentication_factor),
        ("threat", threat_category),
    ):
        if value != MISSING_CATEGORY:
            parts.append(f"{name}={value}")
    return " ".join(parts)


def parse_v5_log(raw: dict[str, Any], *, max_message_chars: int = 2048) -> V5ParsedLog:
    base = parse_log(raw, max_message_chars=max_message_chars)
    message = _text(raw.get("message_sanitized"))
    fields: dict[str, list[Any]] = {}
    structured_parser = "text_semantics"
    payload_status = "not_applicable"
    drain_message = base.normalized_message

    if base.message_format in {"json", "windows_json"}:
        structured_parser = "json_recursive"
        json_fields, success = _extract_json_fields(message)
        fields.update(json_fields)
        fields.update(_extract_envelope_fields(message))
        payload_status = "success" if success else ("partial" if fields else "failed")
    elif base.message_format == "windows_xml":
        structured_parser = "xml_tree"
        fields, success = _extract_xml_fields(message)
        payload_status = "success" if success else "failed"
    elif base.message_format == "cef":
        structured_parser = "cef_fields"
        fields, success, residual = _extract_cef_fields(message)
        payload_status = "success" if success else "failed"
        drain_message = normalize_for_template(
            residual or message, max_chars=max_message_chars
        )
    elif base.message_format == "vpc_flow":
        structured_parser = "vpc_flow_fields"
        fields, success = _extract_vpc_fields(message)
        payload_status = "success" if success else "failed"
    elif base.message_format == "blank":
        structured_parser = "blank"
        payload_status = "blank"
        drain_message = ""

    schema_paths = sorted(fields)
    schema_signature = ",".join(schema_paths)
    schema_id = (
        _stable_key("sch", base.parser_group, schema_signature)
        if schema_paths
        else MISSING_CATEGORY
    )

    action = _canonical_action(base.semantic_action, fields)
    outcome = _canonical_outcome(fields, action)
    reason = _canonical_reason(fields, message)
    factor = _authentication_factor(fields)
    event_type = _event_type(fields)
    event_code = _structured_event_code(base, fields)
    network_protocol = _structured_protocol(base, fields)
    http_method = _structured_http_method(base, fields)
    http_status = _structured_http_status(base, fields)
    combined_text = f"{message[:16000]} {_all_scalar_text(fields)}"
    event_category = _event_category(
        fields,
        event_type=event_type,
        event_name=base.event_name,
        protocol=network_protocol,
        http_method=http_method,
        combined_text=combined_text,
    )

    service_name = _first_category(
        fields, {"service", "service_name", "streamname", "provider_name"}
    )
    application_name = _first_category_at_paths(
        fields,
        (
            "application.name",
            "application_name",
            "app.name",
            "app_name",
            "envelope.application",
        ),
    )
    rule_name = _first_category_at_paths(
        fields,
        (
            "rule.name",
            "rule_name",
            "rulename",
            "signature.name",
            "cef.header.name",
        ),
    )
    lowered = combined_text.lower()
    threat_category = "malware" if "malware" in lowered else MISSING_CATEGORY
    severity = _event_severity(fields)

    structured_src_port = _numeric_value(
        fields, {"srcport", "src_port", "source_port", "spt"}
    )
    structured_dst_port = _numeric_value(
        fields, {"dstport", "dst_port", "destination_port", "dpt", "dport"}
    )
    source_port = (
        structured_src_port if structured_src_port >= 0 else base.src_port_from_message
    )
    destination_port = (
        structured_dst_port if structured_dst_port >= 0 else base.dst_port
    )

    source_ip_present = _ip_present(
        fields, {"src", "srcaddr", "src_ip", "source_ip", "ip"}
    )
    destination_ip_present = _ip_present(
        fields, {"dst", "dstaddr", "dst_ip", "destination_ip"}
    )
    malware_present = int("malware" in lowered)
    detection_present = int(
        any(token in lowered for token in ("detection", "detected", "alert"))
    )
    authentication_present = int(
        event_category == "authentication"
        or factor != MISSING_CATEGORY
        or base.is_auth_failure == 1
    )
    user_present = int(
        bool(
            _values_for_aliases(
                fields, {"user", "username", "user_name", "duser", "suser"}
            )
        )
    )
    process_present = int(
        bool(
            _values_for_aliases(fields, {"process", "process_name", "pid", "processid"})
        )
        or base.is_process_creation == 1
    )

    security_values: Iterable[Any] = (
        event_code,
        event_category,
        event_type,
        action,
        outcome,
        reason,
        factor,
        network_protocol,
        source_port,
        destination_port,
        severity,
        threat_category,
    )
    security_count = sum(
        value not in {MISSING_CATEGORY, -1, ""} for value in security_values
    )
    semantic_template = _semantic_template(
        base,
        schema_id=schema_id,
        structured_parser=structured_parser,
        event_category=event_category,
        event_type=event_type,
        event_action=action,
        event_outcome=outcome,
        event_reason=reason,
        authentication_factor=factor,
        threat_category=threat_category,
        event_code=event_code,
        network_protocol=network_protocol,
        http_method=http_method,
    )
    semantic_template_id = _stable_key("sem", base.parser_group, semantic_template)
    return V5ParsedLog(
        base=base,
        structured_parser=structured_parser,
        payload_parse_status=payload_status,
        schema_id=schema_id,
        schema_signature=schema_signature,
        semantic_template_id=semantic_template_id,
        semantic_template=semantic_template,
        event_category=event_category,
        event_type=event_type,
        event_action=action,
        event_outcome=outcome,
        event_reason=reason,
        authentication_factor=factor,
        service_name=service_name,
        application_name=application_name,
        rule_name=rule_name,
        threat_category=threat_category,
        event_severity=severity,
        event_code=event_code,
        network_protocol=network_protocol,
        http_method=http_method,
        http_status=http_status,
        source_ip_present=source_ip_present,
        destination_ip_present=destination_ip_present,
        source_port=source_port,
        destination_port=destination_port,
        structured_field_count=len(fields),
        security_field_count=security_count,
        payload_parse_success=int(payload_status == "success"),
        payload_parse_error=int(payload_status == "failed"),
        malware_present=malware_present,
        detection_present=detection_present,
        authentication_present=authentication_present,
        rule_name_present=int(rule_name != MISSING_CATEGORY),
        user_present_in_payload=user_present,
        process_present_in_payload=process_present,
        drain_message=drain_message,
    )


def should_use_v5_drain(parsed: V5ParsedLog) -> bool:
    return parsed.base.message_format in {
        "asa",
        "cef",
        "key_value",
        "linux_syslog",
        "syslog_text",
        "free_text",
    } and bool(parsed.drain_message)


class V5GroupedDrainModel:
    """Train-only V5 structured schema counts plus grouped Drain templates."""

    def __init__(self, settings: DrainSettings | None = None) -> None:
        self.settings = settings or DrainSettings()
        self.miners: dict[str, Any] = {}
        self.direct_template_counts: Counter[str] = Counter()
        self.schema_counts: Counter[str] = Counter()
        self.semantic_template_counts: Counter[str] = Counter()
        self.fitted_rows = 0
        self.drain_rows = 0
        self.direct_rows = 0

    def _config(self) -> Any:
        from soc_threat.log_semantics import GroupedDrainModel

        return GroupedDrainModel(self.settings)._config()

    def _new_miner(self) -> Any:
        from drain3.template_miner import TemplateMiner

        return TemplateMiner(None, self._config())

    def _miner_for_fit(self, parser_group: str) -> Any | None:
        miner = self.miners.get(parser_group)
        if miner is not None:
            return miner
        if len(self.miners) >= self.settings.max_groups:
            return None
        miner = self._new_miner()
        self.miners[parser_group] = miner
        return miner

    def fit_raw(self, raw: dict[str, Any]) -> None:
        parsed = parse_v5_log(raw, max_message_chars=self.settings.max_message_chars)
        self.fitted_rows += 1
        if parsed.schema_id != MISSING_CATEGORY:
            self.schema_counts[parsed.schema_id] += 1
        self.semantic_template_counts[parsed.semantic_template_id] += 1
        if should_use_v5_drain(parsed):
            miner = self._miner_for_fit(parsed.base.parser_group)
            if miner is not None:
                miner.add_log_message(parsed.drain_message)
                self.drain_rows += 1
                return
        key = stable_template_key(parsed.base.parser_group, parsed.semantic_template)
        self.direct_template_counts[key] += 1
        self.direct_rows += 1

    def match_parsed(self, parsed: V5ParsedLog) -> TemplateMatch:
        if should_use_v5_drain(parsed):
            miner = self.miners.get(parsed.base.parser_group)
            if miner is not None:
                cluster = miner.match(
                    parsed.drain_message, full_search_strategy="never"
                )
                if cluster is not None:
                    template = cluster.get_template()
                    return TemplateMatch(
                        parser_type="drain",
                        template_key=stable_template_key(
                            parsed.base.parser_group, template
                        ),
                        message_template=template,
                        drain_cluster_id=int(cluster.cluster_id),
                        template_frequency=int(cluster.size),
                        template_seen_train=1,
                    )
            return TemplateMatch(
                parser_type="drain_unmatched",
                template_key=stable_template_key(
                    parsed.base.parser_group, parsed.semantic_template
                ),
                message_template=parsed.semantic_template,
                drain_cluster_id=0,
                template_frequency=0,
                template_seen_train=0,
            )
        key = stable_template_key(parsed.base.parser_group, parsed.semantic_template)
        frequency = int(self.direct_template_counts.get(key, 0))
        return TemplateMatch(
            parser_type=(
                "missing" if parsed.base.message_format == "blank" else "structured"
            ),
            template_key=key,
            message_template=parsed.semantic_template,
            drain_cluster_id=0,
            template_frequency=frequency,
            template_seen_train=int(frequency > 0),
        )

    def feature_record(self, raw: dict[str, Any]) -> dict[str, Any]:
        parsed = parse_v5_log(raw, max_message_chars=self.settings.max_message_chars)
        base = parsed.base
        match = self.match_parsed(parsed)
        schema_frequency = int(self.schema_counts.get(parsed.schema_id, 0))
        semantic_frequency = int(
            self.semantic_template_counts.get(parsed.semantic_template_id, 0)
        )
        semantic_field_count = max(
            base.semantic_field_count, parsed.security_field_count
        )
        return {
            "event_id": _text(raw.get("event_id")),
            "parser_group": base.parser_group,
            "message_format": base.message_format,
            "parser_type": match.parser_type,
            "template_id": match.template_key,
            "message_template": match.message_template,
            "semantic_action": parsed.event_action,
            "network_protocol": parsed.network_protocol,
            "event_code": parsed.event_code,
            "event_name": base.event_name,
            "dst_port_bucket": port_bucket(parsed.destination_port),
            "http_method": parsed.http_method,
            "http_status_bucket": http_status_bucket(parsed.http_status),
            "source_zone": base.source_zone,
            "destination_zone": base.destination_zone,
            "src_port_from_message": parsed.source_port,
            "dst_port_number": parsed.destination_port,
            "dst_port_missing": int(parsed.destination_port < 0),
            "event_code_present": int(parsed.event_code != MISSING_CATEGORY),
            "semantic_action_present": int(parsed.event_action != MISSING_CATEGORY),
            "network_protocol_present": int(
                parsed.network_protocol != MISSING_CATEGORY
            ),
            "semantic_field_count": semantic_field_count,
            "parse_success": int(semantic_field_count > 0),
            "template_seen_train": match.template_seen_train,
            "template_frequency_log1p": float(math.log1p(match.template_frequency)),
            "template_wildcard_count": int(match.message_template.count("<")),
            "message_token_count": int(len(base.normalized_message.split())),
            "drain_cluster_id": match.drain_cluster_id,
            "is_auth_failure": int(
                base.is_auth_failure == 1
                or parsed.event_reason
                in {"authentication_failure", "invalid_passcode", "invalid_password"}
                or parsed.event_outcome == "failure"
                and parsed.authentication_present == 1
            ),
            "is_network_denied": int(
                parsed.event_action in {"deny", "reject", "drop", "block"}
            ),
            "is_process_creation": base.is_process_creation,
            "is_privileged_logon": base.is_privileged_logon,
            "structured_parser": parsed.structured_parser,
            "payload_parse_status": parsed.payload_parse_status,
            "schema_id": parsed.schema_id,
            "semantic_template_id": parsed.semantic_template_id,
            "event_category_v5": parsed.event_category,
            "event_type_v5": parsed.event_type,
            "event_action_v5": parsed.event_action,
            "event_outcome_v5": parsed.event_outcome,
            "event_reason_v5": parsed.event_reason,
            "authentication_factor": parsed.authentication_factor,
            "service_name_v5": parsed.service_name,
            "application_name_v5": parsed.application_name,
            "rule_name_v5": parsed.rule_name,
            "threat_category_v5": parsed.threat_category,
            "structured_field_count": parsed.structured_field_count,
            "security_field_count": parsed.security_field_count,
            "payload_parse_success": parsed.payload_parse_success,
            "payload_parse_error": parsed.payload_parse_error,
            "schema_seen_train": int(schema_frequency > 0),
            "schema_frequency_log1p": float(math.log1p(schema_frequency)),
            "semantic_template_seen_train": int(semantic_frequency > 0),
            "semantic_template_frequency_log1p": float(math.log1p(semantic_frequency)),
            "source_ip_in_message": parsed.source_ip_present,
            "destination_ip_in_message": parsed.destination_ip_present,
            "event_severity_number": parsed.event_severity,
            "malware_present": parsed.malware_present,
            "detection_present": parsed.detection_present,
            "authentication_present": parsed.authentication_present,
            "rule_name_present": parsed.rule_name_present,
            "user_present_in_payload": parsed.user_present_in_payload,
            "process_present_in_payload": parsed.process_present_in_payload,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "version": V5_TEMPLATE_MODEL_VERSION,
            "settings": asdict(self.settings),
            "fitted_rows": self.fitted_rows,
            "drain_rows": self.drain_rows,
            "direct_rows": self.direct_rows,
            "groups": len(self.miners),
            "drain_clusters": int(
                sum(len(miner.drain.clusters) for miner in self.miners.values())
            ),
            "direct_templates": len(self.direct_template_counts),
            "structured_schemas": len(self.schema_counts),
            "semantic_templates": len(self.semantic_template_counts),
        }

    def save(self, model_dir: Path, *, force: bool = False) -> dict[str, Any]:
        if model_dir.exists():
            has_files = any(model_dir.iterdir())
            if has_files and not force:
                raise FileExistsError(
                    f"{model_dir} exists; pass force=True to replace it"
                )
            if has_files:
                shutil.rmtree(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        groups: dict[str, dict[str, Any]] = {}
        from drain3.memory_buffer_persistence import MemoryBufferPersistence

        for parser_group, miner in self.miners.items():
            persistence = MemoryBufferPersistence()
            miner.persistence_handler = persistence
            miner.save_state("final")
            if persistence.state is None:
                raise RuntimeError(f"Drain3 did not serialize group {parser_group}")
            filename = hashlib.sha1(parser_group.encode("utf-8")).hexdigest() + ".bin"
            (model_dir / filename).write_bytes(persistence.state)
            groups[parser_group] = {
                "file": filename,
                "clusters": len(miner.drain.clusters),
                "rows": int(miner.drain.get_total_cluster_size()),
            }
            miner.persistence_handler = None

        manifest = {
            **self.summary(),
            "groups_manifest": groups,
            "direct_template_counts": dict(self.direct_template_counts),
            "schema_counts": dict(self.schema_counts),
            "semantic_template_counts": dict(self.semantic_template_counts),
        }
        (model_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return manifest

    @classmethod
    def load(cls, model_dir: Path) -> "V5GroupedDrainModel":
        manifest_path = model_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("version") != V5_TEMPLATE_MODEL_VERSION:
            raise ValueError(
                f"Unsupported V5 template model version: {manifest.get('version')}"
            )
        model = cls(DrainSettings(**manifest["settings"]))
        model.fitted_rows = int(manifest.get("fitted_rows", 0))
        model.drain_rows = int(manifest.get("drain_rows", 0))
        model.direct_rows = int(manifest.get("direct_rows", 0))
        model.direct_template_counts.update(manifest.get("direct_template_counts", {}))
        model.schema_counts.update(manifest.get("schema_counts", {}))
        model.semantic_template_counts.update(
            manifest.get("semantic_template_counts", {})
        )

        from drain3.memory_buffer_persistence import MemoryBufferPersistence
        from drain3.template_miner import TemplateMiner

        for parser_group, details in manifest["groups_manifest"].items():
            state = (model_dir / details["file"]).read_bytes()
            persistence = MemoryBufferPersistence()
            persistence.state = state
            miner = TemplateMiner(persistence, model._config())
            miner.persistence_handler = None
            model.miners[parser_group] = miner
        return model
