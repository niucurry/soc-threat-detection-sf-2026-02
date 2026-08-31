from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


MISSING_CATEGORY = "__MISSING__"
TEMPLATE_MODEL_VERSION = 1

WINDOWS_EVENT_NAMES = {
    "4624": "logon_success",
    "4625": "logon_failure",
    "4634": "logoff",
    "4648": "explicit_credentials_logon",
    "4672": "special_privileges",
    "4688": "process_creation",
    "4689": "process_exit",
    "4720": "user_created",
    "4722": "user_enabled",
    "4724": "password_reset",
    "4725": "user_disabled",
    "4726": "user_deleted",
    "4740": "account_locked",
    "4768": "kerberos_tgt_request",
    "4769": "kerberos_service_ticket",
    "4771": "kerberos_preauth_failure",
    "4776": "credential_validation",
    "5156": "network_connection_allowed",
    "5157": "network_connection_blocked",
}

PLACEHOLDER_PATTERNS = (
    (re.compile(r"(?i)USER(?:-[A-Za-z0-9]+)+"), "<USER>"),
    (re.compile(r"(?i)ORG(?:-[A-Za-z0-9]+)+"), "<ORG>"),
    (re.compile(r"(?i)CRED(?:-[A-Za-z0-9]+)+"), "<CREDENTIAL>"),
    (re.compile(r"(?i)HOST(?:-[A-Za-z0-9]+)+"), "<HOST>"),
)

IPV4_PATTERN = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
IPV6_PATTERN = re.compile(r"(?i)(?<![0-9a-f:])(?:[0-9a-f]{0,4}:){2,}[0-9a-f]{0,4}(?![0-9a-f:])")
UUID_PATTERN = re.compile(r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")
EMAIL_PATTERN = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
URL_PATTERN = re.compile(r"(?i)\b(?:https?|ftp)://[^\s\"'<>]+")
MAC_PATTERN = re.compile(r"(?i)\b(?:[0-9a-f]{2}[:-]){5}[0-9a-f]{2}\b")
TIMESTAMP_PATTERN = re.compile(
    r"(?i)\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b"
)
LONG_HEX_PATTERN = re.compile(r"(?i)\b[0-9a-f]{16,}\b")
STANDALONE_NUMBER_PATTERN = re.compile(r"(?<![A-Za-z0-9_])\d+(?:\.\d+)?(?![A-Za-z0-9_])")


@dataclass(frozen=True)
class ParsedLog:
    message_format: str
    semantic_action: str
    network_protocol: str
    event_code: str
    event_name: str
    src_port_from_message: int
    dst_port: int
    source_zone: str
    destination_zone: str
    http_method: str
    http_status: int
    parser_group: str
    normalized_message: str
    semantic_template: str
    semantic_field_count: int
    is_auth_failure: int
    is_network_denied: int
    is_process_creation: int
    is_privileged_logon: int


@dataclass(frozen=True)
class TemplateMatch:
    parser_type: str
    template_key: str
    message_template: str
    drain_cluster_id: int
    template_frequency: int
    template_seen_train: int


@dataclass(frozen=True)
class DrainSettings:
    similarity_threshold: float = 0.50
    depth: int = 4
    max_children: int = 100
    max_clusters_per_group: int = 5000
    max_message_chars: int = 2048
    max_groups: int = 128


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _category(value: Any) -> str:
    text = _text(value)
    return text if text else MISSING_CATEGORY


def _valid_port(value: str | int | None) -> int:
    try:
        port = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return -1
    return port if 0 <= port <= 65535 else -1


def normalize_for_template(message: str, *, max_chars: int = 2048) -> str:
    """Mask high-cardinality values while preserving security semantics."""

    normalized = message.replace("\x00", " ").replace("\r", " ").replace("\n", " ")
    normalized = TIMESTAMP_PATTERN.sub("<TIMESTAMP>", normalized)
    normalized = UUID_PATTERN.sub("<UUID>", normalized)
    normalized = EMAIL_PATTERN.sub("<EMAIL>", normalized)
    normalized = URL_PATTERN.sub("<URL>", normalized)
    normalized = IPV4_PATTERN.sub("<IP>", normalized)
    normalized = IPV6_PATTERN.sub("<IPV6>", normalized)
    normalized = MAC_PATTERN.sub("<MAC>", normalized)
    normalized = LONG_HEX_PATTERN.sub("<HEX>", normalized)
    for pattern, replacement in PLACEHOLDER_PATTERNS:
        normalized = pattern.sub(replacement, normalized)
    normalized = STANDALONE_NUMBER_PATTERN.sub("<NUM>", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized[:max_chars]


def detect_message_format(
    message: str,
    *,
    pipeline: str,
    vendor_name: str,
    product_name: str,
) -> str:
    if not message:
        return "blank"
    lowered = message[:4096].lower()
    product_lower = product_name.lower()
    vendor_lower = vendor_name.lower()
    stripped = message.lstrip()
    if (
        "windows" in product_lower
        or "security-auditing" in lowered
        or "winlogbeat" in lowered
    ):
        return "windows_xml" if stripped.startswith("<") else "windows_json"
    if "cef:" in lowered:
        return "cef"
    if (
        "asa firewall" in product_lower
        or re.search(r"(?i)\bdeny\s+(?:tcp|udp|icmp)\s+src\b", message)
        or "%asa-" in lowered
    ):
        return "asa"
    if (
        "aws vpc security" in product_lower
        or re.search(r"(?i)\s(?:ACCEPT|REJECT)\s+(?:OK|NODATA|SKIPDATA)\s*$", message)
    ):
        return "vpc_flow"
    if (
        "linux" in vendor_lower
        or "linux" in product_lower
        or "pam_unix(" in lowered
        or "sshd[" in lowered
    ):
        return "linux_syslog"
    if "{" in message and "}" in message:
        return "json"
    if re.search(r"(?i)\b[A-Za-z_][\w.-]*\s*=\s*[^\s]+", message):
        return "key_value"
    if pipeline.lower() == "syslog" or re.match(r"^<\d{1,3}>", stripped):
        return "syslog_text"
    return "free_text"


def extract_action(message: str) -> str:
    lowered = message.lower()
    patterns = (
        (r"\bdeny(?:ied)?\b", "deny"),
        (r"\breject(?:ed)?\b", "reject"),
        (r"\bdrop(?:ped)?\b", "drop"),
        (r"\bblock(?:ed)?\b|\bblock-url\b", "block"),
        (r"\bauthentication failure\b|\blogin failed\b|\blogon failure\b|\bfailed\b|\bfailure\b", "fail"),
        (r"\ballow(?:ed)?\b|\bpermit(?:ted)?\b", "allow"),
        (r"\baccept(?:ed)?\b", "accept"),
        (r"\bsuccess(?:ful|fully)?\b", "success"),
    )
    for pattern, action in patterns:
        if re.search(pattern, lowered):
            return action
    return MISSING_CATEGORY


def extract_protocol(message: str) -> str:
    match = re.search(r"(?i)\b(tcp|udp|icmpv6|icmp|sctp|gre|esp|tls|https?|dns)\b", message)
    return match.group(1).lower() if match else MISSING_CATEGORY


def extract_event_code(message: str, message_format: str) -> str:
    patterns = (
        r"(?i)\"(?:event_?id|event\.code|code)\"\s*:\s*\"?(\d{3,8})",
        r"(?i)<[^>]*id[^>]*>\s*(\d{3,8})\s*</",
        r"(?i)\bEventID\s*[=:]\s*\"?(\d{3,8})",
        r"(?i)%ASA-\d-(\d{3,8})",
    )
    for pattern in patterns:
        match = re.search(pattern, message)
        if match:
            return match.group(1)
    if message_format == "cef":
        match = re.search(r"(?i)CEF:\d+\|[^|]*\|[^|]*\|[^|]*\|([^|]{1,64})\|", message)
        if match:
            return match.group(1).strip() or MISSING_CATEGORY
    return MISSING_CATEGORY


def extract_zones_and_ports(message: str, message_format: str) -> tuple[str, str, int, int]:
    source_zone = MISSING_CATEGORY
    destination_zone = MISSING_CATEGORY
    src_port = -1
    dst_port = -1
    asa = re.search(
        r"(?i)\bsrc\s+([^:\s]+):[^\s/]+/([0-9]{1,5}).*?\bdst\s+([^:\s]+):[^\s/]+/([0-9]{1,5})",
        message,
    )
    if asa:
        source_zone = asa.group(1).lower()
        destination_zone = asa.group(3).lower()
        src_port = _valid_port(asa.group(2))
        dst_port = _valid_port(asa.group(4))
        return source_zone, destination_zone, src_port, dst_port

    address_ports = re.search(
        r"(?i)\bsrc\s+[^\s/]+/([0-9]{1,5}).*?\bdst\s+[^\s/]+/([0-9]{1,5})",
        message,
    )
    if address_ports:
        src_port = _valid_port(address_ports.group(1))
        dst_port = _valid_port(address_ports.group(2))
        return source_zone, destination_zone, src_port, dst_port

    generic_dst = re.search(
        r"(?i)\b(?:dst_port|destination_port|dport)\s*[=:]\s*\"?([0-9]{1,5})",
        message,
    )
    if generic_dst:
        dst_port = _valid_port(generic_dst.group(1))

    if message_format == "vpc_flow":
        tokens = message.split()
        if len(tokens) >= 8:
            src_port = _valid_port(tokens[5])
            dst_port = _valid_port(tokens[6])
    return source_zone, destination_zone, src_port, dst_port


def extract_http(message: str) -> tuple[str, int]:
    method = MISSING_CATEGORY
    status = -1
    request = re.search(
        r"(?i)(?:\"|\b)(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS|CONNECT|TRACE)\s+[^\s\"]+",
        message,
    )
    if request:
        method = request.group(1).upper()
    status_match = re.search(r"(?i)HTTP/\d(?:\.\d)?\"?\s+([1-5][0-9]{2})\b", message)
    if status_match:
        status = int(status_match.group(1))
    elif method != MISSING_CATEGORY:
        fallback = re.search(r"\"\s+([1-5][0-9]{2})\s+\d+", message)
        if fallback:
            status = int(fallback.group(1))
    return method, status


def _key_signature(message: str, message_format: str) -> str:
    if message_format in {"json", "windows_json"}:
        keys = re.findall(r"\"([A-Za-z_][A-Za-z0-9_.-]{1,40})\"\s*:", message[:12000])
    elif message_format == "windows_xml":
        keys = re.findall(r"</?([A-Za-z_][A-Za-z0-9_.:-]{1,40})", message[:12000])
    else:
        keys = re.findall(r"\b([A-Za-z_][A-Za-z0-9_.-]{1,40})\s*=", message[:6000])
    stable: list[str] = []
    seen: set[str] = set()
    for key in keys:
        lowered = key.lower()
        if lowered.startswith(("user-", "org-", "cred-", "host-")):
            continue
        if lowered not in seen:
            stable.append(lowered)
            seen.add(lowered)
        if len(stable) >= 10:
            break
    return ",".join(stable) if stable else MISSING_CATEGORY


def _semantic_template(
    message: str,
    message_format: str,
    event_code: str,
    event_name: str,
    action: str,
    protocol: str,
    http_method: str,
) -> str:
    key_signature = _key_signature(message, message_format)
    parts = [f"format={message_format}"]
    for name, value in (
        ("event_code", event_code),
        ("event_name", event_name),
        ("action", action),
        ("protocol", protocol),
        ("http_method", http_method),
        ("keys", key_signature),
    ):
        if value != MISSING_CATEGORY:
            parts.append(f"{name}={value}")
    return " ".join(parts)


def parse_log(raw: dict[str, Any], *, max_message_chars: int = 2048) -> ParsedLog:
    message = _text(raw.get("message_sanitized"))
    pipeline = _text(raw.get("pipeline"))
    vendor_name = _text(raw.get("vendor_name"))
    product_name = _text(raw.get("product_name"))
    message_format = detect_message_format(
        message,
        pipeline=pipeline,
        vendor_name=vendor_name,
        product_name=product_name,
    )
    action = extract_action(message) if message else MISSING_CATEGORY
    protocol = extract_protocol(message) if message else MISSING_CATEGORY
    event_code = extract_event_code(message, message_format) if message else MISSING_CATEGORY
    event_name = WINDOWS_EVENT_NAMES.get(event_code, MISSING_CATEGORY)
    source_zone, destination_zone, src_port, dst_port = extract_zones_and_ports(
        message, message_format
    )
    http_method, http_status = extract_http(message)
    parser_group = "|".join(
        [
            _category(pipeline),
            _category(vendor_name),
            _category(product_name),
            message_format,
        ]
    )
    normalized = normalize_for_template(message, max_chars=max_message_chars)
    template = _semantic_template(
        message,
        message_format,
        event_code,
        event_name,
        action,
        protocol,
        http_method,
    )
    semantic_values: Iterable[Any] = (
        action,
        protocol,
        event_code,
        src_port,
        dst_port,
        source_zone,
        destination_zone,
        http_method,
        http_status,
    )
    semantic_count = sum(
        value not in {MISSING_CATEGORY, -1, ""} for value in semantic_values
    )
    lowered = message.lower()
    is_auth_failure = int(
        event_name in {"logon_failure", "kerberos_preauth_failure"}
        or "authentication failure" in lowered
        or "login failed" in lowered
        or "logon failure" in lowered
    )
    is_network_denied = int(action in {"deny", "reject", "drop", "block"})
    return ParsedLog(
        message_format=message_format,
        semantic_action=action,
        network_protocol=protocol,
        event_code=event_code,
        event_name=event_name,
        src_port_from_message=src_port,
        dst_port=dst_port,
        source_zone=source_zone,
        destination_zone=destination_zone,
        http_method=http_method,
        http_status=http_status,
        parser_group=parser_group,
        normalized_message=normalized,
        semantic_template=template,
        semantic_field_count=semantic_count,
        is_auth_failure=is_auth_failure,
        is_network_denied=is_network_denied,
        is_process_creation=int(event_name == "process_creation"),
        is_privileged_logon=int(event_name == "special_privileges"),
    )


def should_use_drain(parsed: ParsedLog) -> bool:
    return parsed.message_format in {
        "asa",
        "cef",
        "key_value",
        "linux_syslog",
        "syslog_text",
        "free_text",
    } and bool(parsed.normalized_message)


def stable_template_key(parser_group: str, template: str) -> str:
    digest = hashlib.sha1(
        (parser_group + "\x00" + template).encode("utf-8", errors="replace")
    ).hexdigest()[:20]
    return f"tpl_{digest}"


def port_bucket(port: int) -> str:
    if port < 0:
        return MISSING_CATEGORY
    if port <= 1023:
        return "well_known_00000_01023"
    if port <= 49151:
        return "registered_01024_49151"
    return "ephemeral_49152_65535"


def http_status_bucket(status: int) -> str:
    return MISSING_CATEGORY if status < 0 else f"{status // 100}xx"


class GroupedDrainModel:
    """Train-only grouped Drain model plus deterministic structured templates."""

    def __init__(self, settings: DrainSettings | None = None) -> None:
        self.settings = settings or DrainSettings()
        self.miners: dict[str, Any] = {}
        self.direct_template_counts: Counter[str] = Counter()
        self.fitted_rows = 0
        self.drain_rows = 0
        self.direct_rows = 0

    def _config(self) -> Any:
        try:
            from drain3.masking import MaskingInstruction
            from drain3.template_miner_config import TemplateMinerConfig
        except ImportError as exc:  # pragma: no cover - exercised by cloud setup
            raise RuntimeError(
                "Drain3 is required for V4 features. Install requirements-npu.txt."
            ) from exc
        config = TemplateMinerConfig()
        config.drain_sim_th = self.settings.similarity_threshold
        config.drain_depth = self.settings.depth
        config.drain_max_children = self.settings.max_children
        config.drain_max_clusters = self.settings.max_clusters_per_group
        config.parametrize_numeric_tokens = True
        config.snapshot_compress_state = True
        config.masking_instructions = [
            MaskingInstruction(IPV4_PATTERN.pattern, "IP"),
            MaskingInstruction(UUID_PATTERN.pattern, "UUID"),
            MaskingInstruction(r"(?<![A-Za-z0-9_])\d+(?:\.\d+)?(?![A-Za-z0-9_])", "NUM"),
        ]
        return config

    def _new_miner(self) -> Any:
        from drain3.template_miner import TemplateMiner

        # Persistence is intentionally disabled while fitting; otherwise Drain3
        # serializes the entire state after frequent template changes.
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
        parsed = parse_log(raw, max_message_chars=self.settings.max_message_chars)
        self.fitted_rows += 1
        if parsed.message_format == "blank":
            key = stable_template_key(parsed.parser_group, parsed.semantic_template)
            self.direct_template_counts[key] += 1
            self.direct_rows += 1
            return
        if should_use_drain(parsed):
            miner = self._miner_for_fit(parsed.parser_group)
            if miner is not None:
                miner.add_log_message(parsed.normalized_message)
                self.drain_rows += 1
                return
        key = stable_template_key(parsed.parser_group, parsed.semantic_template)
        self.direct_template_counts[key] += 1
        self.direct_rows += 1

    def match_parsed(self, parsed: ParsedLog) -> TemplateMatch:
        if should_use_drain(parsed):
            miner = self.miners.get(parsed.parser_group)
            if miner is not None:
                cluster = miner.match(parsed.normalized_message, full_search_strategy="never")
                if cluster is not None:
                    template = cluster.get_template()
                    return TemplateMatch(
                        parser_type="drain",
                        template_key=stable_template_key(parsed.parser_group, template),
                        message_template=template,
                        drain_cluster_id=int(cluster.cluster_id),
                        template_frequency=int(cluster.size),
                        template_seen_train=1,
                    )
            template = parsed.semantic_template
            return TemplateMatch(
                parser_type="drain_unmatched",
                template_key=stable_template_key(parsed.parser_group, template),
                message_template=template,
                drain_cluster_id=0,
                template_frequency=0,
                template_seen_train=0,
            )
        template = parsed.semantic_template
        key = stable_template_key(parsed.parser_group, template)
        frequency = int(self.direct_template_counts.get(key, 0))
        return TemplateMatch(
            parser_type="missing" if parsed.message_format == "blank" else "structured",
            template_key=key,
            message_template=template,
            drain_cluster_id=0,
            template_frequency=frequency,
            template_seen_train=int(frequency > 0),
        )

    def feature_record(self, raw: dict[str, Any]) -> dict[str, Any]:
        parsed = parse_log(raw, max_message_chars=self.settings.max_message_chars)
        match = self.match_parsed(parsed)
        return {
            "event_id": _text(raw.get("event_id")),
            "parser_group": parsed.parser_group,
            "message_format": parsed.message_format,
            "parser_type": match.parser_type,
            "template_id": match.template_key,
            "message_template": match.message_template,
            "semantic_action": parsed.semantic_action,
            "network_protocol": parsed.network_protocol,
            "event_code": parsed.event_code,
            "event_name": parsed.event_name,
            "dst_port_bucket": port_bucket(parsed.dst_port),
            "http_method": parsed.http_method,
            "http_status_bucket": http_status_bucket(parsed.http_status),
            "source_zone": parsed.source_zone,
            "destination_zone": parsed.destination_zone,
            "src_port_from_message": parsed.src_port_from_message,
            "dst_port_number": parsed.dst_port,
            "dst_port_missing": int(parsed.dst_port < 0),
            "event_code_present": int(parsed.event_code != MISSING_CATEGORY),
            "semantic_action_present": int(parsed.semantic_action != MISSING_CATEGORY),
            "network_protocol_present": int(parsed.network_protocol != MISSING_CATEGORY),
            "semantic_field_count": parsed.semantic_field_count,
            "parse_success": int(parsed.semantic_field_count > 0),
            "template_seen_train": match.template_seen_train,
            "template_frequency_log1p": float(math.log1p(match.template_frequency)),
            "template_wildcard_count": int(match.message_template.count("<")),
            "message_token_count": int(len(parsed.normalized_message.split())),
            "drain_cluster_id": match.drain_cluster_id,
            "is_auth_failure": parsed.is_auth_failure,
            "is_network_denied": parsed.is_network_denied,
            "is_process_creation": parsed.is_process_creation,
            "is_privileged_logon": parsed.is_privileged_logon,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "version": TEMPLATE_MODEL_VERSION,
            "settings": asdict(self.settings),
            "fitted_rows": self.fitted_rows,
            "drain_rows": self.drain_rows,
            "direct_rows": self.direct_rows,
            "groups": len(self.miners),
            "drain_clusters": int(
                sum(len(miner.drain.clusters) for miner in self.miners.values())
            ),
            "direct_templates": len(self.direct_template_counts),
        }

    def save(self, model_dir: Path, *, force: bool = False) -> dict[str, Any]:
        if model_dir.exists():
            has_files = any(model_dir.iterdir())
            if has_files and not force:
                raise FileExistsError(f"{model_dir} exists; pass force=True to replace it")
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
        }
        (model_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return manifest

    @classmethod
    def load(cls, model_dir: Path) -> "GroupedDrainModel":
        manifest_path = model_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("version") != TEMPLATE_MODEL_VERSION:
            raise ValueError(
                f"Unsupported template model version: {manifest.get('version')}"
            )
        model = cls(DrainSettings(**manifest["settings"]))
        model.fitted_rows = int(manifest.get("fitted_rows", 0))
        model.drain_rows = int(manifest.get("drain_rows", 0))
        model.direct_rows = int(manifest.get("direct_rows", 0))
        model.direct_template_counts.update(manifest.get("direct_template_counts", {}))

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
