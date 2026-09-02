from __future__ import annotations

import re
import zlib
from dataclasses import dataclass
from typing import Any, Iterable

from soc_threat.log_semantics import MISSING_CATEGORY, _text, parse_log, port_bucket


PAD_TOKEN_ID = 0
EMPTY_TOKEN_ID = 1
DEFAULT_HASH_BUCKETS = 65_536
DEFAULT_MAX_TOKENS = 96

SANITIZER_PATTERN = re.compile(r"(?i)(USER|ORG|CRED|HOST)(?:-[A-Za-z0-9]+)+")
UUID_PATTERN = re.compile(
    r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
)
IPV4_PATTERN = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
IPV6_PATTERN = re.compile(
    r"(?i)(?<![0-9a-f:])(?:[0-9a-f]{0,4}:){2,}[0-9a-f]{0,4}(?![0-9a-f:])"
)
TIMESTAMP_PATTERN = re.compile(
    r"(?i)\b\d{4}-\d{2}-\d{2}[t ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:z|[+-]\d{2}:?\d{2})?\b"
)
EMAIL_PATTERN = re.compile(r"(?i)\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
LONG_HEX_PATTERN = re.compile(r"(?i)\b(?:0x)?[0-9a-f]{12,}\b")
LONG_NUMBER_PATTERN = re.compile(r"(?<![A-Za-z])\d{4,}(?![A-Za-z])")
CAMEL_BOUNDARY_PATTERN = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
LEXICAL_TOKEN_PATTERN = re.compile(r"[a-z][a-z0-9_]{1,39}|\d{1,3}")
FIELD_KEY_PATTERN = re.compile(
    r"(?i)(?:[\"']|<)?([A-Za-z][A-Za-z0-9_.-]{1,47})(?:[\"']|>)?\s*[:=]"
)
XML_TAG_PATTERN = re.compile(r"(?i)<([A-Za-z][A-Za-z0-9_.:-]{1,47})(?:\s|>)")

SECURITY_SIGNAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("malware", re.compile(r"(?i)\bmalware\b")),
    ("potentially_harmful", re.compile(r"(?i)\bpotentially[ _-]+harmful\b")),
    ("threat", re.compile(r"(?i)\bthreat\b")),
    ("authentication", re.compile(r"(?i)\bauth(?:entication)?\b")),
    ("duo_push", re.compile(r"(?i)\bduo[ _-]+push\b")),
    ("no_response", re.compile(r"(?i)\bno[ _-]+response\b")),
    ("invalid_passcode", re.compile(r"(?i)\binvalid[ _-]+passcode\b")),
    ("powershell", re.compile(r"(?i)\bpowershell(?:\.exe)?\b")),
    ("process", re.compile(r"(?i)\bprocess\b")),
)


@dataclass(frozen=True)
class ContentEncoding:
    raw_token_ids: tuple[int, ...]
    field_token_ids: tuple[int, ...]
    raw_token_count: int
    field_token_count: int
    content_family: str
    content_action: str
    content_event_code: str
    content_protocol: str
    content_has_threat: int
    content_has_authentication: int
    content_has_potentially_harmful: int


def _replace_sanitizer(match: re.Match[str]) -> str:
    return f" {match.group(1).lower()}_token "


def normalize_content_text(message: str, *, max_chars: int = 8192) -> str:
    """Mask volatile entities while preserving security-bearing words."""

    if len(message) > max_chars:
        head_size = max_chars * 3 // 4
        message = message[:head_size] + " " + message[-(max_chars - head_size) :]
    text = CAMEL_BOUNDARY_PATTERN.sub(" ", message)
    text = TIMESTAMP_PATTERN.sub(" timestamp_token ", text)
    text = UUID_PATTERN.sub(" uuid_token ", text)
    text = EMAIL_PATTERN.sub(" email_token ", text)
    text = IPV4_PATTERN.sub(" ip_token ", text)
    text = IPV6_PATTERN.sub(" ipv6_token ", text)
    text = SANITIZER_PATTERN.sub(_replace_sanitizer, text)
    text = LONG_HEX_PATTERN.sub(" hex_token ", text)
    text = LONG_NUMBER_PATTERN.sub(" number_token ", text)
    return text.lower()


def _bounded_head_tail(values: list[str], limit: int) -> list[str]:
    if len(values) <= limit:
        return values
    head = (limit + 1) // 2
    return values[:head] + values[-(limit - head) :]


def lexical_tokens(message: str, *, limit: int = 64) -> list[str]:
    normalized = normalize_content_text(message)
    return _bounded_head_tail(LEXICAL_TOKEN_PATTERN.findall(normalized), limit)


def _normalize_field_key(value: str) -> list[str]:
    value = CAMEL_BOUNDARY_PATTERN.sub("_", value)
    value = SANITIZER_PATTERN.sub("token", value)
    return [
        part.lower()
        for part in re.split(r"[^A-Za-z0-9]+", value)
        if len(part) > 1 and not part.isdigit()
    ]


def extract_field_tokens(message: str, *, limit: int = 32) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for pattern in (FIELD_KEY_PATTERN, XML_TAG_PATTERN):
        for match in pattern.finditer(message[:8192]):
            for part in _normalize_field_key(match.group(1)):
                token = f"field_{part}"
                if token not in seen:
                    seen.add(token)
                    tokens.append(token)
                if len(tokens) >= limit:
                    return tokens
    return tokens


def _security_signals(message: str) -> list[str]:
    return [
        name for name, pattern in SECURITY_SIGNAL_PATTERNS if pattern.search(message)
    ]


def _context_tokens(
    raw: dict[str, Any], message: str
) -> tuple[list[str], dict[str, Any]]:
    parsed = parse_log(raw, max_message_chars=4096)
    context = [f"format_{parsed.message_format}"]
    if parsed.semantic_action != MISSING_CATEGORY:
        context.append(f"action_{parsed.semantic_action}")
    if parsed.network_protocol != MISSING_CATEGORY:
        context.append(f"protocol_{parsed.network_protocol}")
    if parsed.event_code != MISSING_CATEGORY:
        context.append(f"event_code_{parsed.event_code}")
    if parsed.event_name != MISSING_CATEGORY:
        context.append(f"event_name_{parsed.event_name}")
    if parsed.http_method != MISSING_CATEGORY:
        context.append(f"http_method_{parsed.http_method.lower()}")
    if parsed.dst_port >= 0:
        context.append(f"dst_port_{port_bucket(parsed.dst_port)}")

    signals = _security_signals(message)
    context.extend(f"signal_{value}" for value in signals)
    # VPC's trailing OK is a record/log status, not a successful security outcome.
    if parsed.message_format == "vpc_flow":
        parts = message.strip().split()
        if len(parts) >= 14:
            context.append(f"log_status_{parts[13].strip().lower()}")

    audit = {
        "content_family": parsed.message_format,
        "content_action": parsed.semantic_action,
        "content_event_code": parsed.event_code,
        "content_protocol": parsed.network_protocol,
        "content_has_threat": int(
            any(
                value in signals
                for value in ("malware", "potentially_harmful", "threat")
            )
        ),
        "content_has_authentication": int(
            "authentication" in signals
            or parsed.is_auth_failure == 1
            or parsed.event_name
            in {"logon_failure", "logon_success", "credential_validation"}
        ),
        "content_has_potentially_harmful": int("potentially_harmful" in signals),
    }
    return context, audit


def _character_symbols(word: str) -> Iterable[str]:
    if len(word) < 4 or word.endswith("_token"):
        return ()
    padded = f"<{word}>"
    candidates: list[str] = []
    for size in (3, 4, 5):
        if len(padded) < size:
            continue
        starts = {0, max(0, (len(padded) - size) // 2), len(padded) - size}
        candidates.extend(
            f"c:{padded[start : start + size]}" for start in sorted(starts)
        )
    return candidates


def symbol_sequence(
    words: list[str], context: list[str], *, max_tokens: int
) -> list[str]:
    context_limit = min(len(context), max(8, max_tokens // 4))
    word_limit = min(len(words), max(16, max_tokens * 5 // 12))
    bigram_limit = max(4, max_tokens // 8)
    selected_words = _bounded_head_tail(words, word_limit)
    symbols: list[str] = [f"ctx:{value}" for value in context[:context_limit]]
    symbols.extend(f"w:{word}" for word in selected_words)
    symbols.extend(
        f"b:{left}_{right}"
        for left, right in list(zip(selected_words, selected_words[1:]))[:bigram_limit]
    )
    char_groups = [list(_character_symbols(word)) for word in selected_words]
    for position in range(9):
        for group in char_groups:
            if position < len(group):
                symbols.append(group[position])
                if len(symbols) >= max_tokens:
                    return symbols[:max_tokens]
    return symbols[:max_tokens]


def hash_symbol(symbol: str, *, buckets: int = DEFAULT_HASH_BUCKETS) -> int:
    if not 4 <= buckets <= 65_536:
        raise ValueError("hash buckets must be between 4 and 65536 for uint16 storage")
    return int(zlib.crc32(symbol.encode("utf-8")) % (buckets - 2)) + 2


def encode_symbols(
    symbols: list[str],
    *,
    buckets: int = DEFAULT_HASH_BUCKETS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> tuple[tuple[int, ...], int]:
    ids = [hash_symbol(value, buckets=buckets) for value in symbols[:max_tokens]]
    count = len(ids)
    if not ids:
        ids = [EMPTY_TOKEN_ID]
        count = 1
    ids.extend([PAD_TOKEN_ID] * (max_tokens - len(ids)))
    return tuple(ids), count


def encode_log_content(
    raw: dict[str, Any],
    *,
    buckets: int = DEFAULT_HASH_BUCKETS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> ContentEncoding:
    message = _text(raw.get("message_sanitized"))
    words = lexical_tokens(message)
    raw_symbols = symbol_sequence(words, [], max_tokens=max_tokens)
    raw_ids, raw_count = encode_symbols(
        raw_symbols, buckets=buckets, max_tokens=max_tokens
    )

    context, audit = _context_tokens(raw, message)
    context.extend(extract_field_tokens(message))
    field_symbols = symbol_sequence(words, context, max_tokens=max_tokens)
    field_ids, field_count = encode_symbols(
        field_symbols, buckets=buckets, max_tokens=max_tokens
    )
    return ContentEncoding(
        raw_token_ids=raw_ids,
        field_token_ids=field_ids,
        raw_token_count=raw_count,
        field_token_count=field_count,
        **audit,
    )
