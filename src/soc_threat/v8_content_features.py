from __future__ import annotations

import re
from dataclasses import dataclass

from soc_threat.content_features import (
    LEXICAL_TOKEN_PATTERN,
    _normalize_field_key,
    encode_symbols,
    normalize_content_text,
    symbol_sequence,
)


V8_CONTENT_VIEWS = ("head", "middle", "tail", "key_value")
DEFAULT_V8_TOKENS_PER_VIEW = 64
DEFAULT_V8_MESSAGE_CHARS_PER_VIEW = 4096

KEY_VALUE_PATTERN = re.compile(
    r"(?is)(?:[\"']|<)?"
    r"(?P<key>[A-Za-z][A-Za-z0-9_.:-]{1,47})"
    r"(?:[\"']|>)?\s*[:=]\s*"
    r"(?P<value>\"[^\"\r\n]{0,192}\"|'[^'\r\n]{0,192}'|[^,\s{}\]<>\"']{1,192})"
)
XML_VALUE_PATTERN = re.compile(
    r"(?is)<(?P<key>[A-Za-z][A-Za-z0-9_.:-]{1,47})(?:\s[^<>]*)?>"
    r"(?P<value>[^<>]{1,192})</"
)
SECURITY_FIELD_PARTS = {
    "action",
    "act",
    "status",
    "result",
    "reason",
    "outcome",
    "decision",
    "event",
    "eventid",
    "event_id",
    "eventcode",
    "event_code",
    "process",
    "processname",
    "process_name",
    "command",
    "commandline",
    "command_line",
    "protocol",
    "severity",
    "category",
    "operation",
}


@dataclass(frozen=True)
class V8ContentEncoding:
    multiview_token_ids: tuple[int, ...]
    head_token_count: int
    middle_token_count: int
    tail_token_count: int
    key_value_token_count: int


def _character_regions(message: str, size: int) -> tuple[str, str, str]:
    if len(message) <= size:
        return message, message, message
    middle_start = max(0, (len(message) - size) // 2)
    return (
        message[:size],
        message[middle_start : middle_start + size],
        message[-size:],
    )


def _window_words(text: str, *, position: str, limit: int) -> list[str]:
    words = LEXICAL_TOKEN_PATTERN.findall(normalize_content_text(text))
    if len(words) <= limit:
        return words
    if position == "head":
        return words[:limit]
    if position == "tail":
        return words[-limit:]
    start = max(0, (len(words) - limit) // 2)
    return words[start : start + limit]


def _bounded_regions(message: str, size: int) -> str:
    regions = _character_regions(message, size)
    if regions[0] == regions[1] == regions[2]:
        return regions[0]
    return "\n".join(regions)


def _normalized_value_words(value: str) -> list[str]:
    stripped = value.strip().strip("\"'")
    return LEXICAL_TOKEN_PATTERN.findall(
        normalize_content_text(stripped, max_chars=512)
    )[:3]


def _key_value_groups(message: str, *, chars_per_region: int) -> list[list[str]]:
    searchable = _bounded_regions(message, chars_per_region)
    groups: list[list[str]] = []
    seen: set[tuple[str, str]] = set()
    for pattern in (KEY_VALUE_PATTERN, XML_VALUE_PATTERN):
        for match in pattern.finditer(searchable):
            key_parts = _normalize_field_key(match.group("key"))
            values = _normalized_value_words(match.group("value"))
            if not key_parts or not values:
                continue
            key = "_".join(key_parts[:3])
            signature = (key, "_".join(values))
            if signature in seen:
                continue
            seen.add(signature)
            group = [f"kv_key:{key}"]
            for value in values:
                group.extend((f"kv_value:{value}", f"kv_pair:{key}={value}"))
            groups.append(group)
    return groups


def _distributed_indices(length: int, count: int) -> list[int]:
    if count >= length:
        return list(range(length))
    if count <= 1:
        return [0]
    return [round(index * (length - 1) / (count - 1)) for index in range(count)]


def key_value_symbols(
    message: str,
    *,
    max_tokens: int,
    chars_per_region: int = DEFAULT_V8_MESSAGE_CHARS_PER_VIEW,
) -> list[str]:
    groups = _key_value_groups(message, chars_per_region=chars_per_region)
    if not groups:
        return []

    prioritized: list[list[str]] = []
    remaining: list[list[str]] = []
    for group in groups:
        key = group[0].removeprefix("kv_key:")
        key_parts = set(key.split("_"))
        if key in SECURITY_FIELD_PARTS or key_parts & SECURITY_FIELD_PARTS:
            prioritized.append(group)
        else:
            remaining.append(group)

    selected: list[list[str]] = []
    used_tokens = 0
    for group in prioritized:
        if used_tokens + len(group) > max_tokens:
            break
        selected.append(group)
        used_tokens += len(group)

    available = max_tokens - used_tokens
    if available > 0 and remaining:
        average_width = max(1, round(sum(map(len, remaining)) / len(remaining)))
        group_slots = max(1, available // average_width)
        for index in _distributed_indices(len(remaining), min(group_slots, len(remaining))):
            group = remaining[index]
            if used_tokens + len(group) > max_tokens:
                continue
            selected.append(group)
            used_tokens += len(group)

    return [symbol for group in selected for symbol in group][:max_tokens]


def encode_multiview_content(
    message: str,
    *,
    buckets: int,
    tokens_per_view: int = DEFAULT_V8_TOKENS_PER_VIEW,
    chars_per_view: int = DEFAULT_V8_MESSAGE_CHARS_PER_VIEW,
) -> V8ContentEncoding:
    if tokens_per_view < 16:
        raise ValueError("tokens_per_view must be at least 16")
    if chars_per_view < 256:
        raise ValueError("chars_per_view must be at least 256")

    head_text, middle_text, tail_text = _character_regions(message, chars_per_view)
    word_limit = max(8, tokens_per_view * 5 // 12)
    word_views = (
        _window_words(head_text, position="head", limit=word_limit),
        _window_words(middle_text, position="middle", limit=word_limit),
        _window_words(tail_text, position="tail", limit=word_limit),
    )
    symbol_views = [
        symbol_sequence(words, [], max_tokens=tokens_per_view) for words in word_views
    ]
    symbol_views.append(
        key_value_symbols(
            message,
            max_tokens=tokens_per_view,
            chars_per_region=chars_per_view,
        )
    )

    ids: list[int] = []
    counts: list[int] = []
    for symbols in symbol_views:
        encoded, count = encode_symbols(
            symbols,
            buckets=buckets,
            max_tokens=tokens_per_view,
        )
        ids.extend(encoded)
        counts.append(count)
    return V8ContentEncoding(
        multiview_token_ids=tuple(ids),
        head_token_count=counts[0],
        middle_token_count=counts[1],
        tail_token_count=counts[2],
        key_value_token_count=counts[3],
    )
