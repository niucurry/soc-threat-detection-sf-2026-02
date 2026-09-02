from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from soc_threat.content_features import (  # noqa: E402
    encode_log_content,
    hash_symbol,
)


def raw(message: str) -> dict[str, str]:
    return {
        "event_id": "event-1",
        "pipeline": "syslog",
        "vendor_name": "",
        "product_name": "",
        "message_sanitized": message,
    }


class ContentFeatureTests(unittest.TestCase):
    def test_dynamic_network_values_do_not_change_raw_content(self) -> None:
        first = encode_log_content(
            raw("Deny tcp src 10.1.2.3/52131 dst 100.64.1.8/443")
        )
        second = encode_log_content(
            raw("Deny tcp src 172.16.1.5/61234 dst 100.64.3.7/443")
        )
        self.assertEqual(first.raw_token_ids, second.raw_token_ids)

    def test_security_action_changes_content_encoding(self) -> None:
        denied = encode_log_content(raw("Deny tcp src 10.1.2.3 dst 10.2.3.4"))
        allowed = encode_log_content(raw("Allow tcp src 10.1.2.3 dst 10.2.3.4"))
        self.assertNotEqual(denied.raw_token_ids, allowed.raw_token_ids)
        self.assertNotEqual(denied.field_token_ids, allowed.field_token_ids)

    def test_field_aware_encoding_marks_potentially_harmful_content(self) -> None:
        encoded = encode_log_content(
            raw(
                'payload={"categories":["Potentially Harmful"],'
                '"actionTaken":"BLOCKED","severityCode":2}'
            )
        )
        self.assertEqual(encoded.content_has_threat, 1)
        self.assertEqual(encoded.content_has_potentially_harmful, 1)
        self.assertEqual(encoded.content_action, "block")
        self.assertNotEqual(encoded.raw_token_ids, encoded.field_token_ids)

    def test_vpc_ok_is_log_status_not_successful_security_outcome(self) -> None:
        encoded = encode_log_content(
            raw(
                "2 100000000001 eni-123 10.0.0.1 10.0.0.2 "
                "59576 6203 6 1 40 1665100000 1665100030 REJECT OK"
            )
        )
        self.assertEqual(encoded.content_family, "vpc_flow")
        self.assertEqual(encoded.content_action, "reject")
        self.assertIn(hash_symbol("ctx:log_status_ok"), encoded.field_token_ids)
        self.assertNotIn(hash_symbol("ctx:outcome_success"), encoded.field_token_ids)

    def test_hash_encoding_has_fixed_uint16_compatible_width(self) -> None:
        encoded = encode_log_content(raw("authentication failed invalid_passcode"))
        self.assertEqual(len(encoded.raw_token_ids), 96)
        self.assertEqual(len(encoded.field_token_ids), 96)
        self.assertGreater(encoded.raw_token_count, 0)
        self.assertLess(max(encoded.field_token_ids), 65536)
        self.assertIn(hash_symbol("c:<au"), encoded.raw_token_ids)


if __name__ == "__main__":
    unittest.main()
