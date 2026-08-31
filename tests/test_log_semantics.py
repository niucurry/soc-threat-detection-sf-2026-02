from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from soc_threat.log_semantics import (  # noqa: E402
    GroupedDrainModel,
    normalize_for_template,
    parse_log,
)


def raw(message: str, product: str = "ASA Firewall") -> dict[str, str]:
    return {
        "event_id": "evt-1",
        "pipeline": "syslog",
        "vendor_name": "Cisco",
        "product_name": product,
        "message_sanitized": message,
    }


class LogSemanticTests(unittest.TestCase):
    def test_asa_extracts_action_protocol_zones_and_ports(self) -> None:
        parsed = parse_log(
            raw(
                "Deny tcp src outside:10.1.2.3/52131 "
                "dst DMZ:100.64.1.8/443 by access-group edge"
            )
        )
        self.assertEqual(parsed.message_format, "asa")
        self.assertEqual(parsed.semantic_action, "deny")
        self.assertEqual(parsed.network_protocol, "tcp")
        self.assertEqual(parsed.source_zone, "outside")
        self.assertEqual(parsed.destination_zone, "dmz")
        self.assertEqual(parsed.src_port_from_message, 52131)
        self.assertEqual(parsed.dst_port, 443)
        self.assertEqual(parsed.is_network_denied, 1)

    def test_asa_without_zone_still_extracts_ports(self) -> None:
        parsed = parse_log(
            raw("Deny tcp src 10.1.2.3/52131 dst 100.64.1.8/443")
        )
        self.assertEqual(parsed.src_port_from_message, 52131)
        self.assertEqual(parsed.dst_port, 443)

    def test_normalization_masks_changing_network_values(self) -> None:
        first = normalize_for_template(
            "Deny tcp src 10.1.2.3/52131 dst 100.64.1.8/443"
        )
        second = normalize_for_template(
            "Deny tcp src 10.8.5.2/61822 dst 100.64.2.9/80"
        )
        self.assertEqual(first, second)
        self.assertIn("<IP>", first)
        self.assertIn("<NUM>", first)

    def test_normalization_handles_embedded_sanitizer_tokens(self) -> None:
        normalized = normalize_for_template(
            "port=3CRED-25097 host=exampleUSER-8710 account=USER-0010-56507ID"
        )
        self.assertIn("<CREDENTIAL>", normalized)
        self.assertIn("<USER>", normalized)
        self.assertNotIn("25097", normalized)

    def test_windows_event_code_gets_semantic_name(self) -> None:
        parsed = parse_log(
            raw(
                '<Event><System><EventID>4625</EventID></System>'
                '<Message>An account failed to log on.</Message></Event>',
                product="Windows Logs",
            )
        )
        self.assertEqual(parsed.message_format, "windows_xml")
        self.assertEqual(parsed.event_code, "4625")
        self.assertEqual(parsed.event_name, "logon_failure")
        self.assertEqual(parsed.is_auth_failure, 1)

    def test_grouped_drain_template_is_stable_after_save_and_load(self) -> None:
        messages = [
            raw("Deny tcp src outside:10.1.2.3/52131 dst DMZ:100.64.1.8/443"),
            raw("Deny tcp src outside:10.8.5.2/61822 dst DMZ:100.64.2.9/80"),
        ]
        model = GroupedDrainModel()
        for row in messages:
            model.fit_raw(row)
        first = model.feature_record(messages[0])
        second = model.feature_record(messages[1])
        self.assertEqual(first["template_id"], second["template_id"])
        self.assertEqual(first["parser_type"], "drain")
        self.assertEqual(first["template_seen_train"], 1)

        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory) / "model"
            model.save(model_dir)
            restored = GroupedDrainModel.load(model_dir)
            restored_record = restored.feature_record(messages[0])
        self.assertEqual(restored_record["template_id"], first["template_id"])
        self.assertEqual(restored_record["message_template"], first["message_template"])


if __name__ == "__main__":
    unittest.main()
