from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from soc_threat.structured_semantics import (  # noqa: E402
    StructuredDrainModel,
    parse_structured_log,
)


def raw(
    message: str,
    *,
    product: str = "",
    vendor: str = "",
    pipeline: str = "syslog",
) -> dict[str, str]:
    return {
        "event_id": "evt-1",
        "pipeline": pipeline,
        "vendor_name": vendor,
        "product_name": product,
        "message_sanitized": message,
    }


class StructuredSemanticTests(unittest.TestCase):
    def test_embedded_json_preserves_malware_action_and_severity(self) -> None:
        row = raw(
            "ORG-1780 ::: action=BLOCKED ::: payload={"
            '"categories":["Malware","Web Threat"],'
            '"actionTaken":"BLOCKED","severityCode":2,'
            '"streamName":"Endpoint Detection"}'
        )
        parsed = parse_structured_log(row)
        self.assertEqual(parsed.base.message_format, "json")
        self.assertEqual(parsed.structured_parser, "json_recursive")
        self.assertEqual(parsed.payload_parse_status, "success")
        self.assertEqual(parsed.event_action, "block")
        self.assertEqual(parsed.event_category, "malware")
        self.assertEqual(parsed.threat_category, "malware")
        self.assertEqual(parsed.event_severity, 2)
        self.assertEqual(parsed.malware_present, 1)
        self.assertNotEqual(parsed.schema_id, "__MISSING__")

    def test_duo_json_extracts_authentication_semantics(self) -> None:
        row = raw(
            "ORG-1780 ::: payload={"
            '"HOST-56507_type":"authentication",'
            '"fHOST-15196":"duo_push",'
            '"reason":"invalid_passcode",'
            '"result":"auth_failure",'
            '"application":{"name":"LastPass"}}'
            " ::: streamName=DUO"
        )
        parsed = parse_structured_log(row)
        self.assertEqual(parsed.payload_parse_status, "success")
        self.assertEqual(parsed.event_category, "authentication")
        self.assertEqual(parsed.authentication_factor, "duo_push")
        self.assertEqual(parsed.event_reason, "invalid_passcode")
        self.assertEqual(parsed.event_outcome, "failure")
        self.assertEqual(parsed.application_name, "lastpass")
        self.assertEqual(parsed.service_name, "duo")
        self.assertEqual(parsed.authentication_present, 1)

    def test_cef_parser_extracts_header_extension_and_residual_message(self) -> None:
        row = raw(
            "CEF:0|Vendor|WAF|1.0|342|GEO_IP_BLOCK|1|"
            "cat=WF dst=10.252.74.96 dpt=80 act=DENY "
            "msg=[GeoIP Match] src=100.64.45.97 spt=51234 requestMethod=GET"
        )
        parsed = parse_structured_log(row)
        self.assertEqual(parsed.base.message_format, "cef")
        self.assertEqual(parsed.payload_parse_status, "success")
        self.assertEqual(parsed.event_action, "deny")
        self.assertEqual(parsed.base.event_code, "342")
        self.assertEqual(parsed.destination_port, 80)
        self.assertEqual(parsed.source_port, 51234)
        self.assertEqual(parsed.http_method, "GET")
        self.assertEqual(parsed.source_ip_present, 1)
        self.assertEqual(parsed.destination_ip_present, 1)
        self.assertEqual(parsed.rule_name, "geo_ip_block")
        self.assertEqual(parsed.application_name, "__MISSING__")
        self.assertIn("GEO_IP_BLOCK", parsed.drain_message)
        self.assertIn("GeoIP Match", parsed.drain_message)

    def test_windows_event_keeps_structured_event_semantics(self) -> None:
        row = raw(
            'ORG-1657 ::: {"beat":"winlogbeat","code":"4672",'
            '"message":"Special privileges assigned to new logon"}'
        )
        parsed = parse_structured_log(row)
        self.assertEqual(parsed.base.message_format, "windows_json")
        self.assertEqual(parsed.base.event_code, "4672")
        self.assertEqual(parsed.base.event_name, "special_privileges")
        self.assertEqual(parsed.structured_parser, "json_recursive")
        self.assertEqual(parsed.payload_parse_status, "success")

    def test_malformed_sanitized_xml_uses_tolerant_leaf_fallback(self) -> None:
        row = raw(
            "<USER-0010-56507><USER-0086><USER-0010-56507ID>4672"
            "</USER-0010-56507ID><Broken AttributeWithoutEquals>ignored"
            "</Broken><Task>Special Privileges</Task></USER-0086>"
            "</USER-0010-56507>",
            product="Windows Logs",
            vendor="Microsoft",
        )
        parsed = parse_structured_log(row)
        self.assertEqual(parsed.base.message_format, "windows_xml")
        self.assertEqual(parsed.payload_parse_status, "success")
        self.assertGreater(parsed.structured_field_count, 0)
        self.assertEqual(parsed.event_code, "4672")

    def test_sanitizer_tokens_are_not_treated_as_semantic_values(self) -> None:
        row = raw(
            'ORG-1657 ::: {"kind":"HOST-56507","outcome":"ORG-0893",'
            '"name":"USER-0010-0346","level":"ORG-0706rmation"}'
        )
        parsed = parse_structured_log(row)
        self.assertEqual(parsed.event_type, "__MISSING__")
        self.assertEqual(parsed.event_outcome, "__MISSING__")
        self.assertEqual(parsed.application_name, "__MISSING__")
        self.assertEqual(parsed.rule_name, "__MISSING__")
        self.assertEqual(parsed.event_severity, -1)

    def test_vpc_flow_parser_reads_positional_fields(self) -> None:
        row = raw(
            "2 123456789012 eni-123 10.0.0.1 10.0.0.2 51515 443 6 "
            "10 840 1676902000 1676902060 REJECT OK",
            pipeline="vpc_flow",
        )
        parsed = parse_structured_log(row)
        self.assertEqual(parsed.base.message_format, "vpc_flow")
        self.assertEqual(parsed.payload_parse_status, "success")
        self.assertEqual(parsed.source_port, 51515)
        self.assertEqual(parsed.destination_port, 443)
        self.assertEqual(parsed.network_protocol, "tcp")
        self.assertEqual(parsed.event_action, "reject")

    def test_v1_2_model_persists_schema_and_semantic_frequencies(self) -> None:
        rows = [
            raw(
                'prefix ::: payload={"event_type":"authentication",'
                '"factor":"duo_push","result":"failure"}'
            ),
            raw(
                'prefix ::: payload={"event_type":"authentication",'
                '"factor":"duo_push","result":"success"}'
            ),
        ]
        model = StructuredDrainModel()
        for row in rows:
            model.fit_raw(row)
        before = model.feature_record(rows[0])
        self.assertEqual(before["schema_seen_train"], 1)
        self.assertEqual(before["semantic_template_seen_train"], 1)
        self.assertEqual(before["payload_parse_success"], 1)

        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory) / "model"
            model.save(model_dir)
            restored = StructuredDrainModel.load(model_dir)
            after = restored.feature_record(rows[0])
        self.assertEqual(after["schema_id"], before["schema_id"])
        self.assertEqual(after["semantic_template_id"], before["semantic_template_id"])
        self.assertEqual(
            after["schema_frequency_log1p"], before["schema_frequency_log1p"]
        )


if __name__ == "__main__":
    unittest.main()
