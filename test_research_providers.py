import os
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from research_providers import (
    AVAILABLE,
    CONTRACT_INVALID,
    CONTRACT_VALID,
    CONFIGURED,
    PARTIAL,
    STALE,
    UNCONFIGURED,
    PROVIDER_SPECS,
    fetch_provider,
    normalize_result,
    provider_catalog,
    provider_payload_contract,
    research_readiness,
    run_contract_checks,
)


class ResearchProviderTests(unittest.TestCase):
    def test_contract_fixtures_cover_all_approved_adapters(self):
        checks = run_contract_checks()
        self.assertEqual(len(checks), len(PROVIDER_SPECS))
        self.assertTrue(all(check["ok"] for check in checks))

    def test_contract_check_reports_only_safe_provider_metadata(self):
        checks = run_contract_checks()
        self.assertTrue(all(set(check) == {"provider", "ok", "reason"} for check in checks))
        self.assertNotIn("fixture", repr(checks))

    def test_success_and_malformed_payloads_are_distinguished(self):
        spec = next(item for item in PROVIDER_SPECS if item.api_name == "fred")
        valid, _ = provider_payload_contract(
            spec, {"observations": [{"date": "2026-08-24", "value": "1.2"}]}
        )
        invalid, _ = provider_payload_contract(spec, {"unexpected": "shape"})
        self.assertTrue(valid)
        self.assertFalse(invalid)
        result = normalize_result(
            spec,
            {"unexpected": "shape"},
            fetched_at="2026-08-24T12:00:00+00:00",
            now=datetime(2026, 8, 24, 12, tzinfo=timezone.utc).timestamp(),
        )
        self.assertEqual(result["contract_status"], CONTRACT_INVALID)
        self.assertEqual(result["status"], PARTIAL)
        self.assertFalse(research_readiness([result])["ready"])

    def test_stale_and_throttled_fixtures_remain_unavailable(self):
        spec = next(item for item in PROVIDER_SPECS if item.api_name == "fred")
        stale = normalize_result(
            spec,
            {"observations": [{"value": "1.2"}]},
            fetched_at="2026-08-22T00:00:00+00:00",
            now=datetime(2026, 8, 24, 12, tzinfo=timezone.utc).timestamp(),
        )
        throttled = normalize_result(
            spec,
            {"Note": "rate limit fixture"},
            fetched_at="2026-08-24T12:00:00+00:00",
            now=datetime(2026, 8, 24, 12, tzinfo=timezone.utc).timestamp(),
        )
        self.assertEqual(stale["status"], STALE)
        self.assertEqual(stale["contract_status"], CONTRACT_VALID)
        self.assertEqual(throttled["status"], PARTIAL)
        self.assertEqual(throttled["contract_status"], CONTRACT_INVALID)
        self.assertFalse(research_readiness([stale, throttled])["ready"])
    def test_catalog_never_calls_configured_provider_connected(self):
        with patch.dict(os.environ, {}, clear=True):
            catalog = provider_catalog()
        self.assertTrue(catalog)
        self.assertTrue(all(item["status"] == UNCONFIGURED for item in catalog))
        self.assertTrue(all(item["fetched_at"] is None for item in catalog))

    def test_credentials_only_move_provider_to_configured(self):
        with patch.dict(os.environ, {"FRED_API_KEY": "secret-marker"}, clear=True):
            fred = next(item for item in provider_catalog() if item["api"] == "fred")
        self.assertEqual(fred["status"], CONFIGURED)
        self.assertNotIn("secret-marker", repr(fred))

    def test_normalized_response_has_provenance_and_freshness(self):
        spec = next(item for item in PROVIDER_SPECS if item.api_name == "fred")
        result = normalize_result(
            spec,
            {"observations": [{"value": "1.2"}]},
            fetched_at="2026-08-24T12:00:00+00:00",
            now=datetime(2026, 8, 24, 12, tzinfo=timezone.utc).timestamp(),
        )
        self.assertEqual(result["status"], AVAILABLE)
        self.assertEqual(result["quality"], "provider response received")
        self.assertEqual(result["freshness_seconds"], 0.0)
        self.assertEqual(result["source"], spec.endpoint)
        self.assertIn("uncertainty", result)

    def test_stale_and_empty_responses_are_not_usable(self):
        spec = PROVIDER_SPECS[0]
        stale = normalize_result(
            spec,
            {"name": "fixture", "filings": {"recent": {}}},
            fetched_at="2026-08-22T00:00:00+00:00",
            now=datetime(2026, 8, 24, 12, tzinfo=timezone.utc).timestamp(),
        )
        empty = normalize_result(
            spec,
            {},
            fetched_at="2026-08-24T12:00:00+00:00",
            now=datetime(2026, 8, 24, 12, tzinfo=timezone.utc).timestamp(),
        )
        self.assertEqual(stale["status"], STALE)
        self.assertEqual(empty["status"], PARTIAL)
        self.assertFalse(research_readiness([stale, empty])["ready"])

    def test_provider_error_payload_is_partial(self):
        spec = PROVIDER_SPECS[0]
        result = normalize_result(
            spec,
            {"Error Message": "rate limit"},
            fetched_at="2026-08-24T12:00:00+00:00",
            now=datetime(2026, 8, 24, 12, tzinfo=timezone.utc).timestamp(),
        )
        self.assertEqual(result["status"], PARTIAL)
        self.assertTrue(result["partial"])

    def test_unconfigured_fetch_is_explicit_and_does_not_network(self):
        with patch.dict(os.environ, {}, clear=True), patch(
            "research_providers._request_json"
        ) as request:
            result = fetch_provider("fred")
        request.assert_not_called()
        self.assertEqual(result["status"], UNCONFIGURED)
        self.assertIsNone(result["data"])

    def test_unknown_provider_is_rejected(self):
        with self.assertRaises(ValueError):
            fetch_provider("not-approved")


if __name__ == "__main__":
    unittest.main()