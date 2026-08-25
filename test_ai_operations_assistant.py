import unittest
import json
from urllib.error import HTTPError, URLError
from unittest.mock import Mock, patch

from ai_operations_assistant import (
    ASSISTANT_NAME,
    OpenAICompatibleProvider,
    ProviderError,
    ProviderHealth,
    REFUSAL_MESSAGE,
    UNKNOWN,
    UNAVAILABLE_MESSAGE,
    answer_question,
    build_assistant_context,
    format_failure_category,
    format_failure_category_counts,
    get_provider_health,
    get_research_context,
)


class AssistantTests(unittest.TestCase):
    def setUp(self):
        self.results = {
            "starting_capital": 25.0,
            "ending_capital": 26.0,
            "profit": 1.0,
            "trades": 2,
            "win_rate": 50.0,
            "max_drawdown": 3.0,
            "evaluations": 10,
            "highest_score": 90,
            "score_80_or_more": 3,
        }
        self.market = Mock(pair_name="XBT/CAD", last_error=None)
        self.candles = [{"timestamp": 1, "close": 100}, {"timestamp": 2, "close": 101}]
        self.historical = {
            "periods": [{"market_return": 20, "return_percent": 5}],
            "aggregate": {"average_return": 5, "worst_drawdown": 4},
            "regime_summary": {"Bull": [{"market_return": 20, "return_percent": 5}],
                               "Bear": [], "Sideways": []},
        }

    def test_context_contains_approved_values_and_is_serializable(self):
        context = build_assistant_context(
            self.results, {"strategy_score": 88, "decision": "BUY"},
            self.market, self.candles, self.historical,
        )
        self.assertEqual(context["performance"]["paper_profit"], 1.0)
        self.assertEqual(context["research"]["regimes"]["Bull"]["completed_periods"], 1)
        self.assertIsInstance(context["market"]["latest_close"], int)

    def test_context_can_carry_read_only_strategy_council(self):
        council = {
            "final_action": "WAIT",
            "council": {"data_quality": "DATA_INSUFFICIENT"},
            "governor": {"approved": False},
        }
        context = build_assistant_context(
            self.results, {"strategy_score": 88, "decision": "BUY"},
            self.market, self.candles, self.historical,
            council_context=council,
        )
        self.assertEqual(context["strategy_council"]["final_action"], "WAIT")
        self.assertFalse(context["strategy_council"]["governor"]["approved"])

    def test_missing_data_is_unknown(self):
        context = build_assistant_context(None, None, Mock(), [], None)
        self.assertEqual(context["strategy"]["latest_score"], UNKNOWN)
        self.assertEqual(context["research"]["status"], UNKNOWN)
        self.assertEqual(context["market"]["latest_close"], UNKNOWN)

    def test_research_classification_preserves_missing_regimes(self):
        research = get_research_context(self.historical)
        self.assertEqual(research["regimes"]["Bull"]["completed_periods"], 1)
        self.assertEqual(research["regimes"]["Bear"]["average_market_return"], UNKNOWN)

    def test_provider_failure_is_unavailable(self):
        provider = Mock()
        provider.answer.side_effect = RuntimeError("offline")
        self.assertEqual(
            answer_question("What is status?", {}, provider=provider),
            UNAVAILABLE_MESSAGE,
        )

    def test_kova_name_is_supported_for_direct_status_questions(self):
        from ai_operations_assistant import ReadOnlySummaryProvider

        context = {
            "status": {
                "paper_trading": "ENABLED",
                "live_trading": "DISABLED",
                "live_market_data": "AVAILABLE",
            }
        }
        self.assertEqual(ASSISTANT_NAME, "Kova")
        response = answer_question(
            "Kova, what is the current status?",
            context,
            provider=ReadOnlySummaryProvider(),
        )
        self.assertTrue(response.startswith("FACT\n"))
        self.assertIn("Paper trading: ENABLED", response)

    def test_read_only_fallback_answers_decision_why_and_change_questions(self):
        from ai_operations_assistant import ReadOnlySummaryProvider

        context = {
            "market": {
                "source": "XBT/CAD · Kraken",
                "latest_close": 50000,
                "latest_timestamp": 1700007200,
            },
            "strategy": {
                "latest_decision": "HOLD",
                "latest_score": 80,
            },
            "strategy_council": {
                "governor": {"approved": False},
                "final_action": "WAIT",
            },
        }
        provider = ReadOnlySummaryProvider()
        decision = provider.answer("What is your current decision?", context, [])
        why = provider.answer("Why?", context, [])
        change = provider.answer("What would make you change your decision?", context, [])
        self.assertTrue(decision.startswith("FACT\n"))
        self.assertIn("Current paper decision: WAIT", decision)
        self.assertIn("XBT/CAD · Kraken", why)
        self.assertTrue(why.startswith("ANALYSIS\n"))
        self.assertIn("fresh, healthy market", change)

    def test_provider_failure_keeps_read_only_fallback_response_unchanged(self):
        provider = Mock()
        provider.answer.side_effect = ProviderError(
            "raw provider outage with secret context",
            "network_error",
        )
        self.assertEqual(
            answer_question(
                "What is status? api-key=do-not-leak",
                {"credential": "do-not-leak"},
                provider=provider,
            ),
            UNAVAILABLE_MESSAGE,
        )

    def test_provider_receives_conversation_context(self):
        provider = Mock()
        provider.answer.return_value = "FACT\n\nPaper mode is enabled."
        history = [{"role": "user", "content": "hello"}]
        answer_question("What mode?", {}, history, provider)
        provider.answer.assert_called_once_with("What mode?", {}, history)

    def test_mutation_request_is_refused_without_provider_call(self):
        provider = Mock()
        self.assertEqual(answer_question("Place a trade now", {}, provider=provider), REFUSAL_MESSAGE)
        provider.answer.assert_not_called()

    def test_no_execution_capability_is_exposed(self):
        from ai_operations_assistant import build_assistant_context
        self.assertFalse(any(name in dir(build_assistant_context) for name in ("execute", "trade", "write")))

    def test_provider_health_tracks_latency_availability_and_failure_category(self):
        health = ProviderHealth("Test provider")
        health.record_success(12.345)
        health.record_failure("timeout", 15000.789)
        snapshot = health.snapshot()
        self.assertEqual(snapshot["availability"], "DEGRADED")
        self.assertEqual(snapshot["requests"], 2)
        self.assertEqual(snapshot["success_rate_percent"], 50.0)
        self.assertEqual(snapshot["last_latency_ms"], 15000.8)
        self.assertEqual(snapshot["last_failure_category"], "timeout")
        self.assertEqual(snapshot["failure_categories"], {"timeout": 1})
        self.assertNotIn("question", snapshot)
        self.assertNotIn("context", snapshot)

    def test_failure_category_display_is_concise_and_safe(self):
        self.assertEqual(format_failure_category("rate_limit"), "Rate limit")
        self.assertEqual(
            format_failure_category("provider_outage"),
            "Provider outage",
        )
        self.assertEqual(
            format_failure_category_counts({
                "rate_limit": 2,
                "provider_outage": 1,
            }),
            "Rate limit: 2 · Provider outage: 1",
        )
        self.assertEqual(
            format_failure_category("raw provider details"),
            "Provider error",
        )
        self.assertEqual(format_failure_category_counts({}), UNKNOWN)

    def test_openai_timeout_is_categorized_without_prompt_telemetry(self):
        health = ProviderHealth("Test managed provider")
        provider = OpenAICompatibleProvider(
            "https://provider.test/chat/completions",
            "not-a-real-key",
            health=health,
        )
        with patch("ai_operations_assistant.urlopen", side_effect=TimeoutError("slow")):
            with self.assertRaises(ProviderError) as raised:
                provider.answer("What is status?", {"secret": "context"}, [])
        self.assertEqual(raised.exception.category, "timeout")
        snapshot = health.snapshot()
        self.assertEqual(snapshot["last_failure_category"], "timeout")
        self.assertNotIn("status", snapshot)
        self.assertNotIn("secret", snapshot)

    def test_openai_transport_failure_is_categorized_without_raw_error_telemetry(self):
        health = ProviderHealth("Test managed provider")
        provider = OpenAICompatibleProvider(
            "https://provider.test/chat/completions",
            "not-a-real-key",
            health=health,
        )
        raw_error = "connection reset with bearer secret"
        with patch(
            "ai_operations_assistant.urlopen",
            side_effect=URLError(raw_error),
        ):
            with self.assertRaises(ProviderError) as raised:
                provider.answer(
                    "What is status? prompt-marker",
                    {"credential": "credential-marker"},
                    [],
                )
        self.assertEqual(raised.exception.category, "network_error")
        snapshot = health.snapshot()
        self.assertEqual(snapshot["last_failure_category"], "network_error")
        self.assertEqual(snapshot["failure_categories"], {"network_error": 1})
        for value in ("prompt-marker", "credential-marker", raw_error):
            self.assertNotIn(value, json.dumps(snapshot))

    def test_openai_rate_limit_is_categorized_without_sensitive_telemetry(self):
        health = ProviderHealth("Test managed provider")
        provider = OpenAICompatibleProvider(
            "https://provider.test/chat/completions",
            "credential-marker",
            health=health,
        )
        raw_error = "rate limit details with provider secret"
        http_error = HTTPError(
            provider.endpoint,
            429,
            raw_error,
            {"Retry-After": "60"},
            None,
        )
        with patch("ai_operations_assistant.urlopen", side_effect=http_error):
            with self.assertRaises(ProviderError) as raised:
                provider.answer(
                    "What is status? prompt-marker",
                    {"credential": "credential-marker", "context": "context-marker"},
                    [],
                )
        self.assertEqual(raised.exception.category, "rate_limit")
        snapshot = health.snapshot()
        self.assertEqual(snapshot["last_failure_category"], "rate_limit")
        self.assertEqual(snapshot["failure_categories"], {"rate_limit": 1})
        for value in ("prompt-marker", "context-marker", "credential-marker", raw_error):
            self.assertNotIn(value, json.dumps(snapshot))

    def test_openai_server_error_is_categorized_without_sensitive_telemetry(self):
        health = ProviderHealth("Test managed provider")
        provider = OpenAICompatibleProvider(
            "https://provider.test/chat/completions",
            "credential-marker",
            health=health,
        )
        raw_error = "provider outage details with secret"
        http_error = HTTPError(
            provider.endpoint,
            503,
            raw_error,
            {},
            None,
        )
        with patch("ai_operations_assistant.urlopen", side_effect=http_error):
            with self.assertRaises(ProviderError) as raised:
                provider.answer(
                    "What is status? prompt-marker",
                    {"credential": "credential-marker", "context": "context-marker"},
                    [],
                )
        self.assertEqual(raised.exception.category, "provider_outage")
        snapshot = health.snapshot()
        self.assertEqual(snapshot["last_failure_category"], "provider_outage")
        self.assertEqual(snapshot["failure_categories"], {"provider_outage": 1})
        for value in ("prompt-marker", "context-marker", "credential-marker", raw_error):
            self.assertNotIn(value, json.dumps(snapshot))

    def test_openai_malformed_response_is_categorized_without_raw_error_telemetry(self):
        health = ProviderHealth("Test managed provider")
        provider = OpenAICompatibleProvider(
            "https://provider.test/chat/completions",
            "not-a-real-key",
            health=health,
        )
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        raw_error = "malformed provider body with secret context"
        response.read.return_value = raw_error.encode()
        with patch("ai_operations_assistant.urlopen", return_value=response):
            with self.assertRaises(ProviderError) as raised:
                provider.answer(
                    "What is status? prompt-marker",
                    {"credential": "credential-marker"},
                    [],
                )
        self.assertEqual(raised.exception.category, "response_validation")
        snapshot = health.snapshot()
        self.assertEqual(snapshot["last_failure_category"], "response_validation")
        self.assertEqual(snapshot["failure_categories"], {"response_validation": 1})
        for value in ("prompt-marker", "credential-marker", raw_error):
            self.assertNotIn(value, json.dumps(snapshot))

    def test_openai_valid_response_is_successful_without_request_telemetry(self):
        health = ProviderHealth("Test managed provider")
        provider = OpenAICompatibleProvider(
            "https://provider.test/chat/completions",
            "credential-marker",
            health=health,
        )
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.return_value = (
            b'{"choices": [{"message": {"content": "FACT\\n\\nPaper mode is enabled."}}]}'
        )
        with patch("ai_operations_assistant.urlopen", return_value=response):
            answer = provider.answer(
                "What is status? prompt-marker",
                {"credential": "credential-marker"},
                [],
            )
        self.assertEqual(answer, "FACT\n\nPaper mode is enabled.")
        snapshot = health.snapshot()
        self.assertEqual(snapshot["availability"], "HEALTHY")
        self.assertEqual(snapshot["last_outcome"], "SUCCESS")
        self.assertEqual(snapshot["failure_categories"], {})
        for value in ("prompt-marker", "credential-marker"):
            self.assertNotIn(value, json.dumps(snapshot))

    def test_provider_health_snapshot_is_aggregate_only(self):
        snapshot = get_provider_health()
        self.assertEqual(
            set(snapshot),
            {
                "provider",
                "availability",
                "requests",
                "successes",
                "failures",
                "success_rate_percent",
                "last_latency_ms",
                "last_outcome",
                "last_failure_category",
                "failure_categories",
            },
        )


if __name__ == "__main__":
    unittest.main()
