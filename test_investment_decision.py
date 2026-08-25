import unittest
from datetime import datetime, timezone

from investment_decision import (
    AssetType,
    DataQuality,
    Decision,
    Direction,
    InvestmentDecisionRecord,
    OptionStrategy,
    adapt_btc_cad_strategy_evaluation,
    analyze_defined_risk_option_strategy,
    evaluate_investment_candidate,
    evaluate_defined_risk_option_candidate,
    fetch_public_option_quote_candidates,
    normalize_option_contract,
    review_defined_risk_option_candidates,
)


OBSERVED_AT = "2026-08-22T12:00:00+00:00"
OPTION_NOW = datetime(2026, 8, 22, 12, 0, tzinfo=timezone.utc)


def option_quote(option_type="CALL", strike=100.0, bid=4.0, ask=6.0, **overrides):
    quote = {
        "underlying": "ABC",
        "option_type": option_type,
        "strike": strike,
        "expiration": "2026-09-21T20:00:00+00:00",
        "bid": bid,
        "ask": ask,
        "underlying_price": 100.0,
        "observed_at": OBSERVED_AT,
    }
    quote.update(overrides)
    return normalize_option_contract(quote, now=OPTION_NOW)


def complete_candidate(**overrides):
    values = {
        "instrument": "ABC",
        "underlying": "ABC",
        "asset_type": AssetType.STOCK,
        "strategy": "VALUE",
        "direction": Direction.LONG,
        "thesis": "Reliable positive expected value.",
        "proposed_decision": Decision.HOLD,
        "expected_return": 0.08,
        "expected_risk": 0.03,
        "maximum_loss": 100.0,
        "liquidity": 80.0,
        "estimated_transaction_cost": 1.0,
        "estimated_slippage": 0.5,
        "confidence": 75.0,
        "time_horizon": "12 months",
        "portfolio_impact": 2.0,
        "concentration_impact": 1.0,
        "correlation_impact": 0.2,
        "market_regime_compatibility": "NORMAL",
        "risk_score": 25.0,
        "observed_at": OBSERVED_AT,
    }
    values.update(overrides)
    return evaluate_investment_candidate(**values)


class InvestmentDecisionTests(unittest.TestCase):
    def test_public_option_provider_normalizes_read_only_chain_legs(self):
        payload = {
            "optionChain": {
                "result": [{
                    "expirationDates": [int(OPTION_NOW.timestamp()) + 86400 * 30],
                    "quote": {"regularMarketPrice": 100.0},
                    "options": [{
                        "calls": [{
                            "contractSymbol": "ABC260921C00100000",
                            "strike": 100,
                            "bid": 4,
                            "ask": 6,
                            "lastTradeDate": int(OPTION_NOW.timestamp()),
                        }],
                        "puts": [],
                    }],
                }]
            }
        }

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                import json
                return json.dumps(payload).encode()

        snapshot = fetch_public_option_quote_candidates(
            "abc", opener=lambda request, timeout: Response()
        )
        self.assertTrue(snapshot["available"])
        self.assertEqual(snapshot["symbol"], "ABC")
        self.assertEqual(snapshot["candidates"][0]["strategy"], "LONG_CALL")
        self.assertEqual(
            snapshot["candidates"][0]["contracts"][0]["observed_at"],
            OBSERVED_AT,
        )

    def test_option_metadata_is_preserved_without_fabrication(self):
        quote = option_quote(
            implied_volatility=0.42,
            delta=0.55,
            gamma=0.03,
            theta=-0.08,
            vega=0.12,
            volume=1200,
            open_interest=4500,
        )
        analysis = analyze_defined_risk_option_strategy(
            OptionStrategy.LONG_CALL, contracts=[quote], now=OPTION_NOW
        ).to_dict()
        self.assertEqual(analysis["quote_metadata"]["volume"], 1200)
        self.assertEqual(analysis["quote_metadata"]["open_interest"], 4500)
        self.assertEqual(analysis["quote_metadata"]["implied_volatility"], 0.42)
        self.assertEqual(analysis["expiration_warning"], "NORMAL")

    def test_invalid_optional_option_metadata_is_rejected(self):
        with self.assertRaises(ValueError):
            option_quote(volume=-1)

    def test_public_option_provider_does_not_invent_missing_observation_time(self):
        payload = {
            "optionChain": {
                "result": [{
                    "expirationDates": [int(OPTION_NOW.timestamp()) + 86400 * 30],
                    "quote": {"regularMarketPrice": 100.0},
                    "options": [{
                        "calls": [{"strike": 100, "bid": 4, "ask": 6}],
                        "puts": [],
                    }],
                }]
            }
        }

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                import json
                return json.dumps(payload).encode()

        snapshot = fetch_public_option_quote_candidates(
            "ABC", opener=lambda request, timeout: Response()
        )
        self.assertTrue(snapshot["available"])
        reviewed = review_defined_risk_option_candidates(
            snapshot["candidates"], now=OPTION_NOW
        )
        self.assertEqual(reviewed[0]["status"], "REJECTED")
        self.assertIn("observed_at", reviewed[0]["rejection_reason"])

    def test_public_option_provider_reports_outage_without_fallback_quotes(self):
        snapshot = fetch_public_option_quote_candidates(
            "ABC", opener=lambda request, timeout: (_ for _ in ()).throw(
                TimeoutError("timed out")
            )
        )
        self.assertFalse(snapshot["available"])
        self.assertEqual(snapshot["candidates"], [])
        self.assertIn("unavailable", snapshot["error"])

    def test_option_contract_normalization_rejects_bad_or_stale_quotes(self):
        for overrides in (
            {"ask": 3.0},
            {"bid": None},
            {"expiration": "2026-08-21T20:00:00+00:00"},
            {"observed_at": "2026-08-22T11:00:00+00:00"},
        ):
            with self.assertRaises(ValueError):
                normalize_option_contract(
                    {**{
                        "underlying": "ABC", "option_type": "CALL", "strike": 100,
                        "expiration": "2026-09-21T20:00:00+00:00", "bid": 4, "ask": 6,
                        "underlying_price": 100, "observed_at": OBSERVED_AT,
                    }, **overrides},
                    now=OPTION_NOW,
                    max_data_age_seconds=60,
                )

    def test_defined_risk_strategy_economics(self):
        call = option_quote()
        put = option_quote("PUT", bid=5.0, ask=7.0)
        lower_call = option_quote(strike=95, bid=8, ask=10)
        higher_call = option_quote(strike=105, bid=3, ask=5)
        lower_put = option_quote("PUT", strike=95, bid=2, ask=4)
        higher_put = option_quote("PUT", strike=105, bid=8, ask=10)

        long_call = analyze_defined_risk_option_strategy(
            OptionStrategy.LONG_CALL, contracts=[call], now=OPTION_NOW
        )
        self.assertEqual(long_call.break_even, 105.0)
        self.assertEqual(long_call.maximum_loss, 500.0)
        self.assertEqual(long_call.cost, 500.0)
        self.assertEqual(long_call.slippage, 100.0)
        self.assertEqual(long_call.days_to_expiration, 30)

        long_put = analyze_defined_risk_option_strategy(
            OptionStrategy.LONG_PUT, contracts=[put], now=OPTION_NOW
        )
        self.assertEqual(long_put.break_even, 94.0)
        self.assertEqual(long_put.maximum_profit, 9400.0)

        covered = analyze_defined_risk_option_strategy(
            OptionStrategy.COVERED_CALL, contracts=[call], now=OPTION_NOW
        )
        self.assertEqual(covered.break_even, 95.0)
        self.assertEqual(covered.maximum_loss, 9500.0)

        secured = analyze_defined_risk_option_strategy(
            OptionStrategy.CASH_SECURED_PUT, contracts=[put], now=OPTION_NOW
        )
        self.assertEqual(secured.break_even, 94.0)
        self.assertEqual(secured.maximum_profit, 600.0)

        bull = analyze_defined_risk_option_strategy(
            OptionStrategy.BULL_CALL_SPREAD,
            contracts=[lower_call, higher_call],
            now=OPTION_NOW,
        )
        self.assertEqual(bull.break_even, 100.0)
        self.assertEqual(bull.maximum_loss, 500.0)
        self.assertEqual(bull.maximum_profit, 500.0)

        bear = analyze_defined_risk_option_strategy(
            OptionStrategy.BEAR_PUT_SPREAD,
            contracts=[lower_put, higher_put],
            now=OPTION_NOW,
        )
        self.assertEqual(bear.break_even, 99.0)
        self.assertEqual(bear.maximum_loss, 600.0)
        self.assertEqual(bear.maximum_profit, 400.0)

        protective = analyze_defined_risk_option_strategy(
            OptionStrategy.PROTECTIVE_PUT, contracts=[put], now=OPTION_NOW
        )
        self.assertEqual(protective.break_even, 106.0)
        self.assertEqual(protective.maximum_loss, 600.0)

        collar = analyze_defined_risk_option_strategy(
            OptionStrategy.COLLAR, contracts=[put, call], now=OPTION_NOW
        )
        self.assertEqual(collar.break_even, 101.0)
        self.assertEqual(collar.maximum_loss, 100.0)

    def test_option_quantity_and_risk_governor(self):
        analysis = analyze_defined_risk_option_strategy(
            OptionStrategy.LONG_CALL, contracts=[option_quote()], quantity=2, now=OPTION_NOW
        )
        self.assertEqual(analysis.maximum_loss, 1000.0)
        fields = {
            "expected_return": 0.1, "expected_risk": 0.2, "liquidity": 80,
            "confidence": 80, "time_horizon": "30 days", "portfolio_impact": 2,
            "concentration_impact": 1, "correlation_impact": 1,
            "market_regime_compatibility": "NORMAL", "risk_score": 20,
            "observed_at": OBSERVED_AT,
        }
        record = evaluate_defined_risk_option_candidate(
            analysis, instrument="ABC 100C", thesis="Defined loss.",
            proposed_decision=Decision.BUY, risk_approved=True,
            risk_veto=lambda: (False, "options exposure blocked"), **fields
        )
        self.assertEqual(record.decision, Decision.REJECT)
        self.assertEqual(record.rejection_reason, "options exposure blocked")

    def test_unsupported_undefined_risk_strategy_is_rejected(self):
        with self.assertRaises(ValueError):
            analyze_defined_risk_option_strategy(
                "SHORT_CALL", contracts=[option_quote()], now=OPTION_NOW
            )

    def test_option_candidate_review_keeps_accepted_and_rejected_candidates(self):
        reviewed = review_defined_risk_option_candidates(
            [
                {
                    "instrument": "ABC 100C",
                    "strategy": "LONG_CALL",
                    "contracts": [{
                        "underlying": "ABC", "option_type": "CALL", "strike": 100,
                        "expiration": "2026-09-21T20:00:00+00:00", "bid": 4, "ask": 6,
                        "underlying_price": 100, "observed_at": OBSERVED_AT,
                    }],
                },
                {
                    "instrument": "ABC short call",
                    "strategy": "SHORT_CALL",
                    "contracts": [{
                        "underlying": "ABC", "option_type": "CALL", "strike": 100,
                        "expiration": "2026-09-21T20:00:00+00:00", "bid": 4, "ask": 6,
                        "underlying_price": 100, "observed_at": OBSERVED_AT,
                    }],
                },
                {
                    "instrument": "ABC stale put",
                    "strategy": "LONG_PUT",
                    "contracts": [{
                        "underlying": "ABC", "option_type": "PUT", "strike": 100,
                        "expiration": "2026-09-21T20:00:00+00:00", "bid": 4, "ask": 6,
                        "underlying_price": 100,
                        "observed_at": "2026-08-22T11:00:00+00:00",
                    }],
                },
            ],
            now=OPTION_NOW,
            max_data_age_seconds=60,
        )
        self.assertEqual([item["status"] for item in reviewed],
                         ["ACCEPTED", "REJECTED", "REJECTED"])
        self.assertEqual(reviewed[0]["analysis"]["maximum_loss"], 500.0)
        self.assertIn("strategy is invalid", reviewed[1]["rejection_reason"])
        self.assertIn("stale", reviewed[2]["rejection_reason"])

    def test_record_supports_every_required_asset_type(self):
        for asset_type in (
            AssetType.STOCK,
            AssetType.ETF,
            AssetType.OPTION,
            AssetType.DEFINED_RISK_OPTION_STRATEGY,
            AssetType.CASH,
        ):
            maximum_loss = 0.0 if asset_type == AssetType.CASH else 100.0
            record = complete_candidate(
                asset_type=asset_type,
                maximum_loss=maximum_loss,
            )
            self.assertEqual(record.asset_type, asset_type)
            self.assertIn("maximum_loss", record.to_dict())

    def test_record_supports_every_valid_decision(self):
        for decision in Decision:
            record = complete_candidate(
                proposed_decision=decision,
                risk_approved=True if decision in {Decision.BUY, Decision.SELL, Decision.REBALANCE} else None,
            )
            if decision == Decision.REJECT:
                self.assertEqual(record.data_quality, DataQuality.SUFFICIENT)
                self.assertTrue(record.rejection_reason)
            else:
                self.assertEqual(record.decision, decision)

    def test_actionable_decision_requires_explicit_risk_approval(self):
        record = complete_candidate(proposed_decision=Decision.BUY)
        self.assertEqual(record.decision, Decision.REJECT)
        self.assertIn("Risk approval", record.rejection_reason)

    def test_risk_veto_has_final_authority(self):
        record = complete_candidate(
            proposed_decision=Decision.BUY,
            risk_approved=True,
            risk_veto=lambda: (False, "concentration limit exceeded"),
        )
        self.assertEqual(record.decision, Decision.REJECT)
        self.assertEqual(record.rejection_reason, "concentration limit exceeded")

    def test_missing_critical_data_is_data_insufficient_and_non_actionable(self):
        record = complete_candidate(
            proposed_decision=Decision.BUY,
            risk_approved=True,
            maximum_loss=None,
        )
        self.assertEqual(record.data_quality, DataQuality.DATA_INSUFFICIENT)
        self.assertEqual(record.decision, Decision.REJECT)

    def test_option_without_maximum_loss_is_rejected(self):
        record = complete_candidate(
            asset_type=AssetType.OPTION,
            maximum_loss=None,
            proposed_decision=Decision.HOLD,
        )
        self.assertEqual(record.decision, Decision.REJECT)
        self.assertIn("maximum_loss", record.rejection_reason)

    def test_stale_data_is_rejected(self):
        record = complete_candidate(
            now=datetime(2026, 8, 22, 12, 1, tzinfo=timezone.utc),
            max_data_age_seconds=60,
        )
        self.assertEqual(record.data_quality, DataQuality.STALE)
        self.assertEqual(record.decision, Decision.REJECT)

    def test_explicit_degraded_data_cannot_remain_a_hold(self):
        record = complete_candidate(data_quality=DataQuality.STALE)
        self.assertEqual(record.decision, Decision.REJECT)
        self.assertEqual(record.data_quality, DataQuality.STALE)

    def test_invalid_freshness_limit_fails_closed(self):
        record = complete_candidate(max_data_age_seconds=float("nan"))
        self.assertEqual(record.decision, Decision.REJECT)
        self.assertEqual(record.data_quality, DataQuality.INVALID)

    def test_risk_governor_failure_fails_closed(self):
        record = complete_candidate(
            proposed_decision=Decision.BUY,
            risk_approved=True,
            risk_veto=lambda: (_ for _ in ()).throw(RuntimeError("unavailable")),
        )
        self.assertEqual(record.decision, Decision.REJECT)
        self.assertEqual(record.data_quality, DataQuality.INVALID)

    def test_invalid_numeric_data_fails_closed(self):
        record = complete_candidate(expected_risk=float("nan"))
        self.assertEqual(record.data_quality, DataQuality.INVALID)
        self.assertEqual(record.decision, Decision.REJECT)

    def test_wait_does_not_require_risk_approval(self):
        record = complete_candidate(proposed_decision=Decision.WAIT)
        self.assertEqual(record.decision, Decision.WAIT)
        self.assertEqual(record.data_quality, DataQuality.SUFFICIENT)

    def test_btc_adapter_does_not_fabricate_missing_financial_fields(self):
        record = adapt_btc_cad_strategy_evaluation(
            {
                "decision": "BUY CANDIDATE",
                "strategy_score": 80,
                "timestamp": OBSERVED_AT,
            }
        )
        self.assertEqual(record.asset_type, AssetType.CRYPTO)
        self.assertEqual(record.decision, Decision.REJECT)
        self.assertEqual(record.data_quality, DataQuality.DATA_INSUFFICIENT)
        self.assertIn("expected_return", record.rejection_reason)

    def test_btc_adapter_accepts_existing_epoch_timestamp_shape(self):
        record = adapt_btc_cad_strategy_evaluation(
            {
                "decision": "NO TRADE",
                "strategy_score": 45,
                "timestamp": 1787390400,
            }
        )
        self.assertEqual(record.decision, Decision.REJECT)
        self.assertIsInstance(record.observed_at, str)

    def test_records_are_serializable_and_have_required_fields(self):
        record = complete_candidate()
        serialized = record.to_dict()
        required = {
            "instrument",
            "underlying",
            "asset_type",
            "strategy",
            "direction",
            "thesis",
            "expected_return",
            "expected_risk",
            "maximum_loss",
            "liquidity",
            "estimated_transaction_cost",
            "estimated_slippage",
            "confidence",
            "time_horizon",
            "portfolio_impact",
            "concentration_impact",
            "correlation_impact",
            "market_regime_compatibility",
            "risk_score",
            "decision",
            "rejection_reason",
            "data_quality",
        }
        self.assertTrue(required.issubset(serialized))
        self.assertEqual(serialized["decision"], "HOLD")


if __name__ == "__main__":
    unittest.main()