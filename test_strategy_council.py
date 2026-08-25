import unittest

from strategy_council import (
    LENSES,
    aggregate_strategy_council,
    evaluate_risk_governor,
    finalize_council_decision,
    normalize_lens_evidence,
)


def evidence(decision="BUY", confidence=80):
    return {
        lens: {"status": "AVAILABLE", "decision": decision, "confidence": confidence}
        for lens in LENSES
    }


class StrategyCouncilTests(unittest.TestCase):
    def test_every_lens_is_explicit(self):
        result = normalize_lens_evidence({"trend": {"status": "AVAILABLE", "decision": "BUY"}})
        self.assertEqual(set(result), set(LENSES))
        self.assertEqual(result["value"]["status"], "UNAVAILABLE")

    def test_missing_lens_blocks_council(self):
        result = aggregate_strategy_council(evidence())
        result = aggregate_strategy_council({key: value for key, value in evidence().items() if key != "macro"})
        self.assertEqual(result.decision, "WAIT")
        self.assertEqual(result.data_quality, "DATA_INSUFFICIENT")

    def test_disagreement_blocks_consolidated_action(self):
        values = evidence("BUY")
        values["macro"] = {"status": "AVAILABLE", "decision": "SELL", "confidence": 90}
        values["dividend"] = {"status": "AVAILABLE", "decision": "SELL", "confidence": 90}
        self.assertEqual(aggregate_strategy_council(values).decision, "WAIT")

    def test_council_consolidates_unanimous_lenses(self):
        result = aggregate_strategy_council(evidence("BUY", 75))
        self.assertEqual(result.decision, "BUY")
        self.assertEqual(result.data_quality, "SUFFICIENT")
        self.assertEqual(result.confidence, 75)

    def test_risk_governor_veto_overrides_action(self):
        council = aggregate_strategy_council(evidence("BUY"))
        governor = evaluate_risk_governor(
            "BUY", exposure_percent=.8, concentration_percent=.8,
            proposed_position_percent=.8,
        )
        final = finalize_council_decision(council, governor)
        self.assertEqual(final.final_action, "REJECT")
        self.assertIn("concentration", final.final_reason)

    def test_risk_governor_blocks_unhealthy_or_live_policy(self):
        result = evaluate_risk_governor(
            "BUY", exposure_percent=0, concentration_percent=0,
            proposed_position_percent=0, data_health="DEGRADED",
            paper_trading=True, live_trading=True,
        )
        self.assertFalse(result.approved)
        self.assertIn("market data health", result.veto_reason)
        self.assertIn("paper-only", result.veto_reason)

    def test_non_actionable_council_result_remains_wait(self):
        council = aggregate_strategy_council({})
        governor = evaluate_risk_governor(
            "WAIT", exposure_percent=0, concentration_percent=0, proposed_position_percent=0,
        )
        self.assertEqual(finalize_council_decision(council, governor).final_action, "WAIT")


if __name__ == "__main__":
    unittest.main()