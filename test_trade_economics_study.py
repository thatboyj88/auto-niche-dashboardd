import unittest

from trade_economics_study import (
    BREAK_EVEN_MOVE_PERCENT,
    TradeEconomicsStudy,
)


def candle(timestamp, close, high=None, low=None):
    return {
        "timestamp": timestamp,
        "open": close,
        "high": high if high is not None else close,
        "low": low if low is not None else close,
        "close": close,
        "volume": 100.0,
    }


def evaluation(candle_index=0, score=85, rsi=65):
    return {
        "candle": candle_index,
        "timestamp": 1_700_000_000,
        "strategy_score": score,
        "rsi": rsi,
        "long_term_trend": True,
        "short_term_momentum": True,
        "rsi_condition": True,
        "volume": True,
        "price_above_ema21": True,
        "decision": "BUY",
    }


class TradeEconomicsStudyTests(unittest.TestCase):
    def setUp(self):
        self.study = TradeEconomicsStudy()

    def test_forward_windows_and_target_times(self):
        candles = [
            candle(index, 100.0, high=100.0)
            for index in range(21)
        ]
        candles[1]["high"] = 101.1
        candles[3]["high"] = 102.1
        candles[5]["high"] = 104.1

        signal = self.study._build_signal(
            evaluation(),
            candles,
            None,
            "Bull",
            "Period A",
        )

        self.assertEqual(signal["forward_returns"][1], 0.0)
        self.assertEqual(signal["targets"]["break_even"]["candles"], 1)
        self.assertEqual(signal["targets"]["two_percent"]["candles"], 3)
        self.assertEqual(signal["targets"]["take_profit"]["candles"], 5)
        self.assertEqual(signal["movement_category"], "Reached 4%")

    def test_unreached_target_is_classified_as_never_break_even(self):
        candles = [candle(index, 100.0) for index in range(21)]

        signal = self.study._build_signal(
            evaluation(),
            candles,
            None,
            "Bear",
            "Period B",
        )

        self.assertFalse(signal["targets"]["break_even"]["reached"])
        self.assertEqual(
            signal["movement_category"],
            "Never reached break-even",
        )

    def test_score_band_includes_score_100_and_rsi_upper_band(self):
        signals = [
            {
                "score": 100,
                "rsi": 85,
                "forward_returns": {
                    1: 1.0,
                    3: 1.0,
                    5: 1.0,
                    10: 1.0,
                    20: 1.0,
                },
                "targets": {
                    "break_even": {"reached": True},
                    "two_percent": {"reached": False},
                    "take_profit": {"reached": False},
                },
                "mfe_percent": 1.0,
                "mae_percent": 0.0,
                "trade": None,
            },
        ]

        score = self.study._band_economics(
            signals,
            lambda signal: signal["score"],
            (("95-100", 95, 100),),
        )
        rsi = self.study._band_economics(
            signals,
            lambda signal: signal["rsi"],
            (("80+", 80, None),),
        )

        self.assertEqual(score["95-100"]["signals"], 1)
        self.assertEqual(rsi["80+"]["signals"], 1)
        self.assertAlmostEqual(
            score["95-100"]["break_even_rate"],
            100.0,
        )

    def test_score_band_includes_intermediate_upper_boundary(self):
        signals = [
            {
                "score": 84,
                "rsi": 65,
                "forward_returns": {
                    1: 0.0,
                    3: 0.0,
                    5: 0.0,
                    10: 0.0,
                    20: 0.0,
                },
                "targets": {
                    "break_even": {"reached": False},
                    "two_percent": {"reached": False},
                    "take_profit": {"reached": False},
                },
                "mfe_percent": 0.0,
                "mae_percent": 0.0,
                "trade": None,
            },
        ]

        score = self.study._band_economics(
            signals,
            lambda signal: signal["score"],
            (("80-84", 80, 84),),
        )

        self.assertEqual(score["80-84"]["signals"], 1)

    def test_rsi_band_with_no_lower_bound_accepts_low_rsi(self):
        signals = [
            {
                "score": 85,
                "rsi": 45,
                "forward_returns": {
                    1: 0.0,
                    3: 0.0,
                    5: 0.0,
                    10: 0.0,
                    20: 0.0,
                },
                "targets": {
                    "break_even": {"reached": False},
                    "two_percent": {"reached": False},
                    "take_profit": {"reached": False},
                },
                "mfe_percent": 0.0,
                "mae_percent": 0.0,
                "trade": None,
            },
        ]

        rsi = self.study._band_economics(
            signals,
            lambda signal: signal["rsi"],
            (("<50", None, 50),),
        )

        self.assertEqual(rsi["<50"]["signals"], 1)

    def test_movement_categories_have_complete_counts(self):
        signals = []
        for category in (
            "Never reached break-even",
            "Reached break-even, not 2%",
            "Reached 2%, not 4%",
            "Reached 4%",
        ):
            signals.append({"movement_category": category})

        result = self.study._movement_categories(signals)

        self.assertEqual(sum(item["signals"] for item in result.values()), 4)
        self.assertEqual(result["Reached 4%"]["percent"], 25.0)

    def test_break_even_threshold_remains_existing_value(self):
        self.assertGreater(BREAK_EVEN_MOVE_PERCENT, 1.0)
        self.assertLess(BREAK_EVEN_MOVE_PERCENT, 1.1)


if __name__ == "__main__":
    unittest.main()