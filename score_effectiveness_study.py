from datetime import datetime, timezone

from multi_period_backtest import MultiPeriodBacktester
from strategy_backtest import StrategyBacktester
from yahoo_btc_cad_data import YahooBTCADMarketData


STARTING_CAPITAL = 25.00
MAX_FORWARD_CANDLES = 20
FORWARD_HORIZONS = (1, 3, 5, 10, 20)
MIN_BUCKET_EVALUATIONS = 20
MIN_INDEPENDENT_PERIODS = 2
SCORE_BUCKETS = (
    ("0-59", 0, 59),
    ("60-69", 60, 69),
    ("70-79", 70, 79),
    ("80-89", 80, 89),
    ("90-100", 90, 100),
)

# These are the complete annual windows from the current stable ten-year
# evidence set. They are pinned so a rolling provider response cannot silently
# replace a studied period.
SCORE_STUDY_PERIODS = (
    {
        "period": "Period A",
        "start_date": "2016-08-20",
        "end_date": "2017-08-19",
        "regime": "Bull",
    },
    {
        "period": "Period B",
        "start_date": "2017-08-20",
        "end_date": "2018-08-19",
        "regime": "Bull",
    },
    {
        "period": "Period C",
        "start_date": "2018-08-20",
        "end_date": "2019-08-19",
        "regime": "Bull",
    },
    {
        "period": "Period D",
        "start_date": "2019-08-20",
        "end_date": "2020-08-18",
        "regime": "Sideways",
    },
    {
        "period": "Period E",
        "start_date": "2020-08-19",
        "end_date": "2021-08-18",
        "regime": "Bull",
    },
    {
        "period": "Period F",
        "start_date": "2021-08-19",
        "end_date": "2022-08-18",
        "regime": "Bear",
    },
    {
        "period": "Period G",
        "start_date": "2022-08-19",
        "end_date": "2023-08-18",
        "regime": "Bull",
    },
    {
        "period": "Period H",
        "start_date": "2023-08-19",
        "end_date": "2024-08-17",
        "regime": "Bull",
    },
    {
        "period": "Period I",
        "start_date": "2024-08-18",
        "end_date": "2025-08-17",
        "regime": "Bull",
    },
    {
        "period": "Period J",
        "start_date": "2025-08-18",
        "end_date": "2026-08-17",
        "regime": "Bear",
    },
)


class ScoreEffectivenessStudy:
    """
    Measure whether recorded strategy scores predict future price movement.

    The StrategyBacktester is run first to produce the original evaluation
    history. Future candles are only read afterward for this report; they are
    never supplied back to score generation or trade execution.
    """

    def analyze_period(self, period_spec, candles):
        backtester = StrategyBacktester(
            starting_capital=STARTING_CAPITAL,
        )
        backtester.run(candles)
        evaluations = backtester.results()["evaluation_history"]
        analysis = self.analyze_evaluations(evaluations, candles)

        return {
            "period": period_spec["period"],
            "start_date": period_spec["start_date"],
            "end_date": period_spec["end_date"],
            "regime": period_spec["regime"],
            "market_return": (
                (candles[-1]["close"] / candles[0]["close"]) - 1
            ) * 100,
            "candles": len(candles),
            **analysis,
        }

    def analyze_evaluations(self, evaluations, candles):
        observations = []

        for evaluation in evaluations:
            candle_index = evaluation["candle"]
            last_forward_index = candle_index + MAX_FORWARD_CANDLES
            if last_forward_index >= len(candles):
                continue

            entry_close = candles[candle_index]["close"]
            forward_candles = candles[
                candle_index + 1:last_forward_index + 1
            ]
            forward_returns = {
                horizon: (
                    (candles[candle_index + horizon]["close"] /
                     entry_close) - 1
                ) * 100
                for horizon in FORWARD_HORIZONS
            }
            maximum_favorable = (
                max(candle["high"] for candle in forward_candles) /
                entry_close - 1
            ) * 100
            maximum_adverse = (
                min(candle["low"] for candle in forward_candles) /
                entry_close - 1
            ) * 100

            observations.append({
                "evaluation_number": evaluation["evaluation_number"],
                "candle": candle_index,
                "timestamp": evaluation["timestamp"],
                "score": evaluation["strategy_score"],
                "entry_close": entry_close,
                "forward_returns": forward_returns,
                "maximum_favorable_movement": maximum_favorable,
                "maximum_adverse_movement": maximum_adverse,
            })

        buckets = {}
        for label, minimum, maximum in SCORE_BUCKETS:
            bucket_observations = [
                observation
                for observation in observations
                if minimum <= observation["score"] <= maximum
            ]
            buckets[label] = self._summarize_bucket(
                label,
                bucket_observations,
            )

        return {
            "evaluations_recorded": len(evaluations),
            "valid_evaluations": len(observations),
            "buckets": buckets,
            "observations": observations,
        }

    def _summarize_bucket(self, label, observations):
        result = {
            "bucket": label,
            "evaluations": len(observations),
            "insufficient_evidence": (
                len(observations) < MIN_BUCKET_EVALUATIONS
            ),
            "maximum_favorable_movement": self._maximum_or_none(
                observation["maximum_favorable_movement"]
                for observation in observations
            ),
            "maximum_adverse_movement": self._minimum_or_none(
                observation["maximum_adverse_movement"]
                for observation in observations
            ),
            "horizons": {},
        }

        for horizon in FORWARD_HORIZONS:
            values = [
                observation["forward_returns"][horizon]
                for observation in observations
            ]
            positive = sum(value > 0 for value in values)
            negative = sum(value < 0 for value in values)
            result["horizons"][horizon] = {
                "average_return": self._average_or_none(values),
                "median_return": self._median_or_none(values),
                "positive_percent": (
                    positive / len(values) * 100
                    if values
                    else None
                ),
                "negative_percent": (
                    negative / len(values) * 100
                    if values
                    else None
                ),
            }

        return result

    @staticmethod
    def aggregate_periods(period_results):
        observations = [
            observation
            for period in period_results
            for observation in period["observations"]
        ]

        buckets = {}
        study = ScoreEffectivenessStudy()
        for label, _, _ in SCORE_BUCKETS:
            bucket_observations = [
                observation
                for observation in observations
                if study._bucket_contains(label, observation["score"])
            ]
            buckets[label] = study._summarize_bucket(
                label,
                bucket_observations,
            )

        return {
            "periods": len(period_results),
            "insufficient_period_coverage": (
                len(period_results) < MIN_INDEPENDENT_PERIODS
            ),
            "valid_evaluations": len(observations),
            "buckets": buckets,
            "period_results": period_results,
        }

    @staticmethod
    def _bucket_contains(label, score):
        for bucket_label, minimum, maximum in SCORE_BUCKETS:
            if label == bucket_label:
                return minimum <= score <= maximum
        raise ValueError(f"Unknown score bucket: {label}")

    @staticmethod
    def _average_or_none(values):
        return sum(values) / len(values) if values else None

    @staticmethod
    def _median_or_none(values):
        if not values:
            return None
        ordered = sorted(values)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[middle]
        return (ordered[middle - 1] + ordered[middle]) / 2

    @staticmethod
    def _maximum_or_none(values):
        values = list(values)
        return max(values) if values else None

    @staticmethod
    def _minimum_or_none(values):
        values = list(values)
        return min(values) if values else None

def select_score_study_periods(source_candles, runner=None):
    runner = runner or MultiPeriodBacktester()
    selected = []

    for specification in SCORE_STUDY_PERIODS:
        candles = [
            candle
            for candle in source_candles
            if specification["start_date"] <=
            runner.format_date(candle["timestamp"]) <=
            specification["end_date"]
        ]
        actual_dates = (
            runner.format_date(candles[0]["timestamp"]),
            runner.format_date(candles[-1]["timestamp"]),
        ) if candles else (None, None)

        if (
            len(candles) != 365 or
            actual_dates != (
                specification["start_date"],
                specification["end_date"],
            )
        ):
            raise RuntimeError(
                f"{specification['period']} no longer matches its "
                "recorded 365-candle date boundary"
            )

        regime, _ = runner.classify_regime(candles)
        if regime != specification["regime"]:
            raise RuntimeError(
                f"{specification['period']} was expected to be "
                f"{specification['regime']}, but is now {regime}"
            )

        selected.append({
            **specification,
            "candles": candles,
        })

    return selected


def load_score_study_periods():
    loader = YahooBTCADMarketData(data_range="10y")
    return select_score_study_periods(loader.load())


def run_score_effectiveness_study():
    study = ScoreEffectivenessStudy()
    period_inputs = load_score_study_periods()
    period_results = [
        study.analyze_period(
            {
                key: value
                for key, value in period.items()
                if key != "candles"
            },
            period["candles"],
        )
        for period in period_inputs
    ]
    by_regime = {}
    for regime in ("Bull", "Bear", "Sideways"):
        regime_periods = [
            result
            for result in period_results
            if result["regime"] == regime
        ]
        by_regime[regime] = study.aggregate_periods(regime_periods)

    return {
        "source": "Yahoo Finance BTC/CAD aggregated daily data",
        "periods": period_results,
        "by_regime": by_regime,
    }


def _format_number(value, suffix="%"):
    if value is None:
        return "N/A"
    return f"{value:+.2f}{suffix}"


def print_report(results):
    print("STRATEGY SCORE EFFECTIVENESS STUDY — ANALYSIS ONLY")
    print("REAL-MONEY TRADING: DISABLED")
    print(
        "Forward returns use the close at each evaluation and exclude "
        "evaluations without a full 20-candle lookahead."
    )

    for regime in ("Bull", "Bear", "Sideways"):
        aggregate = results["by_regime"][regime]
        period_names = ", ".join(
            period["period"]
            for period in results["periods"]
            if period["regime"] == regime
        )
        print("")
        print(f"=== {regime} periods: {period_names or 'none'} ===")
        print(
            f"Valid evaluations: {aggregate['valid_evaluations']} "
            f"across {aggregate['periods']} independent periods"
        )
        if aggregate["insufficient_period_coverage"]:
            print(
                "INSUFFICIENT REGIME COVERAGE — fewer than "
                f"{MIN_INDEPENDENT_PERIODS} independent periods; "
                "treat all results below as exploratory."
            )

        for bucket in aggregate["buckets"].values():
            evidence_label = (
                " — INSUFFICIENT EVIDENCE"
                if bucket["insufficient_evidence"]
                else ""
            )
            print(
                f"Score {bucket['bucket']} "
                f"({bucket['evaluations']} evaluations){evidence_label}: "
                f"MFE20 {_format_number(bucket['maximum_favorable_movement'])}, "
                f"MAE20 {_format_number(bucket['maximum_adverse_movement'])}"
            )
            for horizon in FORWARD_HORIZONS:
                statistics = bucket["horizons"][horizon]
                print(
                    f"  {horizon:>2}d: "
                    f"avg {_format_number(statistics['average_return'])}, "
                    f"median {_format_number(statistics['median_return'])}, "
                    f"positive "
                    f"{_format_number(statistics['positive_percent'])}, "
                    f"negative "
                    f"{_format_number(statistics['negative_percent'])}"
                )


def main():
    results = run_score_effectiveness_study()
    print_report(results)
    return results


if __name__ == "__main__":
    main()