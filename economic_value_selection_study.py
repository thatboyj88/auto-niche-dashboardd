"""Predeclared entry-time economic-value selection study.

This module is research-only. It trains one fixed regularized linear model on
research-period control trades, then runs a separate executable candidate
backtest with the frozen selector and unchanged control execution.
"""

from contextlib import contextmanager

import strategy_backtest as strategy_backtest_module
from multi_period_backtest import MultiPeriodBacktester
from out_of_sample_validation import _split_periods
from score_effectiveness_study import (
    SCORE_STUDY_PERIODS,
    select_score_study_periods,
)
from strategy_backtest import StrategyBacktester
from strategy_calibration_study import STARTING_CAPITAL, FEE_PERCENT, SLIPPAGE_PERCENT
from yahoo_btc_cad_data import YahooBTCADMarketData


BREAK_EVEN_PERCENT = (2 * FEE_PERCENT + 2 * SLIPPAGE_PERCENT) * 100
MODEL_ALPHA = 1.0
MIN_TRAINING_TRADES = 30
MIN_TRAINING_PERIODS = 6
MIN_VALIDATION_TRADES_FOR_PROMOTION = 20
MIN_TRADES_PER_USABLE_PERIOD = 3
MIN_VALIDATION_PERIODS_FOR_PROMOTION = 2
CONCENTRATION_LIMIT_PERCENT = 60.0


def _features(evaluation):
    """Fixed entry-time feature vector; no post-entry values are accepted."""
    price = evaluation["current_price"]
    return (
        float(evaluation["strategy_score"]) / 100.0,
        float(evaluation["rsi"]) / 100.0,
        (price - evaluation["ema21"]) / price,
        (evaluation["ema21"] - evaluation["ema50"]) / price,
        (evaluation["ema50"] - evaluation["ema200"]) / price,
        float(evaluation["long_term_trend"]),
        float(evaluation["short_term_momentum"]),
        float(evaluation["volume"]),
        float(evaluation["price_above_ema21"]),
    )


def _solve(matrix, vector):
    """Small deterministic Gaussian solver for the fixed ridge model."""
    size = len(vector)
    augmented = [list(row) + [vector[i]] for i, row in enumerate(matrix)]
    for pivot in range(size):
        row = max(range(pivot, size), key=lambda index: abs(augmented[index][pivot]))
        augmented[pivot], augmented[row] = augmented[row], augmented[pivot]
        divisor = augmented[pivot][pivot]
        if abs(divisor) < 1e-12:
            raise RuntimeError("Economic-value model design matrix is singular")
        augmented[pivot] = [value / divisor for value in augmented[pivot]]
        for index in range(size):
            if index == pivot:
                continue
            factor = augmented[index][pivot]
            augmented[index] = [
                left - factor * right
                for left, right in zip(augmented[index], augmented[pivot])
            ]
    return [augmented[index][-1] for index in range(size)]


class EconomicValueModel:
    """One fixed ridge model with a fixed predicted-net-positive cutoff."""

    def __init__(self, alpha=MODEL_ALPHA):
        self.alpha = alpha
        self.means = None
        self.scales = None
        self.coefficients = None

    def fit(self, rows):
        if len(rows) < MIN_TRAINING_TRADES:
            raise RuntimeError("Insufficient research trades for economic-value model")
        feature_rows = [row[0] for row in rows]
        targets = [row[1] for row in rows]
        width = len(feature_rows[0])
        self.means = [
            sum(row[index] for row in feature_rows) / len(feature_rows)
            for index in range(width)
        ]
        self.scales = [
            max(
                (sum((row[index] - self.means[index]) ** 2 for row in feature_rows)
                 / len(feature_rows)) ** 0.5,
                1e-9,
            )
            for index in range(width)
        ]
        transformed = [
            [1.0] + [
                (value - self.means[index]) / self.scales[index]
                for index, value in enumerate(row)
            ]
            for row in feature_rows
        ]
        size = width + 1
        matrix = [[0.0] * size for _ in range(size)]
        vector = [0.0] * size
        for row, target in zip(transformed, targets):
            for left in range(size):
                vector[left] += row[left] * target
                for right in range(size):
                    matrix[left][right] += row[left] * row[right]
        for index in range(1, size):
            matrix[index][index] += self.alpha
        self.coefficients = _solve(matrix, vector)
        return self

    def predict(self, features):
        if self.coefficients is None:
            raise RuntimeError("Economic-value model must be fitted first")
        transformed = [1.0] + [
            (value - self.means[index]) / self.scales[index]
            for index, value in enumerate(features)
        ]
        return sum(left * right for left, right in zip(self.coefficients, transformed))

    def accepts(self, features):
        return self.predict(features) > 0.0


def _evaluation_map(result):
    return {item["candle"]: item for item in result["evaluation_history"]}


def _training_rows(period_pairs):
    rows = []
    for result, _candles in period_pairs:
        evaluations = _evaluation_map(result)
        for trade in result["trades_history"]:
            evaluation = evaluations.get(trade["entry_candle"])
            if evaluation is not None:
                rows.append((_features(evaluation), trade["net_profit_loss"]))
    return rows


def _fit_period_model(period_pairs):
    rows = _training_rows(period_pairs)
    if len(period_pairs) < MIN_TRAINING_PERIODS:
        raise RuntimeError("Insufficient research periods for economic-value model")
    return EconomicValueModel().fit(rows)


@contextmanager
def _selector(model):
    original = strategy_backtest_module.calculate_strategy_score

    def wrapped(*args):
        score, decision, reasons, conditions = original(*args)
        if decision == "BUY CANDIDATE":
            evaluation = {
                "strategy_score": score,
                "rsi": args[4],
                "ema21": args[1],
                "ema50": args[2],
                "ema200": args[3],
                "current_price": args[5],
                "long_term_trend": conditions["long_term_trend"],
                "short_term_momentum": conditions["short_term_momentum"],
                "volume": conditions["volume"],
                "price_above_ema21": conditions["price_above_ema21"],
            }
            # The wrapper receives EMA values in the same order as the
            # production scorer; the model only sees entry-time values.
            if not model.accepts(_features(evaluation)):
                decision = "NO TRADE"
        return score, decision, reasons, conditions

    strategy_backtest_module.calculate_strategy_score = wrapped
    try:
        yield
    finally:
        strategy_backtest_module.calculate_strategy_score = original


def _run(candles, model=None):
    backtester = StrategyBacktester(starting_capital=STARTING_CAPITAL)
    if model is None:
        backtester.run(candles)
    else:
        with _selector(model):
            backtester.run(candles)
    return backtester.results()


def _metrics(results):
    trades = [
        trade
        for result in results
        for trade in result["trades_history"]
    ]
    gross = sum(trade["gross_profit_loss_before_costs"] for trade in trades)
    net = sum(trade["net_profit_loss"] for trade in trades)
    costs = sum(trade["fees"] + trade["estimated_slippage"] for trade in trades)
    return {
        "periods": len(results),
        "trades": len(trades),
        "gross": gross,
        "net": net,
        "fees": sum(trade["fees"] for trade in trades),
        "slippage": sum(trade["estimated_slippage"] for trade in trades),
        "cost_share": costs / abs(gross) * 100 if gross else 0.0,
        "return_percent": net / (STARTING_CAPITAL * len(results)) * 100
        if results else 0.0,
        "maximum_drawdown": max(
            (result["max_drawdown"] for result in results), default=0.0
        ),
        "positive_cost_clearing_share_percent": (
            sum(
                trade["gross_profit_loss_before_costs"] / (
                    trade["position_size"] * trade["market_entry_price"]
                )
                >= BREAK_EVEN_PERCENT / 100
                for trade in trades
            ) / len(trades) * 100
            if trades else 0.0
        ),
    }


def _period_metrics(results):
    return [
        {
            "period": result.get("period", "UNLABELED"),
            "trades": len(result["trades_history"]),
            "net": sum(
                trade["net_profit_loss"]
                for trade in result["trades_history"]
            ),
        }
        for result in results
    ]


def _periods(candles):
    selected = select_score_study_periods(candles)
    if len(selected) != len(SCORE_STUDY_PERIODS):
        raise RuntimeError("Economic-value study requires all fixed periods")
    research, validation = _split_periods(selected)
    return research, validation


def run_economic_value_selection_study():
    candles = YahooBTCADMarketData(data_range="10y").load()
    if not candles:
        raise RuntimeError("Yahoo BTC/CAD data unavailable")
    research, validation = _periods(candles)
    control_research = [( _run(period["candles"]), period["candles"]) for period in research]
    control_validation = [_run(period["candles"]) for period in validation]
    if len(_training_rows(control_research)) < MIN_TRAINING_TRADES:
        raise RuntimeError("Research does not meet minimum economic-value training sample")
    cross_fitted = []
    for held_out_index, held_out in enumerate(research):
        training = [
            pair for index, pair in enumerate(control_research)
            if index != held_out_index
        ]
        model = _fit_period_model(training)
        cross_fitted.append(_run(held_out["candles"], model))
    model = _fit_period_model(control_research)
    candidate_research = [_run(period["candles"], model) for period in research]
    candidate_validation = []
    for period in validation:
        result = _run(period["candles"], model)
        result["period"] = period["period"]
        candidate_validation.append(result)
    control_research_metrics = _metrics([result for result, _ in control_research])
    candidate_research_metrics = _metrics(candidate_research)
    control_validation_metrics = _metrics(control_validation)
    candidate_validation_metrics = _metrics(candidate_validation)
    return {
        "real_money_trading": False,
        "features": (
            "score, RSI, normalized EMA gaps, and existing entry conditions"
        ),
        "break_even_percent": BREAK_EVEN_PERCENT,
        "control": control_validation_metrics,
        "candidate": candidate_validation_metrics,
        "research_control": control_research_metrics,
        "research_candidate": candidate_research_metrics,
        "cross_fitted_research": _metrics(cross_fitted),
        "validation_periods_evaluated": len(validation),
        "validation_period_metrics": _period_metrics(candidate_validation),
        "minimum_validation_trades_for_promotion": MIN_VALIDATION_TRADES_FOR_PROMOTION,
        "validation_evidence_status": (
            "SUFFICIENT_FOR_PROMOTION_REVIEW"
            if candidate_validation_metrics["trades"] >= MIN_VALIDATION_TRADES_FOR_PROMOTION
            else "INCONCLUSIVE_SMALL_SAMPLE"
        ),
        "promotion_status": "RESEARCH_ONLY",
        "note": (
            "One predeclared model and one predicted-net-positive cutoff were "
            "used. Validation was not used for fitting or tuning."
        ),
    }


def print_report(results):
    print("BTC/CAD ENTRY-TIME ECONOMIC-VALUE SELECTION STUDY — ANALYSIS ONLY")
    print("REAL-MONEY TRADING: DISABLED")
    print(results["note"])
    print(f"Features: {results['features']}")
    print(f"Cost break-even requirement: {results['break_even_percent']:.3f}%")
    print("\n=== Research ===")
    print("Control:", results["research_control"])
    print("Candidate:", results["research_candidate"])
    print("\n=== Untouched validation ===")
    print("Control:", results["control"])
    print("Candidate:", results["candidate"])
    print("\n=== Validation evidence ===")
    print(results["validation_evidence_status"])
    print("\n=== Promotion boundary ===")
    print(results["promotion_status"])
    print("The original entry signal, control exits, and production behavior were unchanged.")


if __name__ == "__main__":
    print_report(run_economic_value_selection_study())