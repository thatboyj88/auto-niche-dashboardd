from datetime import datetime, timezone

from kraken_live_data import KrakenMarketData
from strategy_backtest import StrategyBacktester
from config import STARTING_CAPITAL


INTERVAL_MINUTES = 60

CONDITION_LABELS = {
    "long_term_trend": "Long-term trend",
    "short_term_momentum": "Short-term momentum",
    "rsi": "RSI",
    "volume": "Volume",
    "price_above_ema21": "Price above EMA21",
}


def format_timestamp(timestamp):
    return datetime.fromtimestamp(
        timestamp,
        tz=timezone.utc
    ).isoformat()


def main():
    market_data = KrakenMarketData(
        interval=INTERVAL_MINUTES
    )
    candles = market_data.load()

    print("--------------------------------")
    print("KRAKEN LIVE MARKET PAPER BACKTEST")
    print("--------------------------------")
    print(f"Pair: {market_data.pair_name or 'XBT/CAD'}")
    print(
        "Kraken Pair Identifier: "
        f"{market_data.pair_identifier or 'Unavailable'}"
    )
    print(f"Interval: {INTERVAL_MINUTES} minutes")

    if not candles:
        print(
            "ERROR: "
            f"{market_data.last_error or 'No candles loaded'}"
        )
        print("--------------------------------")
        print("REAL-MONEY TRADING: DISABLED")
        print("================================")
        return 1

    backtester = StrategyBacktester(
        starting_capital=STARTING_CAPITAL
    )
    backtester.run(candles)
    results = backtester.results()

    condition_counts = results["condition_counts"]

    print(f"Candles loaded: {len(candles)}")
    print(
        "First timestamp: "
        f"{format_timestamp(candles[0]['timestamp'])}"
    )
    print(
        "Last timestamp: "
        f"{format_timestamp(candles[-1]['timestamp'])}"
    )
    print("")
    print(f"Starting capital: ${results['starting_capital']:.2f}")
    print(f"Ending capital: ${results['ending_capital']:.4f}")
    print(f"Total profit: ${results['profit']:.4f}")
    print(f"Trades: {results['trades']}")
    print(f"Wins: {results['wins']}")
    print(f"Losses: {results['losses']}")
    print(f"Win rate: {results['win_rate']:.2f}%")
    print(f"Strategy evaluations: {results['evaluations']}")
    print(
        "Highest strategy score: "
        f"{results['highest_score']}/100"
    )
    print(f"Scores >=80: {results['score_80_or_more']}")
    print(f"Total fees: ${results['total_fees']:.4f}")
    print(
        "Estimated slippage: "
        f"${results['total_slippage']:.4f}"
    )
    print(
        "Maximum drawdown: "
        f"{results['max_drawdown']:.2f}%"
    )
    print(f"Lowest RSI: {results['lowest_rsi']:.2f}")
    print(f"Highest RSI: {results['highest_rsi']:.2f}")
    print(
        "RSI conditions passed: "
        f"{condition_counts['rsi']}"
    )
    print("")
    print("Strategy-condition counters:")

    for key, label in CONDITION_LABELS.items():
        print(f"{label}: {condition_counts[key]}")

    print("--------------------------------")
    print("REAL-MONEY TRADING: DISABLED")
    print("================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())