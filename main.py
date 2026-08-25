from config import *
from indicators import (
    calculate_ema,
    calculate_rsi,
    calculate_average_volume
)

from strategy import calculate_strategy_score


# --------------------------------
# SIMULATED MARKET DATA
# --------------------------------

prices = [
    100, 101, 100.5, 102, 103,
    102.5, 104, 105, 104.5, 106,
    107, 106.5, 108, 109, 108.5,
    110, 111, 110.5, 112, 113,
    112.5, 114, 115, 116, 115.5,
    117, 118, 117.5, 119, 120,
    121, 120.5, 122, 123, 124,
    123.5, 125, 126, 125.5, 127
]

volumes = [
    100, 105, 110, 108, 115,
    120, 118, 125, 130, 128,
    135, 140, 138, 145, 150,
    148, 155, 160, 158, 165,
    170, 168, 175, 180, 178,
    185, 190, 188, 195, 200,
    205, 210, 208, 215, 220,
    225, 230, 228, 235, 240
]


# --------------------------------
# CALCULATE INDICATORS
# --------------------------------

ema_9 = calculate_ema(prices, 9)
ema_21 = calculate_ema(prices, 21)
ema_50 = calculate_ema(prices, 50)
ema_200 = calculate_ema(prices, 200)

rsi = calculate_rsi(prices)

average_volume = calculate_average_volume(volumes)

current_price = prices[-1]
current_volume = volumes[-1]


# --------------------------------
# STRATEGY ANALYSIS
# --------------------------------

score, decision, reasons, conditions = calculate_strategy_score(
    ema_9,
    ema_21,
    ema_50,
    ema_200,
    rsi,
    current_price,
    average_volume,
    current_volume
)


# --------------------------------
# DISPLAY RESULTS
# --------------------------------

print("================================")
print("       AI TRADING BOT")
print("================================")

print(f"Current Price: ${current_price:.2f}")

print("--------------------------------")
print("INDICATORS")
print("--------------------------------")

print(f"EMA 9: ${ema_9}")
print(f"EMA 21: ${ema_21}")
print(f"RSI: {rsi:.2f}")
print(f"Average Volume: {average_volume:.2f}")
print(f"Current Volume: {current_volume}")

print("--------------------------------")
print("STRATEGY")
print("--------------------------------")

print(f"Strategy Score: {score}/100")
print(f"Decision: {decision}")

print("--------------------------------")
print("REASONS")
print("--------------------------------")

for reason in reasons:
    print(f"- {reason}")

print("--------------------------------")
print("REAL MONEY TRADING: DISABLED")
print("================================")

from risk_manager import risk_check, get_trade_plan


print("--------------------------------")
print("RISK MANAGEMENT")
print("--------------------------------")

capital = STARTING_CAPITAL
daily_loss = 0.00
trades_today = 0

allowed, risk_message = risk_check(
    capital=capital,
    daily_loss=daily_loss,
    trades_today=trades_today,
    strategy_score=score,
    entry_price=current_price
)

print(f"Trade Allowed: {allowed}")
print(f"Risk Decision: {risk_message}")

if allowed:
    trade_plan = get_trade_plan(
        capital=capital,
        entry_price=current_price
    )

    print("--------------------------------")
    print("TRADE PLAN")
    print("--------------------------------")

    print(f"Position Size: ${trade_plan['position_size']:.2f}")
    print(f"Entry Price: ${trade_plan['entry_price']:.2f}")
    print(f"Stop Loss: ${trade_plan['stop_loss']:.2f}")
    print(f"Take Profit: ${trade_plan['take_profit']:.2f}")

from paper_trading import PaperTradingAccount


print("--------------------------------")
print("PAPER TRADING TEST")
print("--------------------------------")

paper_account = PaperTradingAccount(25.00)

print(f"Starting Balance: ${paper_account.cash:.2f}")

# Simulate a BUY
buy_amount = 10.00
buy_price = 100.00

success, message = paper_account.buy(
    amount=buy_amount,
    price=buy_price
)

print(message)

# Simulate the price increasing
sell_price = 104.00

success, result = paper_account.sell(
    price=sell_price
)

print(f"Paper SELL executed at ${sell_price:.2f}")
print(f"Trade Profit: ${result:.2f}")

status = paper_account.status(sell_price)

print("--------------------------------")
print("ACCOUNT STATUS")
print("--------------------------------")

print(f"Cash: ${status['cash']:.2f}")
print(f"Account Value: ${status['account_value']:.2f}")
print(f"Total Profit: ${status['total_profit']:.2f}")
print(f"Trades: {status['trades']}")

print("--------------------------------")
print("REAL MONEY TRADING: DISABLED")
print("================================")

from market_data import HistoricalMarketData
from generate_test_data import generate_candles


print("--------------------------------")
print("HISTORICAL DATA TEST")
print("--------------------------------")

candles = generate_candles(1000)

market_data = HistoricalMarketData()

for candle in candles:
    market_data.add_candle(
        timestamp=candle["timestamp"],
        open_price=candle["open"],
        high=candle["high"],
        low=candle["low"],
        close=candle["close"],
        volume=candle["volume"]
    )

print(f"Candles Loaded: {market_data.count()}")

closes = market_data.get_closes()
volumes = market_data.get_volumes()

print(f"First Close: ${closes[0]:.2f}")
print(f"Last Close: ${closes[-1]:.2f}")
print(f"First Volume: {volumes[0]:.2f}")
print(f"Last Volume: {volumes[-1]:.2f}")

print("--------------------------------")
print("REAL MONEY TRADING: DISABLED")
print("================================")

from backtester import Backtester


print("--------------------------------")
print("BACKTEST ENGINE TEST")
print("--------------------------------")

backtester = Backtester(25.00)

# Simulated trades

backtester.calculate_trade_result(
    entry_price=100,
    exit_price=104
)

backtester.calculate_trade_result(
    entry_price=105,
    exit_price=103
)

backtester.calculate_trade_result(
    entry_price=100,
    exit_price=106
)

results = backtester.results()

print(f"Starting Capital: ${results['starting_capital']:.2f}")
print(f"Ending Capital: ${results['ending_capital']:.2f}")
print(f"Total Profit: ${results['total_profit']:.2f}")
print(f"Trades: {results['trades']}")
print(f"Wins: {results['wins']}")
print(f"Losses: {results['losses']}")
print(f"Win Rate: {results['win_rate']:.1f}%")

print("--------------------------------")
print("REAL MONEY TRADING: DISABLED")
print("================================")

from strategy_backtest import StrategyBacktester


print("--------------------------------")
print("STRATEGY BACKTEST DIAGNOSTICS")
print("--------------------------------")

strategy_backtester = StrategyBacktester(25.00)
strategy_backtester.run(candles)
results = strategy_backtester.results()

print("--------------------------------")
print("STRATEGY ACCOUNT RESULTS")
print("--------------------------------")
print(f"Starting Capital: ${results['starting_capital']:.2f}")
print(f"Ending Capital: ${results['ending_capital']:.2f}")
print(f"Total Profit: ${results['profit']:.4f}")
print(f"Trades: {results['trades']}")
print(f"Wins: {results['wins']}")
print(f"Losses: {results['losses']}")
print(f"Win Rate: {results['win_rate']:.1f}%")

print("--------------------------------")
print(f"Strategy Evaluations: {results['evaluations']}")
print(f"Highest Strategy Score: {results['highest_score']}/100")
print(f"Scores >= 80: {results['score_80_or_more']}")
print(f"Total Fees: ${results['total_fees']:.4f}")
print(f"Estimated Slippage: ${results['total_slippage']:.4f}")
print(
    f"Maximum Drawdown: "
    f"{results['max_drawdown']:.2f}%"
)
print(f"Lowest RSI: {results['lowest_rsi']:.2f}")
print(f"Highest RSI: {results['highest_rsi']:.2f}")

print("--------------------------------")
print("STRATEGY CONDITION DIAGNOSTICS")
print("--------------------------------")

for condition, count in results["condition_counts"].items():
    print(f"{condition}: {count}/{results['evaluations']}")

from profit_manager import ProfitManager


print("--------------------------------")
print("PROFIT MANAGER TEST")
print("--------------------------------")

profit_manager = ProfitManager(25.00)

print(f"Starting Trading Capital: ${profit_manager.base_capital:.2f}")

# Simulate several profitable trades
profits = [0.40, 0.60, 0.75, 0.80, 1.00, 1.50]

for profit in profits:
    profit_manager.record_profit(profit)

print(f"Profit Reserve: ${profit_manager.profit_reserve:.2f}")

reinvestment = profit_manager.check_reinvestment()

print(f"Reinvested Profit: ${reinvestment:.2f}")

status = profit_manager.status()

print("--------------------------------")
print("PROFIT STATUS")
print("--------------------------------")

print(f"Trading Capital: ${status['base_capital']:.2f}")
print(f"Profit Reserve: ${status['profit_reserve']:.2f}")
print(f"Total Reinvested: ${status['reinvested_profit']:.2f}")

print("--------------------------------")
print("REAL MONEY TRADING: DISABLED")
print("================================")

from historical_data_loader import HistoricalDataLoader


print("--------------------------------")
print("REAL DATA LOADER TEST")
print("--------------------------------")

loader = HistoricalDataLoader()

test_data = [
    {
        "timestamp": 1,
        "open": 100.0,
        "high": 102.0,
        "low": 99.0,
        "close": 101.0,
        "volume": 1000
    },
    {
        "timestamp": 2,
        "open": 101.0,
        "high": 103.0,
        "low": 100.0,
        "close": 102.0,
        "volume": 1100
    }
]

loaded = loader.load_candles(test_data)

print(f"Candles Loaded: {loader.count()}")
print(f"First Close: ${loaded[0]['close']:.2f}")
print(f"Last Close: ${loaded[-1]['close']:.2f}")

print("--------------------------------")
print("REAL MONEY TRADING: DISABLED")
print("================================")