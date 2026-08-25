# AI Trading Bot Configuration
# REAL TRADING IS DISABLED

STARTING_CAPITAL = 25.00

# Execution-cost policy shared by historical and genuine paper paths.
# These values are frozen; changing them requires a separate research review.
FEE_PERCENT = 0.004
SLIPPAGE_PERCENT = 0.001

# Trading settings
MAX_POSITION_PERCENT = 0.40
STOP_LOSS_PERCENT = 0.02
TAKE_PROFIT_PERCENT = 0.04

# Exit promotion guardrails. The control is the only production exit
# configuration; study candidates must pass an independent robustness gate
# before these values can be changed.
EXIT_CONTROL = (STOP_LOSS_PERCENT * 100, TAKE_PROFIT_PERCENT * 100)
EXIT_PROMOTION_MIN_UNTOUCHED_PERIODS = 3
EXIT_PROMOTION_MAX_COST_SHARE_PERCENT = 0.0

# Safety settings
MAX_DAILY_LOSS_PERCENT = 0.03
MAX_TRADES_PER_DAY = 3

# Strategy settings
MIN_STRATEGY_SCORE = 80

RSI_MIN = 45
RSI_MAX = 68

VOLUME_MULTIPLIER = 1.20

# Trading mode
PAPER_TRADING = True
LIVE_TRADING = False