from indicators import (
    calculate_ema,
    calculate_rsi,
    calculate_average_volume
)

from strategy import calculate_strategy_score
from config import (
    MAX_DAILY_LOSS_PERCENT,
    MAX_POSITION_PERCENT,
    MAX_TRADES_PER_DAY,
    MIN_STRATEGY_SCORE,
    STOP_LOSS_PERCENT,
    TAKE_PROFIT_PERCENT,
    FEE_PERCENT,
    SLIPPAGE_PERCENT,
)


class StrategyBacktester:

    def __init__(
        self,
        starting_capital,
        fee_percent=FEE_PERCENT,
        slippage_percent=SLIPPAGE_PERCENT
    ):
        self.starting_capital = starting_capital
        self.capital = starting_capital
        self.equity_curve = [starting_capital]

        self.fee_percent = fee_percent
        self.slippage_percent = slippage_percent

        self.position = 0.0
        self.entry_price = 0.0
        self.entry_value = 0.0
        self.entry_candle = None
        self.entry_timestamp = None
        self.entry_score = 0
        self.entry_rsi = 0.0
        self.entry_fee = 0.0
        self.entry_slippage = 0.0

        self.trades = []
        self.evaluation_history = []

        self.wins = 0
        self.losses = 0

        self.evaluations = 0
        self.highest_score = 0
        self.score_80_or_more = 0
        self.lowest_rsi = 100
        self.highest_rsi = 0

        self.condition_counts = {
            "long_term_trend": 0,
            "short_term_momentum": 0,
            "rsi": 0,
            "volume": 0,
            "price_above_ema21": 0
        }

        self.trades_today = 0
        self.current_day = None
        self.daily_starting_capital = starting_capital

        self.max_drawdown = 0.0
        self.peak_capital = starting_capital

        self.total_fees = 0.0
        self.total_slippage = 0.0

    def run(self, candles):

        for i in range(len(candles)):

            if i < 200:
                continue

            candle = candles[i]
            current_day = candle["timestamp"] // 86400

            if current_day != self.current_day:

                self.current_day = current_day
                self.trades_today = 0
                self.daily_starting_capital = self.capital

            historical = candles[:i + 1]

            prices = [
                candle["close"]
                for candle in historical
            ]

            volumes = [
                candle["volume"]
                for candle in historical
            ]

            current_price = prices[-1]
            current_volume = volumes[-1]

            ema_9 = calculate_ema(prices, 9)
            ema_21 = calculate_ema(prices, 21)
            ema_50 = calculate_ema(prices, 50)
            ema_200 = calculate_ema(prices, 200)

            rsi = calculate_rsi(prices)

            if rsi is not None:
                self.lowest_rsi = min(
                    self.lowest_rsi,
                    rsi
                )

                self.highest_rsi = max(
                    self.highest_rsi,
                    rsi
                )

            average_volume = calculate_average_volume(
                volumes
            )

            if None in (
                ema_9,
                ema_21,
                ema_50,
                ema_200,
                rsi,
                average_volume
            ):
                continue

            self.evaluations += 1

            score, decision, reasons, conditions = (
                calculate_strategy_score(
                    ema_9,
                    ema_21,
                    ema_50,
                    ema_200,
                    rsi,
                    current_price,
                    average_volume,
                    current_volume
                )
            )

            strategy_decision = (
                "BUY"
                if decision == "BUY CANDIDATE"
                else "NO TRADE"
            )

            self.evaluation_history.append({
                "evaluation_number": self.evaluations,
                "candle": i,
                "timestamp": candle["timestamp"],
                "strategy_score": score,
                "decision": strategy_decision,
                "long_term_trend": conditions["long_term_trend"],
                "short_term_momentum": conditions["short_term_momentum"],
                "rsi_condition": conditions["rsi"],
                "volume": conditions["volume"],
                "price_above_ema21": conditions["price_above_ema21"],
                "rsi": rsi,
                "ema21": ema_21,
                "ema50": ema_50,
                "ema200": ema_200,
                "current_price": current_price
            })

            # -----------------------------
            # MANAGE OPEN POSITION
            # -----------------------------

            if self.position > 0:

                stop_price = (
                    self.entry_price *
                    (1 - STOP_LOSS_PERCENT)
                )

                target_price = (
                    self.entry_price *
                    (1 + TAKE_PROFIT_PERCENT)
                )

                if current_price <= stop_price:

                    self.close_position(
                        current_price,
                        "STOP LOSS",
                        i,
                        candle["timestamp"]
                    )

                    self._record_equity(current_price)
                    continue

                if current_price >= target_price:

                    self.close_position(
                        current_price,
                        "TAKE PROFIT",
                        i,
                        candle["timestamp"]
                    )

                    self._record_equity(current_price)
                    continue

            # -----------------------------
            # LOOK FOR NEW TRADE
            # -----------------------------

            if self.position == 0:

                self.highest_score = max(
                    self.highest_score,
                    score
                )

                if score >= MIN_STRATEGY_SCORE:
                    self.score_80_or_more += 1

                for condition, passed in conditions.items():

                    if passed:
                        self.condition_counts[
                            condition
                        ] += 1

                if decision != "BUY CANDIDATE":
                    self._record_equity(current_price)
                    continue

                # Maximum configured trades per day
                if self.trades_today >= MAX_TRADES_PER_DAY:
                    self._record_equity(current_price)
                    continue

                # Maximum configured daily loss
                daily_loss = (
                    self.daily_starting_capital
                    - self.capital
                )

                if daily_loss >= (
                    self.daily_starting_capital * MAX_DAILY_LOSS_PERCENT
                ):
                    self._record_equity(current_price)
                    continue

                # -----------------------------
                # ENTER POSITION
                # -----------------------------

                position_value = (
                    self.capital * MAX_POSITION_PERCENT
                )

                # Entry slippage
                actual_entry_price = (
                    current_price *
                    (1 + self.slippage_percent)
                )

                entry_fee = (
                    position_value *
                    self.fee_percent
                )

                total_entry_cost = (
                    position_value +
                    entry_fee
                )

                if total_entry_cost > self.capital:
                    self._record_equity(current_price)
                    continue

                self.position = (
                    position_value /
                    actual_entry_price
                )

                self.entry_price = (
                    actual_entry_price
                )

                self.entry_value = (
                    position_value
                )
                self.entry_candle = i
                self.entry_timestamp = candle["timestamp"]
                self.entry_score = score
                self.entry_decision = strategy_decision
                self.entry_rsi = rsi
                self.entry_fee = entry_fee
                self.entry_slippage = (
                    position_value *
                    self.slippage_percent
                )

                self.capital -= total_entry_cost

                self.total_fees += entry_fee

                self.total_slippage += (
                    self.entry_slippage
                )

                self.trades_today += 1

            self._record_equity(current_price)

        # Close remaining position
        if self.position > 0:

            self.close_position(
                candles[-1]["close"],
                "END OF TEST",
                len(candles) - 1,
                candles[-1]["timestamp"]
            )

            if self.equity_curve:
                self.equity_curve[-1] = self.capital

    def _record_equity(self, current_price):
        account_value = (
            self.capital +
            (self.position * current_price)
        )

        self.equity_curve.append(account_value)

    def close_position(
        self,
        market_exit_price,
        reason,
        exit_candle=None,
        exit_timestamp=None
    ):

        # Exit slippage
        actual_exit_price = (
            market_exit_price *
            (1 - self.slippage_percent)
        )

        gross_value = (
            self.position *
            actual_exit_price
        )

        exit_fee = (
            gross_value *
            self.fee_percent
        )

        exit_slippage = (
            gross_value *
            self.slippage_percent
        )

        net_value = (
            gross_value -
            exit_fee
        )

        profit = (
            net_value -
            self.entry_value
        )

        self.capital += net_value

        self.total_fees += exit_fee

        self.total_slippage += (
            exit_slippage
        )

        gross_profit_loss = (
            gross_value -
            self.entry_value
        )

        total_trade_fees = (
            self.entry_fee +
            exit_fee
        )

        total_trade_slippage = (
            self.entry_slippage +
            exit_slippage
        )

        net_profit_loss = (
            gross_profit_loss -
            total_trade_fees
        )

        market_entry_price = (
            self.entry_price /
            (1 + self.slippage_percent)
        )
        market_exit_price = (
            actual_exit_price /
            (1 - self.slippage_percent)
        )
        gross_profit_loss_before_costs = (
            self.position *
            (market_exit_price - market_entry_price)
        )

        trade = {
            "trade_number": len(self.trades) + 1,
            "entry_candle": self.entry_candle,
            "entry_timestamp": self.entry_timestamp,
            "entry": self.entry_price,
            "entry_price": self.entry_price,
            "exit_candle": exit_candle,
            "exit_timestamp": exit_timestamp,
            "exit": actual_exit_price,
            "exit_price": actual_exit_price,
            "position_size": self.position,
            "gross_profit_loss": gross_profit_loss,
            "market_entry_price": market_entry_price,
            "market_exit_price": market_exit_price,
            "gross_profit_loss_before_costs":
                gross_profit_loss_before_costs,
            "profit": profit,
            "reason": reason,
            "fees": total_trade_fees,
            "estimated_slippage": total_trade_slippage,
            "net_profit_loss": net_profit_loss,
            "strategy_score": self.entry_score,
            "strategy_decision": self.entry_decision,
            "rsi_at_entry": self.entry_rsi
        }

        self.trades.append(trade)

        if profit > 0:
            self.wins += 1
        else:
            self.losses += 1

        self.position = 0.0
        self.entry_price = 0.0
        self.entry_value = 0.0
        self.entry_candle = None
        self.entry_timestamp = None
        self.entry_score = 0
        self.entry_decision = "NO TRADE"
        self.entry_rsi = 0.0
        self.entry_fee = 0.0
        self.entry_slippage = 0.0

        # Track peak and drawdown
        if self.capital > self.peak_capital:
            self.peak_capital = self.capital

        drawdown = (
            self.peak_capital -
            self.capital
        )

        if self.peak_capital > 0:

            drawdown_percent = (
                drawdown /
                self.peak_capital
            ) * 100

            self.max_drawdown = max(
                self.max_drawdown,
                drawdown_percent
            )

    def results(self):

        total_trades = len(self.trades)

        if total_trades > 0:

            win_rate = (
                self.wins /
                total_trades
            ) * 100

        else:
            win_rate = 0

        total_profit = (
            self.capital -
            self.starting_capital
        )

        return {
            "starting_capital":
                self.starting_capital,

            "ending_capital":
                self.capital,

            "profit":
                total_profit,

            "trades":
                total_trades,

            "wins":
                self.wins,

            "losses":
                self.losses,

            "win_rate":
                win_rate,

            "evaluations":
                self.evaluations,

            "highest_score":
                self.highest_score,

            "score_80_or_more":
                self.score_80_or_more,

            "condition_counts":
                self.condition_counts,

            "total_fees":
                self.total_fees,

            "total_slippage":
                self.total_slippage,

            "max_drawdown":
                self.max_drawdown,

            "lowest_rsi":
                self.lowest_rsi,

            "highest_rsi":
                self.highest_rsi,

            "equity_curve":
                list(self.equity_curve),

            "trades_history":
                list(self.trades),

            "evaluation_history":
                list(self.evaluation_history)
        }