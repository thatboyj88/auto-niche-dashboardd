from config import (
    MAX_POSITION_PERCENT,
    STOP_LOSS_PERCENT,
    TAKE_PROFIT_PERCENT
)


class Backtester:

    def __init__(self, starting_capital, fee_percent=0.004):
        self.starting_capital = starting_capital
        self.capital = starting_capital
        self.fee_percent = fee_percent

        self.trades = []
        self.wins = 0
        self.losses = 0

    def calculate_position_size(self):
        """
        Determine how much capital can be used for one trade.
        """

        return self.capital * MAX_POSITION_PERCENT

    def calculate_trade_result(self, entry_price, exit_price):
        """
        Calculate profit/loss including estimated trading fees.
        """

        position_size = self.calculate_position_size()

        price_change = (
            (exit_price - entry_price) / entry_price
        )

        gross_profit = position_size * price_change

        entry_fee = position_size * self.fee_percent

        exit_value = position_size + gross_profit

        exit_fee = exit_value * self.fee_percent

        net_profit = (
            gross_profit
            - entry_fee
            - exit_fee
        )

        self.capital += net_profit

        if net_profit > 0:
            self.wins += 1
        else:
            self.losses += 1

        self.trades.append({
            "entry": entry_price,
            "exit": exit_price,
            "position_size": position_size,
            "gross_profit": gross_profit,
            "fees": entry_fee + exit_fee,
            "net_profit": net_profit
        })

        return net_profit

    def results(self):

        total_trades = len(self.trades)

        if total_trades > 0:
            win_rate = (
                self.wins / total_trades
            ) * 100
        else:
            win_rate = 0

        total_profit = (
            self.capital - self.starting_capital
        )

        return {
            "starting_capital": self.starting_capital,
            "ending_capital": self.capital,
            "total_profit": total_profit,
            "trades": total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": win_rate
        }