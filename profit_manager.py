class ProfitManager:

    def __init__(self, starting_capital):
        self.base_capital = starting_capital
        self.profit_reserve = 0.0
        self.reinvested_profit = 0.0

    def record_profit(self, profit):
        """
        Add realized profit to the profit reserve.
        """
        if profit > 0:
            self.profit_reserve += profit

    def record_loss(self, loss):
        """
        Losses reduce the profit reserve first.
        """
        if loss > 0:
            self.profit_reserve -= loss

            if self.profit_reserve < 0:
                self.profit_reserve = 0.0

    def check_reinvestment(self):
        """
        Determine whether enough profit has accumulated
        to reinvest some of it.
        """

        if self.profit_reserve >= 5.00:

            reinvestment = self.profit_reserve / 2

            self.reinvested_profit += reinvestment
            self.profit_reserve -= reinvestment

            self.base_capital += reinvestment

            return reinvestment

        return 0.0

    def status(self):
        return {
            "base_capital": self.base_capital,
            "profit_reserve": self.profit_reserve,
            "reinvested_profit": self.reinvested_profit
        }