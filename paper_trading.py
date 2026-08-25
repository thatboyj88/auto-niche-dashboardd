class PaperTradingAccount:

    def __init__(self, starting_capital):
        self.cash = starting_capital
        self.position = 0.0
        self.entry_price = 0.0
        self.total_profit = 0.0
        self.trades = []

    def buy(self, amount, price):
        if amount > self.cash:
            return False, "Insufficient cash."

        self.position = amount / price
        self.entry_price = price
        self.cash -= amount

        return True, f"Paper BUY executed at ${price:.2f}"

    def sell(self, price):
        if self.position <= 0:
            return False, "No open position."

        sale_value = self.position * price
        profit = sale_value - (self.position * self.entry_price)

        self.cash += sale_value
        self.total_profit += profit

        self.trades.append({
            "entry_price": self.entry_price,
            "exit_price": price,
            "profit": profit
        })

        self.position = 0.0
        self.entry_price = 0.0

        return True, profit

    def account_value(self, current_price):
        position_value = self.position * current_price
        return self.cash + position_value

    def status(self, current_price):
        return {
            "cash": self.cash,
            "position": self.position,
            "account_value": self.account_value(current_price),
            "total_profit": self.total_profit,
            "trades": len(self.trades)
        }