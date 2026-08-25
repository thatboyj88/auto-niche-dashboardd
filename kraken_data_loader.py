import csv
import io
import zipfile


class KrakenDataLoader:
    """
    Loads Kraken historical OHLCVT CSV data and converts it
    to the standard candle format used by the trading bot.
    """

    def __init__(self):
        self.candles = []

    def load_csv(self, csv_text):
        self.candles = []

        reader = csv.reader(
            io.StringIO(csv_text)
        )

        for row in reader:

            if not row:
                continue

            # Skip headers or malformed rows
            if not self._is_numeric(row[0]):
                continue

            if len(row) < 6:
                continue

            try:
                timestamp = int(float(row[0]))
                open_price = float(row[1])
                high = float(row[2])
                low = float(row[3])
                close = float(row[4])
                volume = float(row[6]) if len(row) >= 7 else float(row[5])

                self.candles.append({
                    "timestamp": timestamp,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume
                })

            except (ValueError, TypeError):
                continue

        self.candles.sort(
            key=lambda candle: candle["timestamp"]
        )

        return self.candles

    def load_zip(self, zip_path, pair="XBT_CAD"):
        self.candles = []

        with zipfile.ZipFile(zip_path, "r") as archive:

            matching_files = [
                name
                for name in archive.namelist()
                if pair.lower() in name.lower()
                and name.lower().endswith(".csv")
            ]

            if not matching_files:
                raise FileNotFoundError(
                    f"No CSV found for pair: {pair}"
                )

            for filename in matching_files:

                with archive.open(filename) as file:

                    content = file.read().decode(
                        "utf-8",
                        errors="replace"
                    )

                    self.load_csv(content)

        self.candles.sort(
            key=lambda candle: candle["timestamp"]
        )

        return self.candles

    def count(self):
        return len(self.candles)

    def get_closes(self):
        return [
            candle["close"]
            for candle in self.candles
        ]

    def get_volumes(self):
        return [
            candle["volume"]
            for candle in self.candles
        ]

    @staticmethod
    def _is_numeric(value):
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False