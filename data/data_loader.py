import yfinance as yf
import pandas as pd


class DataLoader:
    def __init__(self, tickers, start_date="2015-01-01"):
        self.tickers = tickers
        self.start_date = start_date

    def fetch_data(self):
        print("Downloading data...")
        data = yf.download(self.tickers, start=self.start_date)

        if data.empty:
            raise ValueError("No data downloaded. Check tickers.")

        # Extract Close prices (more reliable for Indian ETFs)
        if isinstance(data.columns, pd.MultiIndex):
            prices = data["Close"]
        else:
            prices = data[["Close"]]

        prices = prices.dropna()

        print("Download complete")
        print("Assets downloaded:", prices.columns.tolist())
        print("Shape:", prices.shape)

        return prices