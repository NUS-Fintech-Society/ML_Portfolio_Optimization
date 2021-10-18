import yfinance as yf
import pandas as pd

from data_scraping.default import constituent_list


class DataScraper:

    @classmethod
    def download_ohlc_csv(cls, start_date:str, end_date:str):
        ohlc_data = cls.get_ohlc_csv(start_date, end_date)
        ohlc_data.to_csv("dataset/ohlc_data.csv")

    @staticmethod
    def get_ohlc_csv(start_date:str, end_date: str):
        ohlc = pd.DataFrame()
        # Get three months before so that we can calculate rolling averages and other technical indicators
        for i in constituent_list:
            df = yf.download(i, start=start_date, end=end_date)
            df['ticker'] = i
            ohlc = pd.concat([ohlc, df])
        return ohlc

    @staticmethod
    def load_price_data():
        data = pd.read_csv("dataset/ohlc_data.csv")
        data["Date"] = pd.to_datetime(data["Date"])
        return data.pivot(index="Date", columns="ticker", values="Adj Close")


