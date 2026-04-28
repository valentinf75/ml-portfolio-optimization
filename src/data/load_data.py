# Download S&P500 dataset from Kaggle and build a prices DataFrame.

from pathlib import Path

import kagglehub
import pandas as pd

from src.config import RAW_DIR, TICKERS, START_DATE, END_DATE
from src.data.preprocess import clean_prices

# Download S&P500 dataset from Kaggle

def download_sp500_csv() :
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(kagglehub.dataset_download("camnugent/sandp500"))
    csv_path = dataset_path / "all_stocks_5yr.csv"
    return csv_path

# Load kaggle CSV into a df

def load_raw_sp500() :
    csv_path = download_sp500_csv()
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


# Keep only selected tickers and build a wide prices table

def build_prices(df_raw) :
    df = df_raw.loc[df_raw["Name"].isin(TICKERS), ["date", "Name", "close"]].copy()
    df.rename(columns={"Name": "ticker", "close": "price"}, inplace=True)
    prices = df.pivot(index="date", columns="ticker", values="price").sort_index()
    return prices


def load_prices() :
    df_raw = load_raw_sp500()
    prices = build_prices(df_raw)
    prices = clean_prices(prices)
    prices = prices.sort_index()
    prices = prices.loc[START_DATE:END_DATE]
    return prices
