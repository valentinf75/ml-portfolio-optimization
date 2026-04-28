import pandas as pd

from src.config import TEST_RATIO

# clean rows with missing values
def clean_prices(prices) :
    return prices.dropna()


# Daily percentage returns from price data
def compute_returns(prices) :
    returns = prices.pct_change().dropna()
    return returns

# train test split based on TEST_RATIO

def split_train_test(prices):
    split_idx = int(len(prices) * (1 - TEST_RATIO))
    prices_train = prices.iloc[:split_idx]
    prices_test = prices.iloc[split_idx:]
    return prices_train, prices_test
