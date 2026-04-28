import pandas as pd

from src.data.preprocess import compute_returns

# simple moving averages for each asset
def add_moving_averages(prices, windows=(20, 60)):
    df = prices.copy()
    for w in windows:
        df[f"MA{w}"] = prices.rolling(w).mean()
    return df


def add_volatility(returns, window=20):
    vol = returns.rolling(window).std()
    vol.columns = [f"{c}_VOL{window}" for c in vol.columns]
    return vol


def compute_technical_features(prices, ma_windows=(20, 60), vol_window=20):
    # compute daily returns
    returns = compute_returns(prices)

    feature_frames = []

    # compute features for each asset separately
    for ticker in prices.columns:
        p = prices[ticker]
        r = returns[ticker]

        # create a small table for this asset
        df_t = pd.DataFrame(index=prices.index)

        df_t[f"{ticker}_PRICE"] = p
        df_t[f"{ticker}_RET"] = r

        # moving averages
        for w in ma_windows:
            df_t[f"{ticker}_MA{w}"] = p.rolling(w).mean()

        # rolling volatility
        df_t[f"{ticker}_VOL{vol_window}"] = r.rolling(vol_window).std()

        feature_frames.append(df_t)

    # merge all assets features into a single table
    features = pd.concat(feature_frames, axis=1)

    # remove incomplete rows
    features = features.dropna()

    return features
