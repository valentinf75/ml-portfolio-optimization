# logistic regression models for supervised next-day direction prediction

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


def build_logistic_data_for_ticker(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    ticker: str,
    train_end_date: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series] | None:
    # build supervised dataset for one ticker
    # X: technical features at date t
    # y: 1 if return_{t+1} > 0, else 0

    cols = [c for c in features.columns if c.startswith(f"{ticker}_")]
    cols = [c for c in cols if not c.endswith("_PRICE")]
    if not cols:
        return None

    X_all = features[cols].copy()

    returns = prices[ticker].pct_change()
    ret_next = returns.shift(-1)

    df = X_all.copy()
    df["ret_next"] = ret_next
    df = df.dropna()

    y_all = (df["ret_next"] > 0).astype(int)
    X_all = df[cols]

    train_mask = df.index <= train_end_date
    X_train = X_all.loc[train_mask]
    y_train = y_all.loc[train_mask]

    X_test = X_all.loc[~train_mask]
    ret_next_test = df.loc[~train_mask, "ret_next"]

    return X_train, y_train, X_test, ret_next_test


def train_logistic_models(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    tickers: list[str],
    train_end_date: pd.Timestamp,
    C: float = 1.0,
    max_iter: int = 1000,
    use_grid_search: bool = True,
) -> Tuple[Dict[str, LogisticRegression], Dict[str, dict]]:
    # train one logistic regression per ticker

    models: Dict[str, LogisticRegression] = {}
    meta: Dict[str, dict] = {}

    for ticker in tickers:
        result = build_logistic_data_for_ticker(prices, features, ticker, train_end_date)
        if result is None:
            continue

        X_train, y_train, X_test, ret_next_test = result
        if X_train.empty:
            continue

        best_params = None

        if use_grid_search:
            param_grid = {
                "C": [0.1, 1.0, 10.0],
                "class_weight": [None, "balanced"],
            }

            base = LogisticRegression(max_iter=max_iter, solver="lbfgs")
            grid = GridSearchCV(base, param_grid, cv=3, n_jobs=-1, verbose=0)
            grid.fit(X_train, y_train)

            clf = grid.best_estimator_
            best_params = grid.best_params_
        else:
            clf = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs")
            clf.fit(X_train, y_train)

        models[ticker] = clf
        meta[ticker] = {
            "X_test": X_test,
            "ret_next_test": ret_next_test,
            "train_size": len(X_train),
            "best_params": best_params,
        }

    return models, meta


def backtest_logistic_portfolio(
    models: Dict[str, LogisticRegression],
    meta: Dict[str, dict],
    initial_capital: float = 1.0,
) -> pd.Series:
    # backtest a long-only portfolio with weights proportional to p(up)

    tickers = list(models.keys())

    common_index = None
    for ticker in tickers:
        idx = meta[ticker]["X_test"].index
        common_index = idx if common_index is None else common_index.intersection(idx)

    common_index = common_index.sort_values()
    n_dates = len(common_index)
    n_assets = len(tickers)

    prob_matrix = np.zeros((n_dates, n_assets), dtype=float)
    ret_matrix = np.zeros((n_dates, n_assets), dtype=float)

    for j, ticker in enumerate(tickers):
        X_test = meta[ticker]["X_test"].loc[common_index]
        ret_next = meta[ticker]["ret_next_test"].loc[common_index]

        proba_up = models[ticker].predict_proba(X_test)[:, 1]

        prob_matrix[:, j] = proba_up
        ret_matrix[:, j] = ret_next.values

    equity_values = []
    current_value = initial_capital

    for i in range(n_dates):
        probs = prob_matrix[i, :]
        rets = ret_matrix[i, :]

        if probs.sum() <= 0:
            weights = np.full_like(probs, 1.0 / len(probs))
        else:
            weights = probs / probs.sum()

        portfolio_ret = float((weights * rets).sum())
        current_value *= (1.0 + portfolio_ret)
        equity_values.append(current_value)

    equity = pd.Series(equity_values, index=common_index, name="equity_logistic_regression")
    return equity