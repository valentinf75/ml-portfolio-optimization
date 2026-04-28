# scripts/run_logistic_regression.py

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Make sure we can import from src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import REPORT_FIG_DIR, REPORT_TABLE_DIR, TICKERS
from src.data.load_data import load_prices
from src.data.preprocess import split_train_test, compute_returns
from src.features.technical_indicators import compute_technical_features
from src.baselines.equal_weight import equity_curve_equal_weight
from src.baselines.markowitz import (
    compute_min_variance_weights,
    equity_curve_markowitz,
)
from src.evaluation.metrics import simple_metrics
from src.models.logistic_regression import (
    train_logistic_models,
    backtest_logistic_portfolio,
)


def main():
    REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_TABLE_DIR.mkdir(parents=True, exist_ok=True)

    # Load prices and split
    prices = load_prices()
    prices_train, prices_test = split_train_test(prices)
    train_end_date = prices_train.index[-1]

    # Compute features
    features = compute_technical_features(prices)

    # Train Logistic Regression models
    models, meta = train_logistic_models(
        prices=prices,
        features=features,
        tickers=TICKERS,
        train_end_date=train_end_date,
    )

    # Backtest Logistic Regression portfolio
    equity_logistic = backtest_logistic_portfolio(models, meta)

    # Baselines on same test index
    equity_eq_full = equity_curve_equal_weight(prices_test)
    equity_eq = equity_eq_full.loc[equity_logistic.index]

    returns_train = compute_returns(prices_train)
    weights_mvp = compute_min_variance_weights(returns_train)
    equity_marko_full = equity_curve_markowitz(prices_test, weights_mvp)
    equity_marko = equity_marko_full.loc[equity_logistic.index]

    # Metrics
    metrics_eq = simple_metrics(equity_eq)
    metrics_marko = simple_metrics(equity_marko)
    metrics_logistic = simple_metrics(equity_logistic)

    metrics_df = pd.DataFrame(
        [metrics_eq, metrics_marko, metrics_logistic],
        index=["baseline_equal_weight", "markowitz_mvp", "logistic_regression"],
    )

    metrics_df.to_csv(
        REPORT_TABLE_DIR / "metrics_test_logistic_vs_baselines.csv"
    )

    # Plot equity curves
    plt.figure(figsize=(10, 5))
    equity_eq.plot(label="Equal-Weight Baseline")
    equity_marko.plot(label="Markowitz MVP")
    equity_logistic.plot(label="Logistic Regression")
    plt.title("Test Equity Curve: Logistic Regression vs Baselines")
    plt.ylabel("Portfolio value (start = 1.0)")
    plt.grid()
    plt.legend()
    plt.savefig(
        REPORT_FIG_DIR / "equity_logistic_vs_baselines_test.png"
    )


if __name__ == "__main__":
    main()
