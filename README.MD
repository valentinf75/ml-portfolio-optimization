# ML Portfolio Optimization

Benchmarking of portfolio allocation strategies on S&P 500 data, comparing classical finance approaches with supervised machine learning models.

## Strategies Compared

| Strategy | Type |
|----------|------|
| Equal-Weight | Baseline |
| Markowitz MVP (Mean-Variance) | Classical finance |
| Random Forest | Supervised ML |
| Logistic Regression | Supervised ML |

## Pipeline

```
Data (yfinance)
    └─► Feature Engineering   (technical indicators, rolling statistics)
            └─► Model Training     (scikit-learn, PyTorch)
                    └─► Backtesting        (returns, Sharpe, drawdown)
                            └─► Evaluation & Reports
```

## Project Structure

```
├── run_all.py              # Entry point — runs the full pipeline
├── src/
│   ├── config.py           # Global parameters (tickers, dates, hyperparams)
│   ├── data/               # Data loading and preprocessing
│   ├── features/           # Technical indicator engineering
│   ├── models/
│   │   ├── random_forest.py
│   │   └── logistic_regression.py
│   ├── baselines/          # Equal-weight and Markowitz strategies
│   └── evaluation/         # Performance metrics and reporting
├── scripts/                # Individual step runners
├── data/                   # Raw and processed datasets
└── reports/                # Generated backtest reports
```

## Quickstart

```bash
pip install -r requirements.txt
python run_all.py
```

## Key Libraries

`scikit-learn` · `PyTorch` · `yfinance` · `pandas` · `numpy` · `matplotlib` · `Streamlit`

## Results

Each strategy is evaluated on: annualized return, Sharpe ratio, maximum drawdown, and Calmar ratio — across multiple S&P 500 market regimes.