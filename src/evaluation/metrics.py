import pandas as pd

# max drawdown = min over time of equity / rolling peak - 1
def compute_max_drawdown(equity) :
    cumulative_max = equity.cummax()
    drawdowns = equity / cumulative_max - 1.0
    return drawdowns.min()


def simple_metrics(equity):
    # compute basic portfolio metrics
    returns = equity.pct_change().dropna()

    # total return from start to end
    total_return = equity.iloc[-1] / equity.iloc[0] - 1

    # daily volatility and sharpe (rf = 0)
    volatility = returns.std()
    if volatility != 0:
        sharpe = returns.mean() / volatility
    else:
        sharpe = 0.0

    # max drawdown
    max_dd = compute_max_drawdown(equity)

    # annual return approximation 252 trading days
    n_days = len(equity)
    if n_days > 1:
        annual_return = (equity.iloc[-1] / equity.iloc[0]) ** (252 / (n_days - 1)) - 1
    else:
        annual_return = 0.0

    # calmar ratio = annual return / abs(max drawdown)
    if max_dd < 0:
        calmar = annual_return / abs(max_dd)
    else:
        calmar = 0.0

    return {
        "total_return": total_return,
        "daily_volatility": volatility,
        "daily_sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
    }

