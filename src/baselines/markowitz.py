#Markowitz Minimum Variance Portfolio (MVP).
# baselines with lowest possible variance for a given return level
# uses the covariance matrix of asset returns to find the optimal weights

import numpy as np
import pandas as pd


def compute_min_variance_weights(returns):
 
    # compute the Minimum Variance Portfolio (MVP)
   
    # Covariance matrix of returns
    cov = returns.cov()
    # Vector [1;1;1...] same shape cov for calculations MVP formula
    ones = np.ones(cov.shape[0])

    # Inverse covariance matrix
    inv_cov = np.linalg.inv(cov)

    # Apply MVP closed-form formula
    #muliplying the inverse covariance matrix by a vector of ones
    raw_weights = np.dot(inv_cov, ones)
    # normalize
    weights = raw_weights / raw_weights.sum()

    return weights


# Compute the equity curve of a Markowitz Minimum Variance portfolio
def equity_curve_markowitz(prices, weights) :
   
    
    prices = prices.dropna()

    # Price relatives compared to the first day
    relatives = prices / prices.iloc[0]

    # Weighted average of relatives using the MVP weights
    equity = (relatives * weights).sum(axis=1)
    equity.name = "equity_markowitz"

    return equity
