import numpy as np
from scipy.stats import norm


def black_scholes_with_dividends(share_price, dividend_yield, t, k, r, volatility):
    M = share_price * np.exp(-dividend_yield * t)
    d1 = (np.log(M / k) + (r + (volatility ** 2) / 2) * t) / (volatility * np.sqrt(t))

    d2 = d1 - volatility * np.sqrt(t)

    return k * np.exp(-r * t) * norm.cdf(-d2) - M * norm.cdf(-d1)
