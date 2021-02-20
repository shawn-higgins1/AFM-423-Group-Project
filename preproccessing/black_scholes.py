import numpy as np
from scipy.stats import norm


def merton_dividend_adjustment(share_price, dividend_yield, t):
    return share_price * np.exp(-dividend_yield * t)


def delta(M, k, r, volatility, t):
    return (np.log(M / k) + (r + (volatility ** 2) / 2) * t) / (volatility * np.sqrt(t))


def delta2(d1, volatility, t):
    return d1 - volatility * np.sqrt(t)


def black_scholes_puts(share_price, dividend_yield, t, k, r, volatility):
    M = merton_dividend_adjustment(share_price, dividend_yield, t)
    d1 = delta(M, k, r, volatility, t)

    d2 = delta2(d1, volatility, t)

    return k * np.exp(-r * t) * norm.cdf(-d2) - M * norm.cdf(-d1)


def black_scholes_calls(share_price, dividend_yield, t, k, r, volatility):
    M = merton_dividend_adjustment(share_price, dividend_yield, t)
    d1 = delta(M, k, r, volatility, t)

    d2 = delta2(d1, volatility, t)

    return M * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)
