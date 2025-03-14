"""
Implementation of the Crank–Nicolson method for the untransformed Black–Scholes PDE for a European call.
Aim to compare the numerical results against the exact Black–Scholes formula at a chosen time t<T. 
"""

from black_sh_derivative import EuropeanOption, AmericanOption

import numpy as np
from math import log, sqrt, exp
from scipy.stats import norm

def black_scholes_call(S, K, r, sigma, t, T):
    """
    Returns the exact Black-Scholes price of a European call
    with current time t, maturity T, strike K, interest rate r,
    volatility sigma, and underlying price S.
    """
    if t == T:
        # At maturity, payoff is max(S-K, 0)
        return max(S - K, 0)
    tau = T - t  # time to maturity
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
