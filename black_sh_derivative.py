"""
Black-Scholes option pricing

created on Feb 17
authur: Wenxin.J
"""

import numpy as np
from scipy.stats import norm

class Option:
    """Base Option class with common attributes and methods"""
    def __init__(self, S0, K, T, r, sigma):
        """
        Initialize base option parameters
        
        Parameters:
        S0 : float
            Current stock price
        K : float
            Strike price
        T : float
            Time to maturity (in years)
        r : float
            Risk-free interest rate (annual)
        sigma : float
            Volatility of the stock (annual)
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
    
    def _calculate_d1(self):
        """Calculate d1 parameter"""
        numerator = (np.log(self.S0 / self.K) + 
                    (self.r + self.sigma**2/2) * self.T)
        denominator = self.sigma * np.sqrt(self.T)
        return numerator / denominator
    
    def _calculate_d2(self):
        """Calculate d2 parameter"""
        return self._calculate_d1() - self.sigma * np.sqrt(self.T)

class EuropeanOption(Option):
    """European Option pricing using Black-Scholes formula"""
    
    def call_price(self):
        """Calculate European call option price"""
        d1 = self._calculate_d1()
        d2 = self._calculate_d2()
        
        term1 = self.S0 * norm.cdf(d1)
        term2 = self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        
        return term1 - term2
    
    def put_price(self):
        """Calculate European put option price"""
        d1 = self._calculate_d1()
        d2 = self._calculate_d2()
        
        term1 = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)
        term2 = self.S0 * norm.cdf(-d1)
        
        return term1 - term2
    
    def calculate_greeks(self):
        """Calculate option Greeks"""
        d1 = self._calculate_d1()
        d2 = self._calculate_d2()
        
        # Calculate Delta
        call_delta = norm.cdf(d1)
        put_delta = call_delta - 1
        
        # Calculate Gamma (same for call and put)
        gamma = norm.pdf(d1) / (self.S0 * self.sigma * np.sqrt(self.T))
        
        # Calculate Theta
        call_theta = (-self.S0 * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T)) -
                     self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2))
        put_theta = (-self.S0 * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T)) +
                    self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2))
        
        return {
            'call_delta': call_delta,
            'put_delta': put_delta,
            'gamma': gamma,
            'call_theta': call_theta,
            'put_theta': put_theta
        }

class AmericanOption(Option):
    """American Option pricing using numerical methods"""
    
    def binomial_tree_price(self, steps=100, option_type='call'):
        """
        Price American option using binomial tree
        
        Parameters:
        steps: int
            Number of steps in the binomial tree
        option_type: str
            'call' or 'put'
        """
        dt = self.T / steps
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1/u
        p = (np.exp(self.r * dt) - d) / (u - d)
        
        # Initialize asset prices at final nodes
        prices = np.zeros(steps + 1)
        for i in range(steps + 1):
            prices[i] = self.S0 * (u ** (steps - i)) * (d ** i)
        
        # Initialize option values at final nodes
        if option_type.lower() == 'call':
            values = np.maximum(prices - self.K, 0)
        else:  # put
            values = np.maximum(self.K - prices, 0)
        
        # Backward induction through the tree
        for step in range(steps-1, -1, -1):
            for i in range(step + 1):
                S = self.S0 * (u ** (step - i)) * (d ** i)
                hold_value = np.exp(-self.r * dt) * (p * values[i] + (1-p) * values[i+1])
                if option_type.lower() == 'call':
                    exercise_value = max(S - self.K, 0)
                else:  # put
                    exercise_value = max(self.K - S, 0)
                values[i] = max(hold_value, exercise_value)
        
        return values[0]

# Example usage
if __name__ == "__main__":
    # Example parameters
    S0 = 100    # Current stock price
    K = 100     # Strike price
    T = 1       # One year until expiry
    r = 0.05    # 5% risk-free rate
    sigma = 0.2 # 20% volatility
    
    # European Option Example
    euro_option = EuropeanOption(S0, K, T, r, sigma)
    euro_call = euro_option.call_price()
    euro_put = euro_option.put_price()
    greeks = euro_option.calculate_greeks()
    
    print("European Option Prices:")
    print(f"Call: ${euro_call:.2f}")
    print(f"Put: ${euro_put:.2f}")
    print("\nGreeks:")
    for greek, value in greeks.items():
        print(f"{greek}: {value:.4f}")
    
    # American Option Example
    amer_option = AmericanOption(S0, K, T, r, sigma)
    amer_call = amer_option.binomial_tree_price(steps=100, option_type='call')
    amer_put = amer_option.binomial_tree_price(steps=100, option_type='put')
    
    print("\nAmerican Option Prices:")
    print(f"Call: ${amer_call:.2f}")
    print(f"Put: ${amer_put:.2f}")