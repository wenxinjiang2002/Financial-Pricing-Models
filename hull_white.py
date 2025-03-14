"""
Hull-White One-Factor Interest Rate Model

Created on: Feb 17
Author: Wenxin.J

The Hull-White model is defined by the SDE:
dr(t) = [θ(t) - α*r(t)]dt + σdW(t)

where:
- r(t) is the short rate
- θ(t) is the drift term (fitted to initial term structure)
- α is the mean reversion speed
- σ is the volatility
- W(t) is a Wiener process

Calibration with Swaption:
Using the Hull-White model, the expected payoff of the swaption is computed by:
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

class HullWhite:
    def __init__(self, alpha, sigma, T, steps, r0=None):
        """
        Initialize Hull-White model parameters
        
        Parameters:
        alpha : float
            Mean reversion speed
        sigma : float
            Volatility of short rate
        T : float
            Time horizon
        steps : int
            Number of time steps
        r0 : float, optional
            Initial short rate (if None, will use forward rate)
        """
        self.alpha = alpha
        self.sigma = sigma
        self.T = T
        self.steps = steps
        self.dt = T/steps
        self.r0 = r0
        
    def simulate_rates(self, n_paths, seed=None):
        """
        Simulate interest rate paths
        
        Parameters:
        n_paths : int
            Number of simulation paths
        seed : int, optional
            Random seed for reproducibility
        
        Returns:
        rates : ndarray
            Matrix of simulated rates (n_paths × steps)
        times : ndarray
            Time points
        """
        if seed is not None:
            np.random.seed(seed)
            
        times = np.linspace(0, self.T, self.steps + 1)
        rates = np.zeros((n_paths, self.steps + 1))
        
        # Set initial rates
        rates[:, 0] = self.r0 if self.r0 is not None else 0
        
        # Simulate paths
        for t in range(self.steps):
            drift = -self.alpha * rates[:, t] * self.dt
            diffusion = self.sigma * np.sqrt(self.dt) * np.random.normal(0, 1, n_paths)
            rates[:, t+1] = rates[:, t] + drift + diffusion
            
        return rates, times
    
    def zero_coupon_bond_price(self, r, t, T):
        """
        Calculate zero-coupon bond price
        
        Parameters:
        r : float
            Current short rate
        t : float
            Current time
        T : float
            Maturity time
        
        Returns:
        price : float
            Bond price
        """
        tau = T - t
        B = (1 - np.exp(-self.alpha * tau)) / self.alpha
        A = (self.sigma * B) ** 2 * (1 - np.exp(-2 * self.alpha * t)) / (4 * self.alpha)
        
        return np.exp(-r * B - A)
    
    def bond_option_price(self, K, t, T, S, option_type='call'):
        """
        Price European options on zero-coupon bonds
        
        Parameters:
        K : float
            Strike price
        t : float
            Time to option expiry
        T : float
            Bond maturity
        S : float
            Current bond price
        option_type : str
            'call' or 'put'
            
        Returns:
        price : float
            Option price
        """
        tau = T - t
        sigma_p = (self.sigma/self.alpha) * np.sqrt(
            (1 - np.exp(-2*self.alpha*t))/2/self.alpha
        ) * (1 - np.exp(-self.alpha*tau))
        
        d1 = (np.log(S/K) + sigma_p**2/2) / sigma_p
        d2 = d1 - sigma_p
        
        if option_type.lower() == 'call':
            return S * norm.cdf(d1) - K * norm.cdf(d2)
        else:  # put
            return K * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    def plot_simulations(self, rates, times, n_paths_to_plot=5):
        """
        Plot simulated interest rate paths
        
        Parameters:
        rates : ndarray
            Simulated rates
        times : ndarray
            Time points
        n_paths_to_plot : int
            Number of paths to display
        """
        plt.figure(figsize=(10, 6))
        for i in range(min(n_paths_to_plot, rates.shape[0])):
            plt.plot(times, rates[i, :], label=f'Path {i+1}')
            
        plt.title('Hull-White Interest Rate Simulations')
        plt.xlabel('Time')
        plt.ylabel('Interest Rate')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def calculate_moments(self, rates):
        """
        Calculate statistical moments of simulated rates
        
        Parameters:
        rates : ndarray
            Simulated rates
            
        Returns:
        dict
            Statistical moments
        """
        return {
            'mean': np.mean(rates, axis=0),
            'std': np.std(rates, axis=0),
            'skew': np.mean(((rates - np.mean(rates, axis=0)) / 
                           np.std(rates, axis=0))**3, axis=0),
            'kurt': np.mean(((rates - np.mean(rates, axis=0)) / 
                           np.std(rates, axis=0))**4, axis=0) - 3
        }

# Example usage
if __name__ == "__main__":
    # Model parameters
    alpha = 0.1    # Mean reversion speed
    sigma = 0.02   # Volatility
    T = 10.0       # Time horizon
    steps = 250    # Number of time steps
    r0 = 0.02      # Initial short rate
    
    # Create model instance
    hw_model = HullWhite(alpha, sigma, T, steps, r0)
    
    # Simulate paths
    rates, times = hw_model.simulate_rates(n_paths=1000, seed=42)
    
    # Plot some paths
    hw_model.plot_simulations(rates, times)
    
    # Calculate and print statistics
    moments = hw_model.calculate_moments(rates)
    print("\nRate Statistics at T:")
    for stat, value in moments.items():
        print(f"{stat}: {value[-1]:.4f}")
    
    # Price a zero-coupon bond
    bond_price = hw_model.zero_coupon_bond_price(r0, 0, 5)
    print(f"\nZero-coupon bond price (T=5): {bond_price:.4f}")
    
    # Price bond options
    call_price = hw_model.bond_option_price(K=0.95, t=1, T=5, S=bond_price)
    put_price = hw_model.bond_option_price(K=0.95, t=1, T=5, S=bond_price, 
                                         option_type='put')
    print(f"Bond call option price: {call_price:.4f}")
    print(f"Bond put option price: {put_price:.4f}")
