import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Parameters from part (1)(b)
E = 10          # Strike price
r = 0.25        # Risk-free rate
q = 0.2         # Dividend yield
sigma = 0.8     # Volatility
T = 1.0         # Maturity
S_values = [10, 20, 25]  # Prices to evaluate

# Binomial tree setup
N = 200                     # Number of time steps
dt = T / N
nu = r - q                  # Drift adjusted for dividend
u = np.exp(sigma * np.sqrt(dt))    # Up factor
d = 1 / u                           # Down factor
p = (np.exp(nu * dt) - d) / (u - d) # Risk-neutral probability

# Display binomial parameters
print(f"U = {u:.4f}, D = {d:.4f}, P = {p:.4f}")

# Binomial pricing function for American call
def binomial_american_call(S0):
    # Initialize prices at maturity
    ST = np.array([S0 * (u ** j) * (d ** (N - j)) for j in range(N + 1)])
    values = np.maximum(ST - E, 0)

    # Backward induction
    for i in range(N - 1, -1, -1):
        values = np.exp(-r * dt) * (p * values[1:i+2] + (1 - p) * values[0:i+1])
        S_nodes = np.array([S0 * (u ** j) * (d ** (i - j)) for j in range(i + 1)])
        values = np.maximum(values, S_nodes - E)  # Early exercise check
    return values[0]

# Compute binomial prices
binomial_results = []
for S0 in S_values:
    price = binomial_american_call(S0)
    binomial_results.append((S0, price))



# Interpolate finite difference (FD) results
euro_interp = interp1d(S, V_eur_call, kind='linear')
amer_interp = interp1d(S, V_am_call, kind='linear')

# Compare results
comparison = []
for S0, bino_price in binomial_results:
    euro_val = euro_interp(S0)
    amer_val = amer_interp(S0)
    comparison.append({
        "S": S0,
        "Binomial American Call": bino_price,
        "FD European Call": euro_val,
        "FD American Call": amer_val
    })

comparison_df = pd.DataFrame(comparison)
print(comparison_df)



