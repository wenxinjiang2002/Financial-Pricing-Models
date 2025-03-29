import numpy as np
import scipy.linalg as linalg
import scipy.interpolate as interp
import pandas as pd
import matplotlib.pyplot as plt

# Parameters
E = 10.0        # Strike price
r = 0.1         # Risk-free rate
sigma = 0.4     # Volatility
T = 1.0 / 3.0   # Time to maturity: 4 months

S_max = 20      # Max stock price
M = 200         # Number of spatial steps
N = 100         # Number of time steps

dS = S_max / M
dt = T / N

S = np.linspace(0, S_max, M + 1)

# Initial conditions
V_eur = np.maximum(E - S, 0)
V_am = V_eur.copy()

def apply_boundary_conditions(V, tau):
    V[0] = E * np.exp(-r * (T - tau))
    V[-1] = 0.0
    return V

def build_tridiagonal_matrices():
    alpha = 0.5 * dt * ((sigma ** 2) * (S[1:M] ** 2) / dS ** 2 - (r * S[1:M]) / dS)
    beta = 1 + dt * ((sigma ** 2) * (S[1:M] ** 2) / dS ** 2 + r)
    gamma = 0.5 * dt * ((sigma ** 2) * (S[1:M] ** 2) / dS ** 2 + (r * S[1:M]) / dS)

    A = np.zeros((M - 1, M - 1))
    B = np.zeros((M - 1, M - 1))

    for j in range(M - 1):
        if j > 0:
            A[j, j - 1] = -alpha[j]
            B[j, j - 1] = alpha[j]
        A[j, j] = beta[j]
        B[j, j] = 2 - beta[j]
        if j < M - 2:
            A[j, j + 1] = -gamma[j]
            B[j, j + 1] = gamma[j]
    return A, B

A, B = build_tridiagonal_matrices()

# ---- European Put Solver ----
def solve_european_put():
    V = np.maximum(E - S, 0)

    lower = np.diag(A, k=-1)
    main = np.diag(A)
    upper = np.diag(A, k=1)

    ab = np.zeros((3, M - 1))
    ab[0, 1:] = upper
    ab[1, :] = main
    ab[2, :-1] = lower

    for n in range(N):
        tau = (n + 1) * dt
        rhs = B @ V[1:M]
        rhs[0] += A[0, 0] * V[0]
        rhs[-1] += A[-1, -1] * V[-1]
        V[1:M] = linalg.solve_banded((1, 1), ab, rhs)
        V = apply_boundary_conditions(V, tau)
    return V

V_eur = solve_european_put()

# ---- American Put with Projected SOR ----
def solve_american_put():
    V = np.maximum(E - S, 0)
    omega = 1.2
    tol = 1e-6
    max_iter = 10000

    for n in range(N):
        tau = (n + 1) * dt
        rhs = B @ V[1:M]
        rhs[0] += A[0, 0] * V[0]
        rhs[-1] += A[-1, -1] * V[-1]
        V_old = V[1:M].copy()
        for _ in range(max_iter):
            V_new = V_old.copy()
            for j in range(M - 1):
                sum_ = 0
                if j > 0:
                    sum_ += A[j, j - 1] * V_new[j - 1]
                if j < M - 2:
                    sum_ += A[j, j + 1] * V_old[j + 1]
                sum_ = (rhs[j] - sum_) / A[j, j]
                V_new[j] = max(E - S[j + 1], V_old[j] + omega * (sum_ - V_old[j]))
            error = np.linalg.norm(V_new - V_old, ord=np.inf)
            V_old = V_new
            if error < tol:
                break
        V[1:M] = V_old
        V = apply_boundary_conditions(V, tau)
    return V

V_am = solve_american_put()

# ---- Interpolation at Requested Points ----
query_S = np.arange(0, 17, 2)
interp_eur = interp.interp1d(S, V_eur, kind='linear')
interp_am = interp.interp1d(S, V_am, kind='linear')

results_df = pd.DataFrame({
    "S": query_S,
    "European Put": interp_eur(query_S),
    "American Put": interp_am(query_S)
})

print(results_df)


