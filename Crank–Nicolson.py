"""
Implementation of the Crank–Nicolson method for the untransformed Black–Scholes PDE for a European call.
Aim to compare the numerical results against the exact Black–Scholes formula at a chosen time t<T. 
"""

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm

# Parameters setup
T = 1.0
SIGMA = 0.2
R = 0.05
K = 100
S_INF = 10 * K # right boundary condition
S_MAX = 2 * K # maximum stock price
N = 1000 # number of time steps in tau
d_tau = T / N

def crank_nicolson_matrices_cal(m):
"""
Construct the Crank-Nicolson matrices for the interior nodes.
param m: number of spatial steps
return: a_matrix, b_matrix matrices
"""
    d_s = S_INF / m
    a = np.zeros(m-1)
    b = np.zeros(m-1)
    c = np.zeros(m-1)
    for j in range(1, m):
        s_j = j * d_s
        a[j-1] = (- SIGMA**2 * s_j / (d_s) + R) * (s_j * d_tau / (4*d_s))
        b[j-1] = (SIGMA**2 * s_j**2 / (d_s**2) + R) * (d_tau / 2)
        c[j-1] = (- SIGMA**2 * s_j / (d_s) - R) * (s_j * d_tau / (4*d_s))
    a_matrix = np.zeros((m-1, m-1))
    b_matrix = np.zeros((m-1, m-1))
    for i in range(m-1):
        # Main diagonal
        a_matrix[i, i] = 1 + b[i]
        b_matrix[i, i] = 1 - b[i]
# Lower diagonal (for i > 0)
        if i > 0:
            a_matrix[i, i-1] = a[i]
            b_matrix[i, i-1] = - a[i]
# Upper diagonal (for i < M-2)
        if i < m-2:
            a_matrix[i, i+1] = c[i]
            b_matrix[i, i+1] = - c[i]
    return a_matrix, b_matrix

def gauss_seidel(a_matrix, d, x0, x_right_boundary, x_left_boundary=0, tol=1e-8, max_iter=10000):
"""
Solve the linear system Ax = d using the Gauss-Seidel method
:param a_matrix: Coefficient matrix
:param d: Right-hand side vector
:param x0: Initial guess for the solution
:param x_right_boundary: Right boundary condition
:param x_left_boundary: Left boundary condition
:param tol: Tolerance for convergence
:param max_iter: Maximum number of iterations
:return: Solution vector x
"""
    n = len(d)
    x = x0.copy()
    for _ in range(max_iter):
        x_old = x.copy()
        x[0] = (d[0] - a_matrix[1, 0] * x_right_boundary - a_matrix[0, 1] * x_old[1]) / a_matrix[0, 0]
        for i in range(n-1):
            x[i] = (d[i] - a_matrix[i, i-1] * x[i-1] - a_matrix[i, i+1] * x_old[i+1]) / a_matrix[i, i]
        x[-1] = (d[-1] - a_matrix[-1,-2] * x[-2] - a_matrix[-1, -1] * x_right_boundary) / a_matrix[-1,-1]
        if np.linalg.norm(x - x_old, ord=np.inf) < tol:
            return x
    print("Gauss-Seidel did not converge in {} iterations".format(max_iter))
    return x



def black_scholes_numerical(m, method='LU'):
    """
    使用 Crank-Nicolson 方法求解 Black-Scholes PDE
    :param m: 空间步数
    :param method: 使用 'LU' 分解或 'Gauss-Seidel' 方法
    :return: 结果解数组
    """
    # 初始化网格和初始解（期权到期时的支付函数）
    s_grid = np.linspace(0, S_INF, m+1)
    v_s0 = np.maximum(s_grid - K, 0)  # V(S, tau=0) = 到期时的支付函数
    v_result = v_s0.copy()  # 全部节点，包含边界

    # 构造 Crank-Nicolson 矩阵
    A, B = crank_nicolson_matrices_cal(m)

    # 若采用 LU 分解，预先分解 A 矩阵
    if method == 'LU':
        lu, piv = la.lu_factor(A)

    # 时间步进：n = 0,...,N-1 对应 tau 从 0 到 T 的步进
    for n in tqdm(range(N), desc=f"{method} Method Time Steps with m={m}"):
        tau = (n+1) * d_tau  # 新的时间层 tau
        # 更新新时间层的边界条件（欧式看涨期权）
        v_max_new = S_INF - K * np.exp(-R * tau)

        # 提取内部节点解（索引 1 到 m-1）
        v_old = v_result[1:m]

        # 计算右端项：B * V_old
        d = B @ v_old

        # 考虑边界条件的调整
        d[-1] -= A[-2, -1] * v_max_new  # 来自 S = S_INF 边界的贡献

        # 求解内部节点的线性系统
        if method == 'LU':
            v_new = la.lu_solve((lu, piv), d)
        elif method == 'Gauss-Seidel':
            v_new = gauss_seidel(A, d, v_old, v_max_new)
        else:
            raise ValueError("未知的方法：请选择 'LU' 或 'Gauss-Seidel'")

        # 更新解，将内部节点和边界条件更新到全解中
        v_result[1:m] = v_new
        v_result[m] = v_max_new

    return v_result

def black_scholes_call(s, k, r, sigma, t):
    """
    计算欧式看涨期权的 Black-Scholes 精确解
    :param s: 标的资产价格
    :param k: 执行价
    :param r: 无风险利率
    :param sigma: 波动率
    :param t: 到期时间
    :return: 看涨期权价格
    """
    s = np.array(s, dtype=float)
    price = np.zeros_like(s)
    # 为避免 log(0)，只对 S > 0 进行计算
    pos = s > 0
    if np.any(pos):
        d1 = (np.log(s[pos] / k) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)
        price[pos] = s[pos] * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)
    return price



# 绘制结果
fig1, ax1 = plt.subplots(figsize=(10, 6))
fig2, ax2 = plt.subplots(figsize=(10, 6))

for i, m in enumerate([100, 200, 400, 800]):  # 测试不同的空间步数
    s_grid = np.linspace(0, S_INF, m+1)

    solution_lu = black_scholes_numerical(m, method='LU')
    solution_gs = black_scholes_numerical(m, method='Gauss-Seidel')
    exact_solution = black_scholes_call(s_grid, K, R, SIGMA, T)

    # 绘制数值解与精确解
    idx = int((S_MAX / S_INF) * m)
    if i == 0:
        ax1.plot(s_grid[:idx], exact_solution[:idx], 'k-', lw=2, label='Exact Black-Scholes')
        ax1.plot(s_grid[:idx], solution_lu[:idx], lw=2, label=f'Numerical (LU) m={m}')
    ax1.plot(s_grid[:idx], solution_gs[:idx], lw=2, label=f'Numerical (Gauss-Seidel) m={m}')

    # 绘制绝对误差
    error_lu = np.abs(solution_lu - exact_solution)
    error_gs = np.abs(solution_gs - exact_solution)
    ax2.plot(s_grid[:idx], error_lu[:idx], lw=2, label=f'LU m={m}')
    ax2.plot(s_grid[:idx], error_gs[:idx], lw=2, label=f'Gauss-Seidel m={m}')

ax1.set_xlabel("Asset Price S")
ax1.set_ylabel("Option Price V")
ax1.set_title("Numerical vs. Exact European Call Prices")
ax1.legend()
ax1.grid(True)

ax2.set_xlabel("Asset Price S")
ax2.set_ylabel("Absolute Error")
ax2.set_title("Absolute Error: Numerical vs. Exact")
ax2.legend()
ax2.grid(True)

plt.show()
