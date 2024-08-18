import numpy as np
from typing import Tuple
from math import sqrt, pi


def compute_p_gaussian(
    X: np.ndarray, Y: np.ndarray, beta: np.ndarray, sigma_sq: np.ndarray
):
    """
    N: number of observations in the sample \n
    p: number of input features\n
    n: number of classes in the top level gating node multinomial distribution \n
    m: number of classes in the lower level gating node multinomial distribution \n

    X: feature matrix, of shape (N, p), first column is 1 \n
    Y: output vector, of shape (N, 1) \n
    beta: coefficients of x forming the mean of the normal distribution, of shape (n, m, p) \n
    sigma_sq: variance of normal distribution, of shape (n, m)

    p_gaussian: values of the densities, of shape (n, m, N)
    """

    # vector of length N
    Y = (Y.T)[0]

    N, p = X.shape
    n, m, _ = beta.shape

    p_gaussian = np.zeros((n, m, N))
    for i in range(n):
        for j in range(m):
            # print("---", i, j, "----")
            beta_ij = beta[i][j]
            mu_ij = np.matmul(beta_ij, X.T)
            sigma_sq_ij = sigma_sq[i][j]

            # print("beta_ij")
            # print(beta_ij)

            # print("mu_ij", mu_ij)
            # print("sigma_sq_ij", sigma_sq_ij)

            coef = 1 / sqrt(2 * pi * sigma_sq_ij)
            exponent = -np.square(Y - mu_ij) / (2 * sigma_sq_ij)

            # print("coef", coef)
            # print("exponent", exponent)

            p_ij = coef * np.exp(exponent)

            # print("p_ij", p_ij)
            p_gaussian[i][j] = p_ij

    # print("p_gaussian")
    # print(p_gaussian)

    return p_gaussian


# N = 4, p = 2
X = np.array([[1, 2], [1, 4], [1, 5], [1, 3]])
Y = np.array([[3], [4], [2], [5]])

# n = 2, m = 3

# (n, m, p)
beta_expert = np.array([[[1, 2], [2, 3], [5, 2]], [[2, 3], [4, 1], [3, 4]]])

# (n, m)
sigma_sq_expert = np.array([[1, 2, 1], [3, 1, 2]])


# p_gaussian = compute_p_gaussian(X, Y, beta=beta_expert, sigma_sq=sigma_sq_expert)
# print(np.sum(p_gaussian))
