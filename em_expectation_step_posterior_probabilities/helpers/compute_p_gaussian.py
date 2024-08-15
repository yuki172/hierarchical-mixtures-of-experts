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

    Y = Y.T
    N, p = X.shape
    n, m, _ = beta.shape

    p_gaussian = np.array([[[0] * N for _ in range(m)] for _ in range(n)])

    for i in range(n):
        for j in range(m):
            beta_ij = beta[i][j]
            mu_ij = np.matmul(beta_ij, X.T)
            sigma_sq_ij = sigma_sq[i][j]

            coef = 1 / (sqrt(2 * pi) * sigma_sq_ij)
            exponent = -np.square(Y - mu_ij) / (2 * sigma_sq_ij)

            p_ij = coef * np.exp(exponent)

            p_gaussian[i][j] = p_ij

    return p_gaussian
