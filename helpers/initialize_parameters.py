import numpy as np
from random import random


def initialize_parameters(p: int, n: int, m: int):
    """
    beta_expert: coefficients of x forming the mean of the normal distribution at the expert node, of shape (n, m, p) \n
    sigma_sq_expert: variance of the normal distribution at the expert node, of shape (n, m) \n

    beta_top: coefficients of the multinomial class probabilities (n classes) at the top gating node, of shape (n - 1, p) \n
    beta_lower: coefficients of the multinomial class probabilities (m classes) at the lower gating nodes (n, m - 1, p) \n
    """
    # initialize all expert distributions to be normal with variance 1
    sigma_sq_expert = np.ones((n, m))

    # the mean will be different for distinct experts
    beta_expert = np.zeros((n, m, p))
    # beta_expert = np.random.randint(1, 10, size=(n, m, p))
    # beta_expert = np.random.rand(n, m, p)
    # for i in range(n):
    #     for j in range(m):
    #         beta_expert[i][j][0] = (i + j) % 2 + random()

    # initialize the top gating node to be multinomial with equal class probabilities
    beta_top = np.zeros((n - 1, p))
    # beta_top = np.random.rand(n - 1, p)

    # initialize the lower gating nodes to be multinomial with equal class probabilities
    beta_lower = np.zeros((n, m - 1, p))
    # beta_lower = np.random.rand(n, m - 1, p)

    return beta_expert, sigma_sq_expert, beta_top, beta_lower
