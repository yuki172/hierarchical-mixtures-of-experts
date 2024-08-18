import numpy as np


def initialize_parameters(p: int, n: int, m: int):
    """
    beta_expert: coefficients of x forming the mean of the normal distribution at the expert node, of shape (n, m, p) \n
    sigma_sq_expert: variance of the normal distribution at the expert node, of shape (n, m) \n

    beta_top: coefficients of the multinomial class probabilities (n classes) at the top gating node, of shape (n - 1, p) \n
    beta_lower: coefficients of the multinomial class probabilities (m classes) at the lower gating nodes (n, m - 1, p) \n
    """
    # initialize all expert distributions to be normal with mean 0 and variance 1
    beta_expert = np.zeros((n, m, p))
    sigma_sq_expert = np.ones((n, m))

    # initialize the top gating node to be multinomial with equal class probabilities
    beta_top = np.zeros((n - 1, p))

    # initialize the lower gating nodes to be multinomial with equal class probabilities
    beta_lower = np.zeros((n, m - 1, p))

    return beta_expert, sigma_sq_expert, beta_top, beta_lower
