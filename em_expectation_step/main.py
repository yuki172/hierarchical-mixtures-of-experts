import numpy as np
from typing import Tuple
from math import sqrt, pi
from helpers.compute_p_gaussian import compute_p_gaussian
from helpers.compute_p_multinomial import compute_p_multinomial
from helpers.compute_h import compute_h


def compute_posterior_probabilities(
    X: np.ndarray,
    Y: np.ndarray,
    beta_expert: np.ndarray,
    sigma_sq_expert: np.ndarray,
    beta_top: np.ndarray,
    beta_lower: np.ndarray,
):
    """
    N: number of observations in the sample \n
    p: number of input features\n
    n: number of classes in the top level gating node multinomial distribution \n
    m: number of classes in the lower level gating node multinomial distribution \n

    X: feature matrix, of shape (N, p), first column is 1 \n
    Y: output vector, of shape (N, 1) \n
    beta_expert: coefficients of x forming the mean of the normal distribution at the expert node, of shape (n, m, p) \n
    sigma_sq_expert: variance of the normal distribution at the expert node, of shape (n, m) \n

    beta_top: coefficients of the multinomial class probabilities (n classes) at the top gating node, of shape (n - 1, p) \n
    beta_lower: coefficients of the multinomial class probabilities (m classes) at the lower gating nodes (n, m - 1, p) \n

    h_top, elements are h_i, conditional probabilities at the top gating node given the observed data, of shape (n, N) \n
    h_lower_cond_top, elements are h_j|i, conditional probabilities are the lower gating nodes, given the top gating node and observed data, of shape (n, m, N) \n
    h_top_lower, elements are h_i_j, conditional probabilities of the joint distribution of the top and lower gating nodes, given the observed data, of shape (n, m, N)
    """

    N, p = X.shape
    n_1, _ = beta_top.shape
    _, m_1, p = beta_lower.shape
    n = n_1 + 1
    m = m_1 + 1

    # (n, m, N)
    p_expert_gaussian = compute_p_gaussian(
        X, Y, beta=beta_expert, sigma_sq=sigma_sq_expert
    )
    # (n, N)
    p_top_gating_multinomial = compute_p_multinomial(X, beta=beta_top)

    # (n, m, N)
    p_lower_gating_multinomial = np.zeros((m, m, N))

    for i in range(n):
        p_lower_i = compute_p_multinomial(X, beta=beta_lower[i])
        p_lower_gating_multinomial[i] = p_lower_i

    h_top, h_lower_cond_top, h_top_lower = compute_h(
        p_expert_gaussian, p_top_gating_multinomial, p_lower_gating_multinomial
    )
    return h_top, h_lower_cond_top, h_top_lower
