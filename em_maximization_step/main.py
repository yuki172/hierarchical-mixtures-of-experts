import numpy as np
from weighted_maximum_likelihood_gaussian.main import (
    weighted_maximum_likelihood_gaussian,
)
from iteratively_reweighted_least_squares_multinomial_with_weights.main import (
    iteratively_reweighted_least_squares_multinomial_with_weights,
)


def compute_next_iter_parameters(
    X: np.ndarray,
    Y: np.ndarray,
    h_top: np.ndarray,
    h_lower_cond_top: np.ndarray,
    h_top_lower: np.ndarray,
):
    """
    N: number of observations in the sample \n
    p: number of input features\n
    n: number of classes in the top level gating node multinomial distribution \n
    m: number of classes in the lower level gating node multinomial distribution \n

    X: feature matrix, of shape (N, p), first column is 1 \n
    Y: output vector, of shape (N, 1) \n

    h_top, elements are h_i, conditional probabilities at the top gating node given the observed data, of shape (n, N) \n
    h_lower_cond_top, elements are h_j|i, conditional probabilities are the lower gating nodes, given the top gating node and observed data, of shape (n, m, N) \n
    h_top_lower, elements are h_i_j, conditional probabilities of the joint distribution of the top and lower gating nodes, given the observed data, of shape (n, m, N) \n

    beta_expert: coefficients of x forming the mean of the normal distribution at the expert node, of shape (n, m, p) \n
    sigma_sq_expert: variance of the normal distribution at the expert node, of shape (n, m) \n

    beta_top: coefficients of the multinomial class probabilities (n classes) at the top gating node, of shape (n - 1, p) \n
    beta_lower: coefficients of the multinomial class probabilities (m classes) at the lower gating nodes (n, m - 1, p) \n
    """

    n, m, N = h_lower_cond_top.shape
    _, p = X.shape

    ### beta_expert, sigma_sq_expert ###

    # (n, m, p)
    beta_expert = np.array(
        [[[0 for _ in range(N)] for _ in range(m)] for _ in range(n)]
    )

    # (n, m)
    sigma_sq_expert = np.array([[0 for _ in range(m)] for _ in range(n)])

    for i in range(n):
        for j in range(m):
            c_ij = h_top_lower[i][j]
            beta_expert_ij, sigma_sq_expert_ij = weighted_maximum_likelihood_gaussian(
                X, Y, c_ij
            )
            beta_expert[i][j] = beta_expert_ij
            sigma_sq_expert[i][j] = sigma_sq_expert_ij

    ### beta_top ###
    top_gating_multinomial_outputs = h_top.T[:, :-1]
    top_gating_observation_weights = np.array([1 for _ in range(N)]).reshape((N, 1))

    # (n - 1, p)

    beta_top = iteratively_reweighted_least_squares_multinomial_with_weights(
        X, Y=h_top.T[:, :-1], c=top_gating_observation_weights
    )

    ### beta_lower ###

    beta_lower = np.array(
        [[[0 for _ in range(p)] for _ in range(m - 1)] for _ in range(n)]
    )

    for i in range(n):
        multinomial_outputs = h_lower_cond_top[i].T[:, :-1]
        observation_weights = h_top[i]
        beta_lower[i] = iteratively_reweighted_least_squares_multinomial_with_weights(
            X, Y=multinomial_outputs, c=observation_weights
        )

    return beta_expert, sigma_sq_expert, beta_top, beta_lower
