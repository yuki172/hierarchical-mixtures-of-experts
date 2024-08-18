import numpy as np
from weighted_maximum_likelihood_gaussian.main import (
    weighted_maximum_likelihood_gaussian,
)
from iteratively_reweighted_least_squares_multinomial_with_weights.main import (
    iteratively_reweighted_least_squares_multinomial_with_weights,
)

from log_font_colors import printColored


def compute_maximum_likelihood_estimates(
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

    printColored("em - maximization step")

    n, m, N = h_lower_cond_top.shape
    _, p = X.shape

    ### beta_expert, sigma_sq_expert ###

    # (n, m, p)
    beta_expert = np.zeros((n, m, p))

    # (n, m)
    sigma_sq_expert = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            c_ij = h_top_lower[i][j]
            beta_expert_ij, sigma_sq_expert_ij = weighted_maximum_likelihood_gaussian(
                X, Y, c_ij
            )
            beta_expert[i][j] = beta_expert_ij
            sigma_sq_expert[i][j] = sigma_sq_expert_ij

    printColored("beta_expert")
    print(beta_expert)

    printColored("sigma_sq_expert")
    print(sigma_sq_expert)

    ### beta_top ###
    top_gating_multinomial_outputs = h_top.T[:, :-1]
    top_gating_observation_weights = np.ones((N, 1))

    printColored("top_gating_multinomial_outputs")
    print(top_gating_multinomial_outputs)

    printColored("top_gating_observation_weights")
    print(top_gating_observation_weights)

    # (n - 1, p)
    beta_top = iteratively_reweighted_least_squares_multinomial_with_weights(
        X, Y=top_gating_multinomial_outputs, c=top_gating_observation_weights
    )

    printColored("beta_top")
    print(beta_top)

    ### beta_lower ###

    # (n, m - 1, p)
    beta_lower = np.zeros((n, m - 1, p))

    for i in range(n):
        printColored(f"---- {i} -----")
        multinomial_outputs = h_lower_cond_top[i].T[:, :-1]
        observation_weights = h_top[i]

        printColored("mulinomial_outputs")
        print(multinomial_outputs)

        printColored("observation_weights")
        print(observation_weights)

        beta_lower[i] = iteratively_reweighted_least_squares_multinomial_with_weights(
            X, Y=multinomial_outputs, c=observation_weights
        )

        printColored(f"beta_lower[i], {i}")
        print(beta_lower[i])

    printColored("beta_lower")
    print(beta_lower)

    return beta_expert, sigma_sq_expert, beta_top, beta_lower


# N = 4, p = 2
X = np.array([[1, 2], [1, 4], [1, 5], [1, 3]])
Y = np.array([[3], [4], [2], [5]])

# n = 3, m = 2

h_top = np.array(
    [
        [4.29045635e-01, 3.40697007e-01, 2.13941849e-05, 6.26759950e-01],
        [5.70828401e-01, 6.59220172e-01, 5.71323628e-01, 3.73231268e-01],
        [1.25964074e-04, 8.28204043e-05, 4.28654978e-01, 8.78178554e-06],
    ]
)
h_lower_cond_top = np.array(
    [
        [
            [9.99990803e-01, 1.00000000e00, 1.00000000e00, 9.99999971e-01],
            [9.19746378e-06, 5.95628834e-12, 1.14983415e-14, 2.92737795e-08],
        ],
        [
            [9.96933133e-01, 6.86547416e-01, 1.75186505e-01, 9.69251826e-01],
            [3.06686697e-03, 3.13452584e-01, 8.24813495e-01, 3.07481738e-02],
        ],
        [
            [9.15336004e-01, 4.08686199e-01, 2.19494875e-03, 9.58137497e-01],
            [8.46639961e-02, 5.91313801e-01, 9.97805051e-01, 4.18625031e-02],
        ],
    ]
)
h_top_lower = np.array(
    [
        [
            [4.29041689e-01, 3.40697007e-01, 2.13941849e-05, 6.26759932e-01],
            [3.94613169e-06, 2.02928961e-12, 2.45997643e-19, 1.83476326e-08],
        ],
        [
            [5.69077746e-01, 4.52585906e-01, 1.00088190e-01, 3.61755088e-01],
            [1.75065477e-03, 2.06634267e-01, 4.71235439e-01, 1.14761799e-02],
        ],
        [
            [1.15299452e-04, 3.38475562e-05, 9.40875707e-04, 8.41415801e-06],
            [1.06646219e-05, 4.89728481e-05, 4.27714102e-01, 3.67627525e-07],
        ],
    ]
)

compute_maximum_likelihood_estimates(
    X, Y, h_top=h_top, h_lower_cond_top=h_lower_cond_top, h_top_lower=h_top_lower
)
