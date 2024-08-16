import numpy as np
from em_expectation_step.main import compute_posterior_probabilities
from em_maximization_step.main import compute_maximum_likelihood_estimates


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


max_diff = 1 / 1000


def compute_coeff_sq_diff(
    beta_expert_curr,
    beta_expert_new,
    sigma_sq_expert_curr,
    sigma_sq_expert_new,
    beta_top_curr,
    beta_top_new,
    beta_lower_curr,
    beta_lower_new,
):
    sq_diff = 0
    for coeff_curr, coeff_new in [
        [beta_expert_curr, beta_expert_new],
        [sigma_sq_expert_curr, sigma_sq_expert_new],
        [beta_top_curr, beta_top_new],
        [beta_lower_curr, beta_lower_new],
    ]:
        sq_diff += np.sum(np.square(np.subtract(coeff_curr, coeff_new)))

    return sq_diff


def hierarchical_mixture_of_experts(X: np.ndarray, Y: np.ndarray, n: int, m: int):
    """
    N: number of observations in the sample \n
    p: number of input features, including intercept

    n: number of classes in the multinomial distribution at the top gating node \n
    m: number of classes in the multinomial distribution at the lower gating nodes \n

    X: feature matrix, of shape (N, p), first column is 1 \n
    Y: output vector, of shape (N, 1)
    """

    N, p = X.shape

    beta_expert_curr, sigma_sq_expert_curr, beta_top_curr, beta_lower_curr = (
        initialize_parameters(p, n, m)
    )

    max_iter_count = 100
    iter_count = 0

    while True:
        iter_count += 1

        # EM algorithm - expectation step
        h_top, h_lower_cond_top, h_top_lower = compute_posterior_probabilities(
            X,
            Y,
            beta_expert=beta_expert_curr,
            sigma_sq_expert=sigma_sq_expert_curr,
            beta_top=beta_top_curr,
            beta_lower=beta_lower_curr,
        )

        # EM algorithm - maximization step
        beta_expert_new, sigma_sq_expert_new, beta_top_new, beta_lower_new = (
            compute_maximum_likelihood_estimates(
                X, Y, h_top, h_lower_cond_top, h_top_lower
            )
        )

        coeff_sq_diff = compute_coeff_sq_diff(
            beta_expert_curr,
            beta_expert_new,
            sigma_sq_expert_curr,
            sigma_sq_expert_new,
            beta_top_curr,
            beta_top_new,
            beta_lower_curr,
            beta_lower_new,
        )

        if iter_count % 10 == 0:
            print("HME main loop", f"iteration count {iter_count}")
            print("sq_diff", coeff_sq_diff)

        beta_expert_curr, sigma_sq_expert_curr, beta_top_curr, beta_lower_curr = (
            beta_expert_new,
            sigma_sq_expert_new,
            beta_top_new,
            beta_lower_new,
        )

        if coeff_sq_diff <= max_diff:
            break

        if iter_count == max_iter_count:
            print("HME main loop", "max_iter_count reached", "stopping")
            break
