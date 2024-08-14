import numpy as np


def compute_posterior_h_top(
    p_expert_gaussian: np.ndarray,
    p_top_gating_multinomial: np.ndarray,
    p_lower_gating_multinomial: np.ndarray,
):
    """
    p_expert_gaussian: values of the densities, of shape (n, m, N) \n
    p_top_gating_multinomial: class probabilities of the multinomial distribution at the top gating node, of shape (n, N) \n
    p_lower_gating_multinomial: class probabilities of the multinomial distributions at the lower gating node, of shape (n, m, N) \n

    h_top: posterior probabilities at the top gating node, of shape (n, N)
    """

    n, m, N = p_expert_gaussian.shape

    observed_p = np.multiply(p_expert_gaussian, p_lower_gating_multinomial)
    for i in range(n):
        for j in range(m):
            observed_p[i][j] = np.multiply(
                observed_p[i][j], p_top_gating_multinomial[i]
            )
