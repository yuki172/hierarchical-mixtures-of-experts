import numpy as np


def compute_h(
    p_expert_gaussian: np.ndarray,
    p_top_gating_multinomial: np.ndarray,
    p_lower_gating_multinomial: np.ndarray,
):
    """
    p_expert_gaussian: values of the densities, of shape (n, m, N) \n
    p_top_gating_multinomial: class probabilities of the multinomial distribution at the top gating node, of shape (n, N) \n
    p_lower_gating_multinomial: class probabilities of the multinomial distributions at the lower gating node, of shape (n, m, N) \n

    h_top, elements are h_i, conditional probabilities at the top gating node given the observed data, of shape (n, N) \n
    h_lower_cond_top, elements are h_j|i, conditional probabilities are the lower gating nodes, given the top gating node and observed data, of shape (n, m, N) \n
    h_top_lower, elements are h_i_j, conditional probabilities of the joint distribution of the top and lower gating nodes, given the observed data, of shape (n, m, N)
    """

    n, m, N = p_expert_gaussian.shape

    # (n, m, N)
    s = np.multiply(p_expert_gaussian, p_lower_gating_multinomial)
    for i in range(n):
        s[i] = np.matmul(s[i], np.diag(p_top_gating_multinomial[i]))

    # (n, N)
    s_col_sum = np.sum(s, axis=1)

    # (1, N)
    s_sum = np.sum(s_col_sum, axis=0)

    h_top = np.matmul(np.array(s_col_sum), np.diag(np.reciprocal(s_sum)))

    h_lower_cond_top = np.array(s)

    for i in range(n):
        h_lower_cond_top[i] = np.matmul(
            np.array(h_lower_cond_top[i]), np.diag(np.reciprocal(s_col_sum[i]))
        )

    h_top_lower = np.array(h_lower_cond_top)
    for i in range(n):
        h_top_lower[i] = np.matmul(h_top_lower[i], np.diag(h_top[i]))

    return h_top, h_lower_cond_top, h_top_lower
