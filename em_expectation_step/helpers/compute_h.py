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

    # print("s")
    # print(s)
    for i in range(n):
        s[i] = np.matmul(s[i], np.diag(p_top_gating_multinomial[i]))

    # print("s")
    # print(s)

    # (n, N)
    s_col_sum = np.sum(s, axis=1)

    # print("s_col_sum")
    # print(s_col_sum)

    # (1, N)
    s_sum = np.sum(s_col_sum, axis=0)

    # print("s_sum")
    # print(s_sum)

    h_top = np.matmul(np.array(s_col_sum), np.diag(np.reciprocal(s_sum)))

    # print("h_top")
    # print(h_top)

    h_lower_cond_top = np.array(s)

    for i in range(n):
        h_lower_cond_top[i] = np.matmul(
            np.array(h_lower_cond_top[i]), np.diag(np.reciprocal(s_col_sum[i]))
        )

    # print("h_lower_cond_top")
    # print(h_lower_cond_top)

    h_top_lower = np.array(h_lower_cond_top)
    for i in range(n):
        h_top_lower[i] = np.matmul(h_top_lower[i], np.diag(h_top[i]))

    # print("h_top_lower")
    # print(h_top_lower)
    return h_top, h_lower_cond_top, h_top_lower


p_expert_gaussian = np.array(
    [
        [[0.1, 0.2, 0.1, 0.4], [0.3, 0.2, 0.1, 0.5]],
        [[0.5, 0.3, 0.2, 0.4], [0.3, 0.2, 0.1, 0.5]],
    ]
)

p_top_gating_multinomial = np.array([[0.2, 0.7, 0.5, 0.6], [0.8, 0.3, 0.5, 0.4]])
p_lower_gating_multinomial = np.array(
    [
        [[0.1, 0.3, 0.4, 0.5], [0.9, 0.7, 0.6, 0.5]],
        [[0.5, 0.4, 0.3, 0.4], [0.5, 0.6, 0.7, 0.6]],
    ]
)

compute_h(p_expert_gaussian, p_top_gating_multinomial, p_lower_gating_multinomial)
