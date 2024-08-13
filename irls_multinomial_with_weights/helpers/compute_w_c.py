import numpy as np
from typing import Any
from log_font_colors import log_font_colors


def compute_W_c(p_prob: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    N: number of observations in the sample \n
    n: number of classes in the multinomial distribution \n
    p_prob: result of compute_p_prob, the class probabilities, of shape (N, n) \n
    c: observation weights, of shape (N, 1)
    """

    N, n = p_prob.shape
    # remove p_n, new shape (N, n - 1)
    p_prob = np.delete(p_prob, -1, axis=1)

    # (i, j) th element is the matrix w_c_ij
    w_c_dict: list[list[np.ndarray]] = [
        [np.array(1) for _ in range(n - 1)] for _ in range(n - 1)
    ]

    c = c.flatten()
    for i in range(len(w_c_dict)):
        w_c_i = None
        for j in range(len(w_c_dict[i])):
            p_prob_i = p_prob[:, i]
            p_prob_j = p_prob[:, j]
            w_c_ij = np.array(1)
            if i == j:
                w_c_ij = np.multiply(c, np.multiply(p_prob_i, 1 - p_prob_j))
            else:
                w_c_ij = np.multiply(-c, np.multiply(p_prob_i, p_prob_j))
            w_c_ij = np.diag(w_c_ij)
            w_c_dict[i][j] = w_c_ij

    w_c_dict_rows: list[np.ndarray] = [np.array(1) for _ in range(n - 1)]
    for i in range(n - 1):
        w_c_i = w_c_dict[i][0]
        for j in range(1, n - 1):
            w_c_i = np.concat((w_c_i, w_c_dict[i][j]), axis=1)
        w_c_dict_rows[i] = w_c_i

    w_c = w_c_dict_rows[0]

    for i in range(1, n - 1):
        w_c = np.concatenate((w_c, w_c_dict_rows[i]), axis=0)

    # print(f"{log_font_colors.OKGREEN}w_c{log_font_colors.ENDC}")
    # print(w_c)

    return w_c
