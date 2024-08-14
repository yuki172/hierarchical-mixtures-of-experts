import numpy as np


def compute_e(Y: np.ndarray, p_prob: np.ndarray) -> np.ndarray:
    """
    N: number of observations in the sample \n
    n: number of classes in the multinomial distribution \n
    Y: output matrix, of shape (N, n - 1), y_n is omitted \n
    c: observation weights, of shape (N, 1)
    p_prob: result of compute_p_prob, the class probabilities, of shape (N, n) \n

    e: of shape (N * (n - 1), 1)
    """

    N, n_1 = Y.shape
    n = n_1 + 1

    e_rows = [np.array(1) for _ in range(n - 1)]

    for i in range(n - 1):
        y_i = Y[:, i]
        p_prob_i = p_prob[:, i]
        e_rows_i = y_i - p_prob_i
        e_rows[i] = e_rows_i
        # print(i, e_rows_i)

    e_row = e_rows[0]
    for i in range(1, n - 1):
        e_row = np.concatenate((e_row, e_rows[i]), axis=0)

    e = np.reshape(e_row, (-1, 1))
    # print(e)
    return e
