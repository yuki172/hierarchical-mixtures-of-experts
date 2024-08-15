import numpy as np


def compute_X_tilde(X: np.ndarray, n: int):
    """
    N: number of observations in the sample \n
    p: number of input features
    n: number of classes in the multinomial distribution \n
    X: feature matrix, of shape (N, p), first column is 1 \n

    X_tilde: of shape ((n - 1) * N, (n - 1) * p), the (n - 1, n - 1) matrix with X on the diagonal
    """

    N, p = X.shape

    def get_ij_th_matrix(i, j):
        if i == j:
            return X
        else:
            return np.zeros((N, p))

    X_tilde_rows = [np.array(1) for _ in range(n - 1)]

    for i in range(n - 1):
        X_tilde_row_i = get_ij_th_matrix(i, 0)
        for j in range(1, n - 1):
            X_tilde_row_i = np.concatenate(
                (X_tilde_row_i, get_ij_th_matrix(i, j)), axis=1
            )
        X_tilde_rows[i] = X_tilde_row_i
        # print(i)
        # print(X_tilde_row_i)

    X_tilde = X_tilde_rows[0]
    for i in range(1, n - 1):
        X_tilde = np.concatenate((X_tilde, X_tilde_rows[i]), axis=0)

    # print(X_tilde)

    return X_tilde
