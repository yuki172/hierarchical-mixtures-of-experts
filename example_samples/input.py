import numpy as np


def sample_input(N: int, p: int):
    """
    N: number of observations \\
    p: number of feature variables \\

    X: input matrix, of shape (N, p - 1) \\
    X_full: input matrix prepended with a column of 1, of shape (N, p)
    """

    # (N, p - 1)
    X = np.zeros((N, p - 1))

    for i in range(N):
        X[i] = np.random.uniform(0, 1, p - 1)

    # (N, p)
    X_full = np.concatenate([np.ones([N, 1]), X], axis=1)

    return X, X_full
