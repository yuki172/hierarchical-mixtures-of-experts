import numpy as np


def compute_eta(X, beta_curr) -> np.ndarray:
    """
    X: feature matrix, of shape (N, p), first column is 1 \n
    beta_curr: coefficients, of shape (n - 1, p)\n

    eta: of shape (N, n - 1)
    """
    return np.matmul(X, beta_curr.T)
