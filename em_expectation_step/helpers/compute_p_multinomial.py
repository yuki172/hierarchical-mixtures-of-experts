import numpy as np


def compute_p_multinomial(X, beta) -> np.ndarray:
    """
    N: number of observations in the sample \n
    p: number of input features
    n: number of classes in the multinomial distribution \n

    X: feature matrix, of shape (N, p), first column is 1 \n
    beta: coefficients, of shape (n - 1, p)\n

    p_prob: class probabilities in the multinomial distribution, of shape (n, N)
    """

    # (N, n - 1)
    eta = np.matmul(X, beta.T)

    N, _n = eta.shape

    # (N, n)
    exp_eta = np.concatenate((np.exp(eta), np.ones((N, 1))), axis=1)

    # (N, 1)
    denominator = np.reciprocal(np.matmul(exp_eta, np.ones((_n + 1, 1))))

    denominator = denominator.flatten()

    # (N, n)
    p_prob = np.matmul(np.diag(denominator), exp_eta)

    return p_prob.T
