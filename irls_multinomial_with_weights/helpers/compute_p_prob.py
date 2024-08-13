import numpy as np
from helpers.compute_eta import compute_eta


def compute_p_prob(X, beta_curr) -> np.ndarray:
    """
    N: number of observations in the sample \n
    p: number of input features
    n: number of classes in the multinomial distribution \n
    X: feature matrix, of shape (N, p), first column is 1 \n
    beta_curr: coefficients, of shape (n - 1, p)\n

    p: class probabilities in the multinomial distribution, of shape (N, n)
    """

    eta = compute_eta(X, beta_curr)

    print("eta")
    print(eta)

    N, _n = eta.shape

    # (N, n)
    exp_eta = np.concatenate((np.exp(eta), np.ones((N, 1))), axis=1)

    # (N, 1)
    denominator = np.reciprocal(np.matmul(exp_eta, np.ones((_n + 1, 1))))

    denominator = denominator.flatten()

    # (N, n)
    p_prob = np.matmul(np.diag(denominator), exp_eta)

    print("p_prob")
    print(p_prob)
    return p_prob
