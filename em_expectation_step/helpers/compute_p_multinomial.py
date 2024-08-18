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
    # print("eta")
    # print(eta)

    N, _n = eta.shape

    # (N, n)
    exp_eta = np.concatenate((np.exp(eta), np.ones((N, 1))), axis=1)

    # print("exp_eta")
    # print(exp_eta)

    # (N, 1)
    denominator = np.reciprocal(np.matmul(exp_eta, np.ones((_n + 1, 1))))

    denominator = denominator.flatten()

    # print("denominator")
    # print(denominator)

    # (N, n)
    p_prob = np.matmul(np.diag(denominator), exp_eta)

    # print("p_prob")
    # print(p_prob)

    return p_prob.T


# N = 4, p = 2
X = np.array([[1, 2], [1, 4], [1, 5], [1, 3]])

# n = 3, p = 2
beta = np.array([[1, 3], [2, 4]])

# compute_p_multinomial(X, beta)
