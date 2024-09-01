import numpy as np


def sample_irls_multinomial(beta: np.ndarray, X_full: np.ndarray):
    """
    N: number of observations
    n: number of classes in the multinomial distribution
    p: number of feature variables

    beta: true coefficients, of shape (n - 1, p)
    X_full: input matrix prepended with a column of 1, of shape (N, p)

    Y: output matrix, of shape (N, n - 1)
    """

    N = X_full.shape[0]
    n_1, p = beta.shape
    n = n_1 + 1

    mu = np.concatenate([np.matmul(X_full, beta.T), np.zeros([N, 1])], axis=1)

    exp = np.exp(mu)
    denom = np.reciprocal(np.sum(exp, axis=1)).reshape(N, 1)

    prob = np.multiply(denom, exp)

    Y = np.zeros((N, n - 1))
    for i in range(N):
        prob_sum = 0
        value = np.random.uniform(0, 1)
        y_i = np.zeros(n - 1)
        for j, p_curr in enumerate(prob[i]):
            prob_sum += p_curr
            if value <= prob_sum:
                if j < n - 1:
                    y_i[j] = 1
                break
        Y[i] = y_i

    return Y


# beta = np.array([[1, -2, 3], [2, 1, -3], [-3, 4, 1]])
# sample_irls_multinomial(beta)
