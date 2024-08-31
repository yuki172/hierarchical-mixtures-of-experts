import numpy as np

N = 100


def sample_irls_multinomial(beta: np.ndarray):
    """
    n: number of classes in the multinomial distribution
    p: number of feature variables

    beta: true coefficients, of shape (n - 1, p)

    X: input matrix, of shape (N, p)
    Y: output matrix, of shape (N, n - 1)
    """

    n_1, p = beta.shape
    n = n_1 + 1

    X = np.zeros((N, p))

    for i in range(N):
        X[i] = np.random.uniform(0, 1, p)

    mu = np.concatenate([np.matmul(X, beta.T), np.zeros([N, 1])], axis=1)

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

    return X, Y


# beta = np.array([[1, -2, 3], [2, 1, -3], [-3, 4, 1]])
# sample_irls_multinomial(beta)
