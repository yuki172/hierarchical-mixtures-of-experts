import numpy as np
from helpers.compute_eta import compute_eta
from helpers.compute_p_prob import compute_p_prob
from helpers.compute_w_c import compute_w_c


def initialize_beta(n, p):
    beta = np.ones((n - 1, p))
    return beta


def irls_multinomial_with_weights(X: np.ndarray, Y: np.ndarray, c: np.ndarray):
    """
    N: number of observations in the sample \n
    p: number of input features
    n: number of classes in the multinomial distribution \n
    X: feature matrix, of shape (N, p), first column is 1 \n
    Y: output matrix, of shape (N, n - 1), y_n is omitted \n
    c: observation weights, of shape (N, 1)
    """
    N, p = X.shape
    N, n_1 = Y.shape
    n = n_1 + 1

    # beta: parameters, of shape (n - 1, p).
    # The i th row consists of the coefficients corresponding to the i th class
    beta_curr = initialize_beta(n, p)

    # beta_array: flattened beta, of shape (1, (n - 1) * p))
    beta_curr_array = beta_curr.copy()
    beta_curr_array.flatten()


# x = np.array([[1, 1], [1, 2], [1, 3]])
# beta = np.array([[1, 1], [2, 1]])
# compute_p_prob(x, beta)

p_prob = np.array([[0.1, 0.5, 0.4], [0.2, 0.4, 0.4], [0.3, 0.1, 0.6]])
c = np.ones((3, 1))
compute_w_c(p_prob, c)
