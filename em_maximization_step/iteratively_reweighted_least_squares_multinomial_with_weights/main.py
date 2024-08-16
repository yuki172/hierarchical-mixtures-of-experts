import numpy as np
from helpers.compute_eta import compute_eta
from helpers.compute_p_prob import compute_p_prob
from helpers.compute_W_c import compute_W_c
from helpers.compute_e import compute_e
from helpers.compute_X_tilde import compute_X_tilde
from helpers.compute_z import compute_z
from scipy.linalg import sqrtm

# import sys, os

# print(os.path.abspath(os.path.join("utils")))
# sys.path.append(os.path.abspath(os.path.join("utils")))

from log_font_colors import printColored


def initialize_beta(n, p):
    beta = np.zeros((n - 1, p))
    return beta


max_diff = 1 / 10000


def iteratively_reweighted_least_squares_multinomial_with_weights(
    X: np.ndarray, Y: np.ndarray, c: np.ndarray
) -> np.ndarray:
    """
    N: number of observations in the sample \n
    p: number of input features, including intercept
    n: number of classes in the multinomial distribution \n
    X: feature matrix, of shape (N, p), first column is 1 \n
    Y: output matrix, of shape (N, n - 1), y_n is omitted \n
    c: observation weights, of shape (N, 1)

    beta_curr: estimated coefficients of x in the generalized linear model, of shape (n - 1, N)
    """
    N, p = X.shape
    N, n_1 = Y.shape
    n = n_1 + 1
    X_tilde = compute_X_tilde(X, n)

    # beta: parameters, of shape (n - 1, p).
    # The i th row consists of the coefficients corresponding to the i th class
    beta_curr = initialize_beta(n, p)

    max_iter_count = 100
    iter_count = 0

    while True:
        iter_count += 1
        p_prob = compute_p_prob(X, beta_curr)
        W_c = compute_W_c(p_prob, c)
        z = compute_z(X_tilde, beta_curr, Y, p_prob)
        K_c: np.ndarray = sqrtm(W_c)  # type: ignore

        beta_new = np.linalg.lstsq(np.matmul(K_c, X_tilde), np.matmul(K_c, z))[0]

        beta_new = np.reshape(beta_new, (n - 1, p))

        diff = np.sqrt(np.sum(np.square(np.subtract(beta_curr, beta_new))))

        if iter_count % 5 == 0:
            print("IRLS multinomial loop", f"iteration {iter_count}")
            print(f"diff: {diff}")

        beta_curr = beta_new

        if diff <= max_diff:
            break

        if iter_count == max_iter_count:
            print("IRLS multinomial loop", "max_iter_count reached", "stopping")
            break

        printColored("beta_new")
        print(beta_new)

    return beta_curr


# x = np.array([[1, 1], [1, 2], [1, 3]])
# beta = np.array([[1, 1], [2, 1]])
# compute_p_prob(x, beta)

p_prob = np.array([[0.1, 0.5, 0.4], [0.2, 0.4, 0.4], [0.3, 0.1, 0.6]])
c = np.ones((3, 1))
c1 = np.array([[1], [2], [3]])
# compute_W_c(p_prob, c)


Y = np.array([[0.2, 0.3], [0.1, 0.5], [0.5, 0.2]])
# compute_e(Y, p_prob)

X = np.array([[1, 2, 3], [1, 4, 3], [1, 6, 7]])
X_tilde = compute_X_tilde(X, 3)
beta_curr = np.array([[1, 2, 1], [4, 2, 2]])
# compute_z(X_tilde, beta_curr, Y, p_prob)

iteratively_reweighted_least_squares_multinomial_with_weights(X, Y, c)
