import numpy as np
from scipy.linalg import sqrtm


def get_beta_weighted_least_squares(X: np.ndarray, Y: np.ndarray, c: np.ndarray):
    """
    Assume that the conditional distribution of Y with respect to X is normal, compute the maximum likelihood estimate of (mu, sigma^2) with observation weights c \n

    N: number of observations in the sample \n
    p: number of input features, including intercept
    n: number of classes in the multinomial distribution \n
    X: feature matrix, of shape (N, p), first column is 1 \n
    Y: output vector, of shape (N, 1) \n
    c: observation weights, of shape (N, 1)

    beta: least squares estimates of the coefficients of x forming the mean, an array of length p \n
    rss: rss of the weighted linear regression algorithm
    """
    N, p = X.shape
    c = c.flatten()
    W = np.diag(c)
    K: np.ndarray = sqrtm(W)  # type: ignore

    # beta is of shape (p, 1), rss is an array of length 1
    beta, rss = np.linalg.lstsq(np.matmul(K, X), np.matmul(K, Y))[:2]

    # array of length p
    beta = beta.reshape(p)

    # the value of rss
    rss = rss[0]

    # print(beta)
    # print(rss)

    return beta, rss
