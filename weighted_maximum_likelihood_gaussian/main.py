import numpy as np
from helpers.get_beta_weighted_least_squares import get_beta_weighted_least_squares
from typing import Tuple


def weighted_maximum_likelihood_gaussian(
    X: np.ndarray, Y: np.ndarray, c: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Assume that the conditional distribution of Y with respect to X is normal, compute the maximum likelihood estimate of (mu, sigma^2) with observation weights c \n

    N: number of observations in the sample \n
    p: number of input features
    n: number of classes in the multinomial distribution \n
    X: feature matrix, of shape (N, p), first column is 1 \n
    Y: output vector, of shape (N, 1) \n
    c: observation weights, of shape (N, 1)
    """
    beta, rss = get_beta_weighted_least_squares(X, Y, c)
    sigma_sq = rss / np.sum(c)

    print("gaussian weight least squares beta")
    print(beta)
    print("gaussian weight least squares sigma_sq")
    print(sigma_sq)
    return beta, sigma_sq


X = np.array([[1, 2, 4], [1, 4, 5], [1, 3, 4], [1, 3, 1]])
Y = np.array([[3], [2], [4], [5]])
c = np.array([[3], [1], [2], [3]])

X1 = np.array([[1], [1], [1], [1]])
c1 = np.array([[1], [1], [1], [1]])
# get_beta_weighted_least_squares(X1, Y, c1)

beta, sigma_sq = weighted_maximum_likelihood_gaussian(X1, Y, c)
