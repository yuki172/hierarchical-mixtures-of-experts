import numpy as np
from typing import Tuple
from math import sqrt, pi


def compute_p_ij_gaussian(
    X: np.ndarray,
    Y: np.ndarray,
    beta_expert: np.ndarray,
    sigma_sq_expert: np.ndarray,
    beta_top: np.ndarray,
    beta_lower: np.ndarray,
):
    """
    N: number of observations in the sample \n
    p: number of input features\n
    n: number of classes in the top level gating node multinomial distribution \n
    m: number of classes in the lower level gating node multinomial distribution \n

    X: feature matrix, of shape (N, p), first column is 1 \n
    Y: output vector, of shape (N, 1) \n
    beta_expert: coefficients of x forming the mean of the normal distribution at the expert node, of shape (n, m, p) \n
    sigma_sq_expert: variance of the normal distribution at the expert node, of shape (n, m) \n

    beta_top: coefficients of the multinomial class probabilities (n classes) at the top gating node, of shape (n - 1, p) \n
    beta_lower: coefficients of the multinomial class probabilities (m classes) at the lower gating nodes (n, m - 1, p) \n

    p_gaussian: values of the densities, of shape (n, m, N)
    """
