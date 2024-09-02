import numpy as np
import sklearn.ensemble
import sklearn.linear_model
import sklearn.mixture
from example_samples.irls_multinomial import sample_irls_multinomial
from example_samples.input import sample_input
from em_maximization_step.iteratively_reweighted_least_squares_multinomial_with_weights.main import (
    iteratively_reweighted_least_squares_multinomial_with_weights,
)
import math
from main import HierarchicalMixturesOfExperts
import sklearn
import time
from helpers.compute_r_sq import compute_r_sq


def test_irls_multinomial(N: int):
    """
    N: number of observations \\
    n: taken to be 4, number of classes in the multinomial distribution \\
    p: taken to be 3, number of features \\
    """
    n = 4
    p = 3

    beta = np.array([[1, -2, 3], [2, 1, -3], [-3, 4, 1]])
    _, X_full = sample_input(N, p)

    Y = sample_irls_multinomial(beta, X_full)
    c = np.ones((N, 1))

    beta_estimates = iteratively_reweighted_least_squares_multinomial_with_weights(
        X_full, Y, c
    )

    print(np.round(beta_estimates))


def test_simulated_mixed_normals(N: int):
    """
    N: number of observations \\

    X: input matrix, of shape (N, p - 1)\\
    X_full: input matrix prepended with a column of 1 \\

    n: taken to be 2, number of classes in the multinomial distribution at the top gating network \\
    m: taken to be 3, number of classes in the multinomial distribution at the lower gating networks \\
    p: taken to be 3, number of features, including the intercept \\

    """

    n = 2
    m = 3
    p = 3

    X, X_full = sample_input(N, p)

    # (n - 1, p)
    beta_top = np.array([[1, -2, 3]])

    # (n, m - 1, p)
    beta_lower = np.array([[[2, 1, -3], [-3, 4, 1]], [[1, -3, 1], [-2, 1, 3]]])

    # (n, m, p)
    beta_expert = np.array(
        [[[2, 4, 1], [3, -1, 2], [3, 2, 1]], [[1, 2, 1], [4, 1, 2], [-3, 2, 2]]]
    )

    # (n, m)
    sigma_sq_expert = np.array([[0.5, 1, 2], [2, 1, 1]])

    # (N, n - 1)
    Y_top = sample_irls_multinomial(beta_top, X_full)

    # (n, N, m - 1)
    Y_lower = np.zeros((n, N, m - 1))
    for i in range(n):
        Y_lower[i] = sample_irls_multinomial(beta_lower[i], X_full)

    # (n, m, N)
    mu_expert = np.zeros((n, m, N))

    X_full_T = X_full.T
    for i in range(n):
        for j in range(m):
            mu_expert[i][j] = np.matmul(beta_expert[i][j], X_full_T)

    # vector of length N
    Y = np.zeros(N)

    for obs in range(N):
        Y_top_obs = Y_top[obs]
        i = n - 1
        for ni in range(n - 1):
            if Y_top_obs[ni] == 1:
                i = ni
                break
        Y_lower_i_obs = Y_lower[i][obs]
        j = m - 1
        for nj in range(m - 1):
            if Y_lower_i_obs[nj] == 1:
                j = nj

        mu_obs = mu_expert[i][j][obs]
        sigma_sq = sigma_sq_expert[i][j]

        Y[obs] = np.random.normal(mu_obs, math.sqrt(sigma_sq))

    model_hme = HierarchicalMixturesOfExperts(n, m)

    model_sklearn_ls = sklearn.linear_model.LinearRegression()

    model_sklearn_rf = sklearn.ensemble.RandomForestRegressor()

    model_sklearn_gb = sklearn.ensemble.GradientBoostingRegressor()

    models = [
        ["hme", model_hme],
        ["sklearn_linear_regression", model_sklearn_ls],
        ["sklearn_random_forest", model_sklearn_rf],
        ["sklean_gradient_boosting", model_sklearn_gb],
    ]

    for model_name, model in models:

        start_time_fit = time.time()
        model.fit(X, Y)
        end_time_fit = time.time()
        time_diff_fit = end_time_fit - start_time_fit

        start_time_predict = time.time()
        Y_pred = model.predict(X)
        end_time_predict = time.time()
        time_diff_predict = end_time_predict - start_time_predict

        r2 = compute_r_sq(Y, Y_pred)

        print(f"Results for {model_name}\n")
        print("R^2", r2)
        print("fit time", time_diff_fit)
        print("predict time", time_diff_predict)
        print("\n\n")


# N_irls = 100
# test_irls_multinomial(N_irls)

N_mixed_normals = 10
test_simulated_mixed_normals(N=N_mixed_normals)
