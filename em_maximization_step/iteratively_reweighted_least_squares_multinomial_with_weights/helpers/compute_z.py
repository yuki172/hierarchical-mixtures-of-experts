import numpy as np
from helpers.compute_e import compute_e
from helpers.compute_W_c import compute_W_c


def compute_z(
    X_tilde: np.ndarray, beta_curr: np.ndarray, Y: np.ndarray, p_prob: np.ndarray
):
    """
    N: number of observations in the sample \n
    p: number of input features
    n: number of classes in the multinomial distribution \n
    X: feature matrix, of shape (N, p), first column is 1 \n
    Y: output matrix, of shape (N, n - 1), y_n is omitted \n
    beta_curr: current coefficients, of shape (n - 1, p) \n
    p_prob: result of compute_p_prob, the class probabilities, of shape (N, n) \n

    z: target vector, of shape ((n - 1) * N, 1)
    """

    N, n_1 = Y.shape
    n = n_1 + 1

    e = compute_e(Y, p_prob)
    W = compute_W_c(p_prob, np.ones((N, 1)))

    # print("e")
    # print(e)
    print("W")
    print(W)

    beta_curr_column = beta_curr.flatten()
    beta_curr_column = np.reshape(beta_curr_column, (-1, 1))
    # print("beta_curr_column")
    # print(beta_curr_column)
    # print("X_tilde")
    # print(X_tilde)

    X_beta = np.matmul(X_tilde, beta_curr_column)
    # print("X_beta")
    # print(X_beta)

    # print("w * e")
    # print(np.linalg.inv(W))
    # print(np.matmul(np.linalg.inv(W), e).shape)

    return X_beta + np.matmul(np.linalg.inv(W), e)
