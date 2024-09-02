import numpy as np
from em_expectation_step.main import compute_posterior_probabilities
from em_maximization_step.main import compute_maximum_likelihood_estimates
from helpers.initialize_parameters import initialize_parameters
from helpers.compute_coeff_sq_diff import compute_coeff_sq_diff
from em_expectation_step.helpers.compute_p_multinomial import compute_p_multinomial
from log_font_colors import printColored
from typing import Callable
from helpers.compute_r_sq import compute_r_sq
from math import inf


class HierarchicalMixturesOfExperts:
    max_diff_default = 1 / 1000

    def __init__(
        self,
        n: int,
        m: int,
        max_diff: float = max_diff_default,
        max_iter: int = 100,
        num_fit_tries=1,
        score: Callable[[np.ndarray, np.ndarray], np.ndarray] = compute_r_sq,
    ):
        """
        n: number of classes at the top gating network \n
        m: number of classes at each lower gating network \n
        max_diff: if sum of squared differences between the new and current coefficients in an iteration is less than or equal to this value, the iterations end.\n
        max_iter: in each run of the algorithm during fitting, if this number of iterations is reached in the loop of the EM algorithm, the algorithm will stop.\n
        num_tries: during fitting, the number of times the algorithm will be run starting at random initial values. This is needed since starting at random initial values may not lead to good maximum likelihood estimates. At least 1 run will be performed.\n
        score: score function to evaluate model accuracy. Used to determine the best coefficients during fitting among all num_tries runs of the algorithm. \n
        """
        self.n = n
        self.m = m
        self.max_diff = max_diff
        self.max_iter = max_iter
        self.beta_expert, self.sigma_sq_expert, self.beta_top, self.beta_lower = (
            np.array([0]),
            np.array([1]),
            np.array([0]),
            np.array([0]),
        )
        self.num_fit_tries = max(1, num_fit_tries)
        self.score = score

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """
        N: number of observations in the sample \n
        p: number of input features, including intercept \n

        X: feature matrix, of shape (N, p - 1)\n
        Y: output vector, of length N
        """

        N, p_1 = X.shape
        p = p_1 + 1

        Y_arr = Y.reshape((N, 1))

        # prepend a column of 1 to account for the intercept
        X_full = np.concatenate((np.ones((N, 1)), X), axis=1)

        n, m, max_diff = self.n, self.m, self.max_diff

        best_coeff = (
            self.beta_expert,
            self.sigma_sq_expert,
            self.beta_top,
            self.beta_lower,
        )
        best_score = -inf

        for _ in range(self.num_fit_tries):
            print("num try")
            beta_expert_curr, sigma_sq_expert_curr, beta_top_curr, beta_lower_curr = (
                initialize_parameters(p, n, m)
            )

            max_iter_count = self.max_iter
            iter_count = 0

            # printColored("HME main loop starts")

            try:
                while True:
                    iter_count += 1

                    # EM algorithm - expectation step
                    h_top, h_lower_cond_top, h_top_lower = (
                        compute_posterior_probabilities(
                            X_full,
                            Y_arr,
                            beta_expert=beta_expert_curr,
                            sigma_sq_expert=sigma_sq_expert_curr,
                            beta_top=beta_top_curr,
                            beta_lower=beta_lower_curr,
                        )
                    )

                    # EM algorithm - maximization step
                    (
                        beta_expert_new,
                        sigma_sq_expert_new,
                        beta_top_new,
                        beta_lower_new,
                    ) = compute_maximum_likelihood_estimates(
                        X_full, Y_arr, h_top, h_lower_cond_top, h_top_lower
                    )

                    coeff_sq_diff = compute_coeff_sq_diff(
                        beta_expert_curr,
                        beta_expert_new,
                        sigma_sq_expert_curr,
                        sigma_sq_expert_new,
                        beta_top_curr,
                        beta_top_new,
                        beta_lower_curr,
                        beta_lower_new,
                    )

                    # if iter_count % 10 == 0:
                    #     printColored(f"HME main loop, iteration count {iter_count}")
                    #     print("sq_diff", coeff_sq_diff)

                    (
                        beta_expert_curr,
                        sigma_sq_expert_curr,
                        beta_top_curr,
                        beta_lower_curr,
                    ) = (
                        beta_expert_new,
                        sigma_sq_expert_new,
                        beta_top_new,
                        beta_lower_new,
                    )

                    if coeff_sq_diff <= max_diff:
                        break

                    if iter_count == max_iter_count:
                        print("HME main loop", "max_iter_count reached", "stopping")
                        break
            except:
                print("")

            # printColored(f"HME main loop ends, diff {coeff_sq_diff}")

            Y_pred_curr = self.__predict(
                X,
                beta_expert_curr,
                beta_top_curr,
                beta_lower_curr,
            )
            score_curr = self.score(Y, Y_pred_curr)

            print("score", score_curr)
            if score_curr > best_score:
                best_score = score_curr

                best_coeff = (
                    beta_expert_curr,
                    sigma_sq_expert_curr,
                    beta_top_curr,
                    beta_lower_curr,
                )
        (
            self.beta_expert,
            self.sigma_sq_expert,
            self.beta_top,
            self.beta_lower,
        ) = best_coeff

    def __predict(
        self,
        X: np.ndarray,
        beta_expert: np.ndarray,
        beta_top: np.ndarray,
        beta_lower: np.ndarray,
    ):
        """
        N: number of observations in the sample \n
        p: number of input features, including intercept \n

        X: feature matrix, of shape (N, p - 1)\n

        beta_expert: coefficients of x forming the mean of the normal distribution at the expert node, of shape (n, m, p) \n
        sigma_sq_expert: variance of the normal distribution at the expert node, of shape (n, m) \n

        beta_top: coefficients of the multinomial class probabilities (n classes) at the top gating node, of shape (n - 1, p) \n
        beta_lower: coefficients of the multinomial class probabilities (m classes) at the lower gating nodes (n, m - 1, p) \n

        """
        N = X.shape[0]
        n, m = self.n, self.m

        # append 1 to account for the intercept
        X = np.concatenate((np.ones((N, 1)), X), axis=1)
        p = X.shape[1]

        # (n, N)
        p_top = compute_p_multinomial(X, beta=beta_top)

        # (n, m, N)
        p_lower = np.zeros((n, m, N))

        for i in range(n):
            p_lower[i] = compute_p_multinomial(X, beta=beta_lower[i])

        # (n, m, N)
        mean_expert = np.zeros((n, m, N))

        for i in range(n):
            for j in range(m):
                mean_expert[i][j] = np.matmul(beta_expert[i][j], X.T)

        # (n, N)
        means_1 = np.sum(np.multiply(mean_expert, p_lower), axis=1)

        means_2 = np.multiply(p_top, means_1)

        y_hat = np.sum(means_2, axis=0)

        return y_hat

    def predict(self, X: np.ndarray):
        """
        N: number of observations

        X: input matrix, of shape (N, p - 1) \n

        returns y_hat: predicted value, a vector of length N
        """

        return self.__predict(
            X,
            beta_expert=self.beta_expert,
            beta_top=self.beta_top,
            beta_lower=self.beta_lower,
        )
