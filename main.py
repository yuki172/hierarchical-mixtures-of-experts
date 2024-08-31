import numpy as np
from em_expectation_step.main import compute_posterior_probabilities
from em_maximization_step.main import compute_maximum_likelihood_estimates
from helpers.initialize_parameters import initialize_parameters
from helpers.compute_coeff_sq_diff import compute_coeff_sq_diff
from em_expectation_step.helpers.compute_p_multinomial import compute_p_multinomial
from log_font_colors import printColored


class HierarchicalMixturesOfExperts:
    max_diff_default = 1 / 1000

    def __init__(self, n: int, m: int, max_diff: float = max_diff_default):
        """
        n: number of classes at the top gating network \n
        m: number of classes at each lower gating network \n
        max_diff: if sum of squared differences between the new and current coefficients in an iteration is less than or equal to this value, the iterations end.
        """
        self.n = n
        self.m = m
        self.max_diff = max_diff
        self.beta_expert, self.sigma_sq_expert, self.beta_top, self.beta_lower = (
            np.array([0]),
            np.array([1]),
            np.array([0]),
            np.array([0]),
        )

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """
        N: number of observations in the sample \n
        p: number of input features, including intercept \n

        X: feature matrix, of shape (N, p - 1)\n
        Y: output vector, of length N
        """

        N, p_1 = X.shape

        Y = Y.reshape((N, 1))

        # append a column of 1 to account for the intercept
        X = np.concatenate((np.ones((N, 1)), X), axis=1)

        p = X.shape[1]

        n, m, max_diff = self.n, self.m, self.max_diff

        # printColored("X")
        # print(X)

        # printColored(f"n {n}, m {m}, N {N}, p {p}")

        beta_expert_curr, sigma_sq_expert_curr, beta_top_curr, beta_lower_curr = (
            initialize_parameters(p, n, m)
        )

        # printColored("initial beta_expert_curr")
        # print(beta_expert_curr)

        # printColored("initial sigma_sq_expert_curr")
        # print(sigma_sq_expert_curr)

        # printColored("initial beta_top_curr")
        # print(beta_top_curr)

        # printColored("initial beta_lower_curr")
        # print(beta_lower_curr)

        max_iter_count = 100
        iter_count = 0

        printColored("HME main loop starts")

        try:
            while True:
                iter_count += 1

                # EM algorithm - expectation step
                h_top, h_lower_cond_top, h_top_lower = compute_posterior_probabilities(
                    X,
                    Y,
                    beta_expert=beta_expert_curr,
                    sigma_sq_expert=sigma_sq_expert_curr,
                    beta_top=beta_top_curr,
                    beta_lower=beta_lower_curr,
                )

                # EM algorithm - maximization step
                beta_expert_new, sigma_sq_expert_new, beta_top_new, beta_lower_new = (
                    compute_maximum_likelihood_estimates(
                        X, Y, h_top, h_lower_cond_top, h_top_lower
                    )
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
            print("error")

        # printColored(f"HME main loop ends, diff {coeff_sq_diff}")

        self.beta_expert, self.sigma_sq_expert, self.beta_top, self.beta_lower = (
            beta_expert_curr,
            sigma_sq_expert_curr,
            beta_top_curr,
            beta_lower_curr,
        )

        # printColored("final beta_expert_curr")
        # print(beta_expert_curr)

        # printColored("final sigma_sq_expert_curr")
        # print(sigma_sq_expert_curr)

        # printColored("final beta_top_curr")
        # print(beta_top_curr)

        # printColored("final beta_lower_curr")
        # print(beta_lower_curr)

    def predict(self, X: np.ndarray):
        """
        N: number of observations

        X: input matrix, of shape (N, p - 1) \n

        returns y_hat: predicted value, a vector of length N
        """

        N = X.shape[0]
        n, m = self.n, self.m

        # append 1 to account for the intercept
        X = np.concatenate((np.ones((N, 1)), X), axis=1)
        p = X.shape[1]

        printColored("X")
        print(X)

        # (n, N)
        p_top = compute_p_multinomial(X, beta=self.beta_top)

        # (n, m, N)
        p_lower = np.zeros((n, m, N))

        for i in range(n):
            p_lower[i] = compute_p_multinomial(X, beta=self.beta_lower[i])

        # (n, m, N)
        mean_expert = np.zeros((n, m, N))

        for i in range(n):
            for j in range(m):
                mean_expert[i][j] = np.matmul(self.beta_expert[i][j], X.T)

        # (n, N)
        means_1 = np.sum(np.multiply(mean_expert, p_lower), axis=1)

        means_2 = np.multiply(p_top, means_1)

        y_hat = np.sum(means_2, axis=0)

        printColored("y_hat")
        print(y_hat)

        return y_hat
