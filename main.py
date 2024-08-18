import numpy as np
from em_expectation_step.main import compute_posterior_probabilities
from em_maximization_step.main import compute_maximum_likelihood_estimates
from helpers.initialize_parameters import initialize_parameters
from helpers.compute_coeff_sq_diff import compute_coeff_sq_diff
from em_expectation_step.helpers.compute_p_multinomial import compute_p_multinomial


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
        Y: output vector, of shape (N, 1)
        """

        N, p_1 = X

        # append a column of 1 to account for the intercept
        X = np.concatenate((np.ones((N, 1)), X), axis=1)

        p = X.shape[1]

        n, m, max_diff = self.n, self.m, self.max_diff

        beta_expert_curr, sigma_sq_expert_curr, beta_top_curr, beta_lower_curr = (
            initialize_parameters(p, n, m)
        )

        max_iter_count = 100
        iter_count = 0

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

            if iter_count % 10 == 0:
                print("HME main loop", f"iteration count {iter_count}")
                print("sq_diff", coeff_sq_diff)

            beta_expert_curr, sigma_sq_expert_curr, beta_top_curr, beta_lower_curr = (
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

        self.beta_expert, self.sigma_sq_expert, self.beta_top, self.beta_lower = (
            beta_expert_curr,
            sigma_sq_expert_curr,
            beta_top_curr,
            beta_lower_curr,
        )

    def predict(self, x: np.ndarray):
        """
        x: input vector, of length p - 1 \n

        returns y_hat: predicted value
        """

        n, m = self.n, self.m

        # append 1 to account for the intercept
        x = np.concatenate(([1], x))
        p = x.shape[0]

        X = x.reshape((1, p))

        # vector of length n
        p_top = compute_p_multinomial(X, beta=self.beta_top).reshape((-1,))

        # (n, m)
        p_lower = np.array([[0 for _ in range(m)] for _ in range(n)])

        for i in range(n):
            p_lower[i] = compute_p_multinomial(X, beta=self.beta_lower[i]).reshape(
                (-1,)
            )

        # (n, m)
        mean_expert = np.sum(np.multiply(self.beta_expert, x), axis=2)

        y_hat = np.sum(
            np.multiply(p_top, np.sum(np.multiply(p_lower, mean_expert), axis=1))
        )

        return y_hat
