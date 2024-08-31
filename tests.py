import numpy as np
from example_samples.irls_multinomial import sample_irls_multinomial

from em_maximization_step.iteratively_reweighted_least_squares_multinomial_with_weights.main import (
    iteratively_reweighted_least_squares_multinomial_with_weights,
)


def test_irls_multinomial():
    """
    n: number of classes in the multinomial distribution
    """

    beta = np.array([[1, -2, 3], [2, 1, -3], [-3, 4, 1]])
    X, Y = sample_irls_multinomial(beta)
    N = X.shape[0]
    c = np.ones((N, 1))

    beta_estimates = iteratively_reweighted_least_squares_multinomial_with_weights(
        X, Y, c
    )

    print(np.round(beta_estimates))


test_irls_multinomial()
