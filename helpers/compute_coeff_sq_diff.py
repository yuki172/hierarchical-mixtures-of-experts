import numpy as np


def compute_coeff_sq_diff(
    beta_expert_curr,
    beta_expert_new,
    sigma_sq_expert_curr,
    sigma_sq_expert_new,
    beta_top_curr,
    beta_top_new,
    beta_lower_curr,
    beta_lower_new,
):
    sq_diff = 0
    for coeff_curr, coeff_new in [
        [beta_expert_curr, beta_expert_new],
        [sigma_sq_expert_curr, sigma_sq_expert_new],
        [beta_top_curr, beta_top_new],
        [beta_lower_curr, beta_lower_new],
    ]:
        sq_diff += np.sum(np.square(np.subtract(coeff_curr, coeff_new)))

    return sq_diff
