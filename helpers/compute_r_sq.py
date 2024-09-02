import numpy as np


def compute_rss(y1: np.ndarray, y2: np.ndarray):

    return np.sum(np.square(np.subtract(y1, y2)))


def compute_r_sq(y1: np.ndarray, y2: np.ndarray):
    N = y1.shape[0]
    rss = compute_rss(y1, y2)

    mean = y1.mean()

    tss = np.sum(np.square(np.subtract(y1, mean)))

    return 1 - rss / tss
