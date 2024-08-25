import numpy as np
from main import HierarchicalMixturesOfExperts


def compute_rss(y1: np.ndarray, y2: np.ndarray):

    return np.sum(np.square(np.subtract(y1, y2)))


def compute_r_sq(y1: np.ndarray, y2: np.ndarray):
    N = y1.shape[0]
    rss = compute_rss(y1, y2)

    mean = y1.mean()

    tss = np.sum(np.square(np.subtract(y1, mean)))

    return 1 - rss / tss


def example1():
    # N = 4, p - 1 = 1
    X = np.array([[2], [4], [5], [3]])
    Y = np.array([3, 4, 2, 5])

    # n = 3, m = 2
    n = 3
    m = 2

    hme = HierarchicalMixturesOfExperts(n=n, m=m)

    hme.fit(X=X, Y=Y)


def example2():
    # N = 10, p - 1 = 2
    X = np.array(
        [
            [2, 3],
            [4, 2],
            [5, 3],
            [3, 1],
            [3, 7],
            [9, 10],
            [2, 9],
            [12, 1],
            [1, 15],
            [20, 30],
        ]
    )
    Y = np.array([3, 4, 2, 5, 10, 7, 8, 9, 11, 20])

    # n = 3, m = 2
    n = 3
    m = 2

    hme = HierarchicalMixturesOfExperts(n=n, m=m)

    hme.fit(X=X, Y=Y)

    y_predicted = hme.predict(X)

    print("R^2 is ", compute_r_sq(y_predicted, Y))


# example1()
example2()
