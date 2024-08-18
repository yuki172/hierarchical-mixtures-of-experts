import numpy as np
from main import HierarchicalMixturesOfExperts


def example1():
    # N = 4, p - 1 = 1
    X = np.array([[2], [4], [5], [3]])
    Y = np.array([[3], [4], [2], [5]])

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
    Y = np.array([[3], [4], [2], [5], [10], [7], [8], [9], [11], [20]])

    # n = 3, m = 2
    n = 3
    m = 2

    hme = HierarchicalMixturesOfExperts(n=n, m=m)

    hme.fit(X=X, Y=Y)

    hme.predict(x=np.array([20, 30]))


# example1()
example2()
