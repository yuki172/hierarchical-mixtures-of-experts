import numpy as np
from main import HierarchicalMixturesOfExperts
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def compute_rss(y1: np.ndarray, y2: np.ndarray):

    return np.sum(np.square(np.subtract(y1, y2)))


def compute_r_sq(y1: np.ndarray, y2: np.ndarray):
    N = y1.shape[0]
    rss = compute_rss(y1, y2)

    mean = y1.mean()

    tss = np.sum(np.square(np.subtract(y1, mean)))

    return 1 - rss / tss


def example3():
    data = sns.load_dataset("tips")
    data = pd.get_dummies(data, drop_first=True)

    for col in [
        "sex_Female",
        "smoker_No",
        "day_Fri",
        "day_Sat",
        "day_Sun",
        "time_Dinner",
    ]:
        data[col] = data[col].astype(int)

    X_df = data.drop("tip", axis=1)
    y_df = data["tip"]

    X = X_df.to_numpy()
    y = y_df.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    y_pred_linear_model = linear_model.predict(X_test)

    print("linear regression R^2")
    print(compute_r_sq(y_test, y_pred_linear_model))

    hme_model = HierarchicalMixturesOfExperts(n=2, m=3)
    hme_model.fit(X=X_train, Y=y_train)

    y_pred_hme = hme_model.predict(X_test)
    print("HME R^2")
    print(compute_r_sq(y_test, y_pred_hme))


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
# example3()
