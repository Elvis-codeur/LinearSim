import numpy as np

DT = 0.01


def f(X):
    return np.array([
        2*X[0] + 4*X[1],
        7*X[0] - 7*X[1]
    ])


def df(X):
    return np.array([
        [2, 4],
        [7, -7]
    ])


def f_sin(X):
    return np.array([
        DT*np.sin(X[0]) + X[1],
        X[1]
    ])

def df_sin(X):
    return np.array([
        [DT*np.cos(X[0]), 1],
        [0,1]
    ])


def newtonRaphson(X0, f, df, n_iteration):
    X = X0
    i = 0
    while i < n_iteration:
        X = X - np.dot(f(X), np.linalg.inv(df(X)))
        i += 1
    return X


if __name__ == "__main__":
    X0 = np.array([1,1 ])
    print(newtonRaphson(X0, f_sin, df_sin, 10))
