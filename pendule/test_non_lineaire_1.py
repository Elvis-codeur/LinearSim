import numpy as np

DT = 0.1


def f(X):
    return np.array([
        X[0] - np.sin(X[0]) - X[1],
        X[1]
    ])


def df(X):
    return np.array([
        [1 - np.cos(X[0]), -1],
        [0, 1]
    ])


def newtonRaphson(X0, f, df, n_iteration):
    X = X0
    i = 0
    while i < n_iteration:
        X = X - np.dot(f(X), np.linalg.inv(df(X)))
        i += 1
    return X

if __name__ == "__main__":
    X0 = np.array([1,1.1])
    print(newtonRaphson(X0,f,df,10))