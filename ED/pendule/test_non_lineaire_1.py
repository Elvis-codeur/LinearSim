import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


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




def newtonRaphson_matrice(X0, f, df, n_iteration):
    X = X0
    i = 0
    while i < n_iteration:
        X = X - np.dot(f(X), np.linalg.inv(df(X)))
        i += 1
    return X

def newtonRaphson(X0, f, df, n_iteration):
    X = X0
    i = 0
    while i < n_iteration:
        X = X - f(X)/df(X)
        i += 1
    return X


def model(y,t):
    return np.sin(y)

def solve(y0,dt,T):
    yi = y0 

    def f(x):
        return x -dt*np.sin(x) - yi
    
    def df(x):
        return 1 - dt*np.cos(x) 

    
    time = 0
    t = [0]
    s = [y0] 

    while(time < T):
        time += dt 
        y0 = newtonRaphson(yi,f,df,10)
        yi = y0 
        t.append(time)
        s.append(y0)

    return t,s 


    


def odeint_result():
    y0 = 5
    # time points
    t = np.arange(0,20,0.01)
    # solve ODE
    y = odeint(model,y0,t)
    return t,y 

def test_newtonRaphson():
    def f(x):
        return x - np.cos(x)
    def df(x):
        return 1 + np.sin(x)
    
    print(newtonRaphson(1,f,df,10))

if __name__ == "__main__":
    test_newtonRaphson()
    t,s =solve(5,0.01,20)
    ode_t, ode_s = odeint_result()

    fig,ax = plt.subplots()
    fig.set_dpi(150)
    ax.plot(t,s,label = "Résultat de mon solveur")
    ax.plot(ode_t,ode_s,label = "Résultat de odeint")
    ax.set_xlabel("x")
    ax.set_ylabel("y(x)")
    ax.legend()
    fig.suptitle("Comparaison entre Odeint du module scipy \n et de mon solveur")
    plt.show()
