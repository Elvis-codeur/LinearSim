import numpy as np 
import matplotlib.pyplot as plt 
g = 9.81
l = 1

K = g/l

DT = 0.01

def f(x):
    return x - np.cos(x)

def df(x):
    return 1 + np.sin(x)

def newtonRaphson(fonction,d_fonction,X0,n_iteration):
    i = 0
    X = X0
    while(i < n_iteration):
        X = X -  fonction(X)/d_fonction(X)
        i += 1
    return X

def f_pendule(x):
    return x - DT*K*np.cos(x) 

def df_pendule(x):
    return 1 + DT*K*np.sin(x)

def solve_pendule(dt,T,f,df,X0,d_X0):
    time = 0
    X = d_X0
    result_time = [0]
    result_s = [X0]

    while time < T:
        X = newtonRaphson(f,df,X,10)
        result_time.append(time)
        result_s.append(X)
        time += dt 
    return result_time,result_s


if __name__ == "__main__":
    print(newtonRaphson(f,df,1,10))
    t,s =  solve_pendule(DT,3,f_pendule,df_pendule,np.pi/4,0)
    plt.plot(t,s)
    plt.show()

