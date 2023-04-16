import numpy as np 
import matplotlib.pyplot as plt 

from scipy.integrate import odeint

def fonction(y,x):
    return -2*np.cos(x)*(1-y) - np.sin(x)*y

def model(y,t):
    dydt = -2*np.cos(t) * y
    return dydt


def euler_explicite(model,x0,t):
    """
    model (y,t): return an expression of t:time ant y: the fontion
    """
    y = [x0]
    for i in range(1,t.shape[0]):
        y.append( y[i-1] + model(y[i-1],t[i-1])*(t[i] - t[i-1]))

    y = np.array(y)
    plt.plot(t,y,color ="red")
    plt.plot(t,odeint(model,x0,t),color="blue")
    plt.show()

def euler_implicite():
    x = np.arange(0,10,0.1)
    y = [1]
    for i in range(1,x.shape[0]):
        y.append(y[i-1]/(1 + 2*(x[i] -x[i-1])))

    plt.plot(x,y,color = "red")
    plt.plot(x,np.exp(-2*x),color ="blue")
    plt.show()
if __name__ =="__main__":
    t = np.arange(0,10,0.001)
    euler_explicite(fonction,1,t)