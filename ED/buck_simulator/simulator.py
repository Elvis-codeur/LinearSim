import numpy as np
import matplotlib.pyplot as plt

import sys

sys.path.append("D:/projet_github/LinearSim")
from buck_simulator.solveur import euler_implicite

def integrate(s, t, x0):
    result = []
    sum = x0
    for i in range(0, s.shape[0]-1):
        sum += s[i]*(t[i+1] - t[i])
        result.append(sum)

    return np.array(result)


l = 1e-2
r = 1
c = 1e-2

f =  1e5
alpha = 0.75
T = 1/f


def charge(X0,t):

    A = np.array(
        [
            [0, -1/l],
            [1/c, -1/(r*c)],
        ]
    )
    B = np.array(
        [
            [1/l, 0],
            [0, 0]
        ]
    )
    Vs = [12, 0]
    C = [1, 1]

    def U(t):
        return Vs
    D = B*0

    s = euler_implicite(A, B, C, D, U, X0, t)
    #print(s.shape)

    dIL = s[:, 0]
    dV0 = s[:, 1]

    return (t,dIL,dV0)
def decharge(X0,t):
    A = np.array(
        [
            [0, -1/l],
            [1/c, -1/(r*c)],
        ]
    )
    B = np.array(
        [
            [0, 0],
            [0, 0]
        ]
    )
    Vs = [12, 0]
    C = [1, 1]

    def U(t):
        return Vs
    D = B*0

    s = euler_implicite(A, B, C, D, U, X0, t)
    #print(s.shape)

    dIL = s[:, 0]
    dV0 = s[:, 1]

    return (t,dIL,dV0)
    





def test_integrale():
    t = np.linspace(0, 8, 1000)
    y = np.sin(10*t)
    z = -1.0/(10.0)*np.cos(10*t)
    i = integrate(y, t, -0.1)
    plt.plot(t, y, label="sin")
    plt.plot(t, z, label="cos")
    plt.plot(t[:-1], i, label="integrate")
    plt.legend()
    plt.show()

def test_charge_decharge():
    X0 = [0.0,0.0]

    for i in range(1,7):
        T = (1/f)
        t1 = np.linspace((i-1)*T,(i)*T*alpha,100)   
        t1_,idl1,v01 =  charge(X0,t1)
        X0 = [idl1[-1],v01[-1]]


        t2 = np.linspace((i)*T*alpha*T,i*T,100)
        t2_,idl2,v02 =  decharge(X0,t2)
        X0 = [idl1[-1],v01[-1]]

        plt.plot(t1_,idl1,color = "red")
        plt.plot(t2_,idl2,color = "blue")

    plt.legend()
    plt.show()


def test_decharge():
    X0 = [1,2]
    t = np.linspace(0,1e-2,100)
    t_,idl,v0 = decharge(X0,t)
    plt.plot(t_,idl,label = "idl")
    plt.plot(t_,v0,label = "idl")
    plt.legend()
    plt.show()

def simulate():
    X0 = np.array([0,0])
    t =  T 
    time = np.array([])
    i = np.array([])
    u = np.array([])

    while t < 1000*T:
        t1 =  np.linspace(t-T,t-(1- alpha)*T  ,10)
        t2 =  np.linspace(t-(1- alpha)*T,t,10)

        t_charge, i_charge, u_charge, = charge(X0,t1)
        X0 = np.array([i_charge[-1],u_charge[-1]])

        t_decharge, i_decharge, u_decharge, = decharge(X0,t2)
        X0 = np.array([i_decharge[-1],u_decharge[-1]])

        time = np.concatenate([time,t1,t2])
        i = np.concatenate([i,i_charge,i_decharge])
        u = np.concatenate([u,u_charge,u_decharge])
        
        t += T



    plt.plot(time,i,label = "i")
    plt.plot(time,u,label = "u")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    X0 = np.array([0,0])
    t = np.linspace(0,1,1000)
    t,i,u = charge(X0,t)
    plt.plot(t,i,label = "I_l")
    plt.plot(t,u,label = "U_c")
    plt.legend()
    plt.show()   
    
    

    