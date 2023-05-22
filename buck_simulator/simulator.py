import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("D:/projet_github/LinearSim")
from core import euler_implicite2, euler_implicite3

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

f =  100
alpha = 0.75



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
    X0 = [0, 0]

    def U(t):
        return Vs
    D = B*0

    s = euler_implicite3(A, B, C, D, U, X0, t)
    print(s.shape)

    dIL = s[:, 0]
    dV0 = s[:, 1]

    return (t[:-1], integrate(dIL, t, X0[0]), integrate(dV0, t, X0[1]))


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

    s = euler_implicite3(A, B, C, D, U, X0, t)
    print(s.shape)

    dIL = s[:, 0]
    dV0 = s[:, 1]

    return (t[:-1], integrate(dIL, t, X0[0]), integrate(dV0, t, X0[1]))
    





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

if __name__ == "__main__":
    test_charge_decharge()