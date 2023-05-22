import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("D:/projet_github/LinearSim")

from core import euler_implicite2,euler_implicite3


def charge():
    l = 1e-2
    r = 1
    c = 1e-2

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
    C = [1, 0]
    X0 = [0, 0]

    def U(t):
        return Vs
    D = B*0

    t = np.linspace(0, 1e-3, 100)

    s = euler_implicite3(A, B, C, D, U, X0, t)
    dIL = s[:][0]
    dV0 = s[:][1]
    plt.plot(t,dIL)
    plt.plot(t,dV0)
    plt.show()


if __name__ == "__main__":
    charge()
