import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from core import *


def test_rapport():

    filename = "C:\\Users\\Elvis\\Downloads\\data_step_1.csv"
    df = pd.read_csv(filename, sep=",", names=["t", "s"], header=None)
    print(df.head(5))

    

    Num = Polynome([6, ])
    Den = Polynome([100, 14, 1])
    H = SystemeLineaire(Num, Den)

    t = df["t"]
    C = np.zeros(Den.coeff.shape[0]-1)
    C[0] = 1
    X0 = np.zeros(Den.coeff.shape[0]-1)

    def U_interne(x):
        return np.array([1, 0])

    srk4 = rk4(H, t, U_interne, C, X0)

    fig, ax = plt.subplots()
    fig.set_dpi(150)

    ax.plot(t, srk4, label="Mon Solveur")
    ax.plot(df["t"], df["s"], label="Matlab")

    ax.set_xlabel("temps (s)")
    ax.set_ylabel("Réponse du sytème")
    #fig.suptitle(r"H(p) = \frac{6}{100p^2 + 14p + 1}")

    print(np.sum(df["s"].values -srk4)**2)

    ax.legend()
    plt.show()


if __name__ == "__main__":
    test_rapport()
