import sys
import matplotlib.pyplot as plt
import numpy as np
from PyQt5 import QtCore, QtWidgets
from core import SystemeLineaire, create_input, euler_implicite, Polynome
from plotting_core import StepWindow


def impulse():
    a = 0


def step():
    a = 0


def bode(H: SystemeLineaire, freq_debut=0.001, freq_fin=100, ppd=50):
    """
    freq_debut: Fréquence de début
    freq_fin: Fréqunce à laquelle arrêter le diagramme
    ppd: Le nombre de point par décade
    """
    # On génère l'echelle frequencielle
    x = np.logspace(np.log10(freq_debut), np.log10(freq_fin),
                    num=int(ppd*(np.log10(freq_fin/freq_debut))))
    # On le transforme en nombre complexe
    # On evalue avec le polynome
    y = H(np.emath.sqrt(-1)*x*2*np.pi)

    # On dessine
    fig, axes = plt.subplots(2, 1)
    axes[0].set_xscale("log")
    axes[0].plot(x, 20*np.log10(np.abs(y)))
    axes[0].grid(True, "both")

    axes[1].set_xscale("log")
    axes[1].plot(x, np.angle(y)*180/np.pi)
    axes[1].grid(True, "both")

    plt.show()


def nyquist(H: SystemeLineaire, freq_debut=0.001, freq_fin=100, ppd=50):
    # On génère l'echelle frequencielle
    x = np.logspace(np.log10(freq_debut), np.log10(freq_fin),
                    num=int(ppd*(np.log10(freq_fin/freq_debut))))
    # On le transforme en nombre complexe
    # On evalue avec le polynome
    y = H(np.emath.sqrt(-1)*x*2*np.pi)

    fig, ax = plt.subplots()
    real = np.real(y)
    imag = np.imag(y)
    ax.plot(real, imag)

    step = x.shape[0]//10
    for i in range(1, step):
        print(i)
        indice = i*10
        ax.arrow(real[indice-1], imag[indice-1],
                 real[indice] - real[indice-1],
                 imag[indice]-imag[indice-1],
                 shape='full', lw=0, length_includes_head=True,
                 head_width=.01, color="r")

    ax.grid(True, "both", linewidth=0.8)
    plt.show()


def nichols(H: SystemeLineaire, freq_debut=0.001, freq_fin=100, ppd=50):
    # On génère l'echelle frequencielle
    x = np.logspace(np.log10(freq_debut),
                    np.log10(freq_fin),
                    num=int(ppd*(np.log10(freq_fin/freq_debut))))
    # On le transforme en nombre complexe
    # On evalue avec le polynome
    y = H(np.emath.sqrt(-1)*x*2*np.pi)

    fig, ax = plt.subplots()
    angle = np.angle(y)*180/np.pi
    moduleDB = 20*np.log10(np.abs(y))
    ax.plot(angle, moduleDB)

    step = x.shape[0]//10
    for i in range(1, step):
        print(i, " Elvis")
        indice = i*10
        ax.arrow(angle[indice-1], moduleDB[indice-1],
                 angle[indice] - angle[indice-1],
                 moduleDB[indice]-moduleDB[indice-1],
                 shape='full', lw=0, length_includes_head=True,
                 head_width=.01, color="r")

    ax.grid(True, "both", linewidth=0.8)
    plt.show()


def step(H: SystemeLineaire):

    C = np.array([1] + [0 for i in range(H.Den.coeff.shape[0] - 2)])
    X0 = np.array([0 for i in range(H.Den.coeff.shape[0] - 1)])

    def j(compteur):
        result = np.flip(H.Num.coeff != 0)*1
        if compteur == 0:
            return result
        else:
            if result[0] > 0:
                result = result*0
                result[0] = 1
                return result
            else:
                return result*0

    constantTemps = H.Den.constanteTemps

    print(constantTemps, "\n\n\n")

    t = np.arange(0, np.max(constantTemps)*10, np.min(constantTemps)/5)

    if H.Den.coeff.shape[0] > H.Num.coeff.shape[0] + 1:
        U = create_input(H.Den.coeff.shape[0]-2, H.Num.coeff.shape[0]-1)
    else:
        U = create_input(H.Num.coeff.shape[0], H.Num.coeff.shape[0]-1)

    print(j(0), j(1), X0)

    print("in step")

    def kl(compteur):
        return [1]

    s = euler_implicite(H, t, kl, C, X0, "")

    app = QtWidgets.QApplication(sys.argv)
    window = StepWindow()
    window.plot(t, s)
    app.exec_()


if __name__ == "__main__":
    K = 6
    w0 = 0.1
    z = 0.7
    Den = Polynome([1/(w0**2), 2*z/w0, 1])
    Num = Polynome([K])
    print(Den+Num)
    H = SystemeLineaire(Num, Den + Num)
    step(H)
