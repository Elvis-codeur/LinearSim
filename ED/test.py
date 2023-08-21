from core import Polynome, SystemeLineaire
import numpy as np
import matplotlib.pyplot as plt
from plotting import bode, nyquist, nichols


def test_evaluation_polynomial():
    Num = Polynome(coeff=[1, 0, 1])
    Den = Polynome(coeff=[1, 0, 2])
    H = SystemeLineaire(Num, Den)
    x = np.arange(-10, 10, 1/100)
    y = H(x)
    plt.plot(x, y)
    plt.show()


def test_bode():
    Num = Polynome([1])
    Den = Polynome([1, 1])
    H = SystemeLineaire(Num, Den)
    bode(H, 0.001, 1000)


def test_nyquist():
    Num = Polynome([1])
    Den = Polynome([1, 1])
    H = SystemeLineaire(Num, Den)
    nyquist(H)


def test_nichols():
    Num = Polynome([1])
    Den = Polynome([1, 1])*Polynome([1, 2.56])*Polynome([1, 25.6])
    H = SystemeLineaire(Num, Den)
    nichols(H)
    print(Den)


def test_simulation_equation_euler_explicit():
    g = Polynome([1, 6])*Polynome([1, 3])
    A = g.matriceCompagne
    print(A, g.coeff, g.roots, g.constanteTemps)
    B = np.array([1, 0])
    U = np.array([1, 0])
    C = np.array([1, 0])
    X = np.array([0, 0])
    d_Xn = np.zeros_like(X)
    time = np.arange(0, np.max(g.constanteTemps)*10,
                     np.min(g.constanteTemps)/5)
    dt = 0
    y = np.zeros_like(time)
    for i in range(1, time.shape[0]):
        dt = time[i] - time[i-1]
        d_Xn = np.dot(A, X) + B*U
        X = X + d_Xn*dt
        s = np.dot(C, X)
        y[i] = s

    plt.plot(time[:-1], y[:-1])
    plt.show()


def test_simulation_equation_euler_implicite(H: SystemeLineaire, temps,
                                             U, C, X0, input_type="compteur"):
    """
    H: C'est la fonction de transfert en temps continue de la fonction linéaire
    temps: Le temps pour la simulation
    U: Une fonction qui renvoie l'entrée
    C: La matrice C
    X0: Les condititions initiale
    type_temps: Pour savoir si on envoie le compteur dans U ou si on envoie le temps
    """
    A = H.Den.matriceCompagne
    # print(A)
    B = H.Num.matrice_commande(A.shape[0])

    #print(A,"\n\n", B)

    X = X0

    dt = 0
    y = np.zeros_like(temps)
    compteur = 0
    T = 0
    for i in range(1, temps.shape[0]):
        dt = temps[i] - temps[i-1]

        # On doit diviser la sortie par le coefficient de y'(n)
        if input_type == "compteur":
            Input = U(compteur) / H.Den.coeff[0]
        else:
            Input = U(T)/H.Den.coeff[0]

        Xt = np.linalg.inv(np.eye(A.shape[0], A.shape[0]) - A*dt)

        #print(B,Input,np.dot(B, Input),"Elvis")
        print(Xt,X,B, Input,np.dot(B, Input))
        X = np.dot(Xt, X + np.dot(B, Input)*dt)

        s = np.dot(C, X)
        y[i] = s

        compteur += 1
        T += dt

    fig, ax = plt.subplots()
    ax.plot(temps[:-1], y[:-1])
    return fig, ax


def create_input(dim,derive_n):
    def U(compteur):
        if compteur == 0:
            return np.array([1 for i in range(derive_n+1)]+
                            [0 for i in range(dim-derive_n-1)])
        else:
            return np.array([1]+[0 for i in range(dim-1)])

    return U


def test_cos(i):
    return np.array(
        [
            np.sin(10*i),  # u
            10*np.cos(10*i),  # u'
            -100*np.sin(10*i),  # u''
        ]
    )


if __name__ == "__main__":

    K = 6
    w0 = 0.1
    z = 0.7
    Den = Polynome([1, 1])
    Num = Polynome([1, 0])
    H = SystemeLineaire(Num, Den)

    C = np.array([1] + [0 for i in range(Den.coeff.shape[0] - 2)])
    X0 = np.array([0 for i in range(Den.coeff.shape[0] - 1)])

    def j(compteur):
        result = np.flip(Num.coeff != 0)*1
        if compteur == 0:
            return result
        else:
            if result[0] > 0:
                result = result*0
                result[0] = 1
                return result
            else:
                return result*0
        

    t = np.arange(0, 10, 0.1)
    if Den.coeff.shape[0] > Num.coeff.shape[0] + 1:
        U = create_input(Den.coeff.shape[0]-1,Num.coeff.shape[0]-1)
    else:
        U = create_input(Num.coeff.shape[0],Num.coeff.shape[0]-1)

    print(j(0),j(1), X0)
    
    
    fig, ax = test_simulation_equation_euler_implicite(H, t,j, C, X0, "")
    # ax.plot(t,np.sin(10*t))
    plt.show()
