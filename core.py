import numpy as np
import matplotlib.pyplot as plt 

class Fonction():
    def __init__(self, *args, **kwargs) -> None:
        pass

    def init(self):
        pass


class Polynome(Fonction):

    def __init__(self, coeff=[], roots=[],
                 *args, **kwargs):

        self.coeff = np.array(coeff)
        self.roots = np.array(roots)
        self.constanteTemps = np.array([])
        self.matriceCompagne = np.array([])
        self.matriceModale = np.array([])

        self.init()
        super(Polynome, self).__init__(*args, **kwargs)

    def init(self):
        super().init()
        t = self.coeff.shape[0]
        self.matriceCompagne = np.zeros((t-1, t-1))
        for x in range(0, t-2):
            for y in range(0, t-2):
                if (x == y):
                    self.matriceCompagne[x][y+1] = 1

        for i in range(1, t):
            self.matriceCompagne[-1, i-1] = -(self.coeff[t-i]/self.coeff[0])

        # print(self.matriceCompagne)
        self.roots = np.linalg.eig(self.matriceCompagne)[0]
        self.matriceModale = np.zeros_like(self.matriceCompagne)
        np.fill_diagonal(self.matriceModale, self.roots)

        self.constanteTemps = np.abs(
            np.real(
                np.divide(
                    np.ones_like(self.roots),
                    np.abs(self.roots),
                    out=np.zeros_like(self.roots),
                    where=self.roots != 0
                )))
        # print(self.roots)
        # print(self.matriceCompagne)
        # print(self.matriceModale)

    def matrice_commande(self, dim: int):
        # print(dim,self.coeff.shape)
        """if dim >= self.coeff.shape[0]:
            l = dim 
        else:"""

        l = self.coeff.shape[0]

        result = np.zeros(shape=(dim, l))

        for i in range(self.coeff.shape[0]):
            result[dim-1, i] = self.coeff[self.coeff.shape[0] - i-1]

        # print(result)

        return result

    def __str__(self) -> str:
        return self.coeff.__str__()

    def __repr__(self) -> str:
        return self.coeff.__repr__()

    def add_or_substract(self, other, signe):
        result = 0
        # On prend le degre du polynome qui a le grand degre
        max_size = np.max([self.coeff.shape[0], other.coeff.shape[0]])

        # On calcule le nombre de coefficient 0 à mettre à droite
        diff = max_size - np.min([self.coeff.shape[0], other.coeff.shape[0]])

        # On agrandit la taille du petit polynome pour avoir la bonne taille
        if self.coeff.shape[0] > other.coeff.shape[0]:
            _coeff = np.concatenate([np.zeros((diff,)), other.coeff], 0)
            if signe == "+":
                result = Polynome(coeff=self.coeff + _coeff)
            elif signe == "-":
                result = Polynome(coeff=self.coeff - _coeff)
            else:
                raise RuntimeError(
                    "L'opérateur {} n'est pris en charge ".format(signe))

        elif self.coeff.shape[0] < other.coeff.shape[0]:
            _coeff = np.concatenate([np.zeros((diff,)), self.coeff], 0)
            if signe == "+":
                result = Polynome(coeff=self.coeff + _coeff)
            elif signe == "-":
                result = Polynome(coeff=self.coeff - _coeff)
            else:
                raise RuntimeError(
                    "L'opérateur {} n'est pris en charge ".format(signe))

        else:
            if signe == "+":
                result = Polynome(coeff=self.coeff + _coeff)
            elif signe == "-":
                result = Polynome(coeff=self.coeff - _coeff)
            else:
                raise RuntimeError(
                    "L'opérateur {} n'est pris en charge ".format(signe))
        return result

    def __mul__(self, other):
        # Calcule le produit entre en le  polynome et un autre
        return Polynome(np.convolve(self.coeff, other.coeff, "full"))

    def __pow__(self, number: int):
        # Calcule la puissance du polynome
        result_coeff = self.coeff
        for i in range(number-1):
            result_coeff = np.convolve(result_coeff, self.coeff, "full")
        return Polynome(coeff=result_coeff)

    def __add__(self, other):
        # Calcule la somme entre le polynome et un autre
        return self.add_or_substract(other, "+")

    def __sub__(self, other):
        # Calcule la différence entre le polynome et un autre
        return self.add_or_substract(other, "-")

    def __call__(self, x):
        # Évalue le polynome pour un nombre complexe x quelconque
        max_pow = self.coeff.shape[0]
        result = np.zeros_like(x, dtype=np.float64)
        for i in range(max_pow):
            result += np.power(x, i)*self.coeff[max_pow-i-1]

        return result


class FonctionRatiaonnelle(Fonction):
    def __init__(self, Num: Fonction, Den: Fonction, *args, **kwargs) -> None:
        self.Num = Num
        self.Den = Den
        super().__init__(*args, **kwargs)


class SystemeLineaire(FonctionRatiaonnelle):
    def __init__(self, Num: Fonction, Den: Fonction, *args, **kwargs) -> None:
        super().__init__(Num, Den, *args, **kwargs)
        self.Num = Num
        self.Den = Den
        self.Num.init()
        self.Den.init()

    def __call__(self, x):
        return self.Num(x)/self.Den(x)


def euler_implicite(H: SystemeLineaire, temps,
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

    print(A,"\n\n", B)

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

        #print(B,Input,"elv")
        #print(B,Input,np.dot(B, Input),"Elvis")
        #print(Xt, X, B, Input, np.dot(B, Input))
        X = np.dot(Xt, X + np.dot(B, Input)*dt)

        s = np.dot(C, X)
        y[i] = s

        compteur += 1
        T += dt

    return y


def euler_implicite2(A,B,C,D,U,X0,temps,input_type="compteur"):
    """
    H: C'est la fonction de transfert en temps continue de la fonction linéaire
    temps: Le temps pour la simulation
    U: Une fonction qui renvoie l'entrée
    C: La matrice C
    X0: Les condititions initiale
    type_temps: Pour savoir si on envoie le compteur dans U ou si on envoie le temps
    """
    print(A,"\n\n", B)

    X = X0

    dt = 0
    y = np.zeros_like(temps)
    y[0] = X0
    compteur = 0
    T = 0
    for i in range(1, temps.shape[0]):
        dt = temps[i] - temps[i-1]

        # On doit diviser la sortie par le coefficient de y'(n)
        if input_type == "compteur":
            Input = U(compteur) 
        else:
            Input = U(T)

        Xt = np.linalg.inv(np.eye(A.shape[0], A.shape[0]) - A*dt)

        #print(B,Input,"elv")
        #print(B,Input,np.dot(B, Input),"Elvis")
        #print(Xt, X, B, Input, np.dot(B, Input))
        X = np.dot(Xt, X + np.dot(B, Input)*dt)

        s = np.dot(C, X) + np.dot(D,Input)
        y[i] = s

        compteur += 1
        T += dt

    return y

def create_input(dim,derive_n):
    def U(compteur):
        if compteur == 0:
            return np.array([1 for i in range(derive_n+1)]+
                            [0 for i in range(dim-derive_n-1)])
        else:
            return np.array([1]+[0 for i in range(dim-1)])

    return U


def euler_implicite3(A,B,C,D,U,X0,temps,input_type="compteur"):
    """
    H: C'est la fonction de transfert en temps continue de la fonction linéaire
    temps: Le temps pour la simulation
    U: Une fonction qui renvoie l'entrée
    C: La matrice C
    X0: Les condititions initiale
    type_temps: Pour savoir si on envoie le compteur dans U ou si on envoie le temps
    """
    print(A,"\n\n", B)

    X = X0

    dt = 0
    y = []
    y.append(X0)
    compteur = 0
    T = 0
    for i in range(1, temps.shape[0]):
        dt = temps[i] - temps[i-1]

        # On doit diviser la sortie par le coefficient de y'(n)
        if input_type == "compteur":
            Input = U(compteur) 
        else:
            Input = U(T)

        Xt = np.linalg.inv(np.eye(A.shape[0], A.shape[0]) - A*dt)

        #print(B,Input,"elv")
        #print(B,Input,np.dot(B, Input),"Elvis")
        #print(Xt, X, B, Input, np.dot(B, Input))


        X = np.dot(Xt, X + np.dot(B, Input)*dt)

        #s = np.dot(C, X) + np.dot(D,Input)
        y.append(X)

        compteur += 1
        T += dt

    return np.array(y)


def test_polynome():
    a = Polynome(coeff=[1, 0, 1])
    b = Polynome(coeff=[1, 0])
    c = a + b
    d = a - b
    e = a*b
    f = Polynome(coeff=[1, -1])
    g = Polynome([1])

    # print(a,b,c,d,e)
    # print(f,g,(g([1,2,4,5])))
    g.init()
    print(g.coeff)
    print(g.matrice_commande(4))

def test_charge_decharge():
    r = 100
    c = 1e-6
    A = np.array([-1/(r*c)])
    B = np.array([1/(r*c)])
    C = np.array([1])
    D = np.array([0])

    U_ = 1
    def U(x):
        return U_
    
    pas = r*c*5

    t_final = np.array([])
    s_final = np.array([])
    X0 = 0
    for i in range(0,10):
        t = np.linspace(i*pas,pas*(i+1),10)
        #print(t)
        s = euler_implicite2(A,B,C,D,U,X0,t)
        
        X0 = s[-1]

        if((i+1) % 2):
            U_ = 1
        else:
            U_ = 0

        t_final = np.concatenate([t_final,t])
        s_final = np.concatenate([s_final,s])

    plt.plot(t_final,s_final)
    plt.show()


def test_decharge():
    r = 100
    c = 1e-6
    A = np.array([-1/(r*c)])
    B = np.array([1/(r*c)])
    C = np.array([1])
    D = np.array([0])

    U_ = 0
    X0 =  1

    def U(x):
        return U_
    
    pas = r*c*5
    t = np.linspace(0,pas,100)
    s = euler_implicite2(A,B,C,D,U,X0,t)
    plt.plot(t,s)
    plt.show()



if __name__ == "__main__":
    test_charge_decharge()