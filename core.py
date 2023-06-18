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
        result = np.zeros(shape=(dim, max(dim,l)))
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
    y[0] = np.dot(X0,C)
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
        #print("\n Elvis \n")
        #print(Xt, X, B, Input)
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
    #print(A,"\n\n", B)

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


def rk4_ordre_1(model,y0,dt,T):
    time = 0.0 
    y = y0 
    t = [time]
    s = [y0]
    while time < T:
        k1 = model(time,y)
        k2 = model(time + dt/2, y + (dt/2)*k1)
        k3 = model(time + dt/2, y + (dt/2)*k2)
        k4 = model(time + dt, y + dt*k3)
        y =  y + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        time += dt 

        t.append(time)
        s.append(y)
    return np.array(t),np.array(s) 


def rk4(H: SystemeLineaire, temps,
                    U, C, X0, input_type="compteur"):
    
    A = H.Den.matriceCompagne
    # print(A)
    B = H.Num.matrice_commande(A.shape[0])

    print(A,"\n\n", B)

    X = X0

    dt = 0
    y = np.zeros_like(temps)
    y[0] = np.dot(X0,C)
    compteur = 0
    T = 0
    for i in range(1, temps.shape[0]):
        dt = temps[i] - temps[i-1]

        # On doit diviser la sortie par le coefficient de y'(n)
        if input_type == "compteur":
            Input = U(compteur) / H.Den.coeff[0]
        else:
            Input = U(T)/H.Den.coeff[0]


        K1 = np.dot(A,X) + np.dot(B,Input)
        K2 = np.dot(A,X +(dt/2)*K1)+np.dot(B,Input)
        K3 = np.dot(A,X + (dt/2)*K2)+np.dot(B,Input)
        K4 = np.dot(A,X + dt*K3)+np.dot(B,Input)

        X = (dt/6)*(K1 + 2*K2 + 2*K3 + K4) +X

        s = np.dot(C, X)
        y[i] = s

        compteur += 1
        T += dt

    return y


def mo(t,y):
    return 4*y + 9 


def test_rk1_ordre_1():
    t,s = rk4(mo,0,0.01,1)
    s2 = (9/4)*(np.exp(4*t) -1)
    plt.plot(t,s,label ="s")
    plt.plot(t,s2,label = "s2")
    plt.legend()
    plt.show()


def U(x):
    return np.array([1,0,0])


def test_rk4():
    Den = Polynome([1,1])*Polynome([1,7])*Polynome([1,3])
    Num = Polynome([27])

    print(Den,Num)

    H = SystemeLineaire(Num,Den)
    t = np.linspace(0,7,1000)
    C = np.array([1,0,0])
    X0 = np.array([0,0,0])

    s = euler_implicite(H,t,U,C,X0)
    plt.plot(t,s,label = "euleur implicite")
    
    srk4 = rk4(H,t,U,C,X0)
    plt.plot(t,srk4,label = "rk4")

    plt.legend()

    plt.show()


def get_tf_from_step_response(t,s,order:int):
    steady = s[-1]

    d_kernel = np.array([1,-1])
    dt = np.convolve(t,d_kernel,"same")
    d_s_array = [s]
    ds = s

    for i in range(order):
        ds = np.convolve(ds,d_kernel,"same")
        d_s_array.append(ds/dt)

    d_s_array = np.array(d_s_array)
    #print(d_s_array)
    #print(d_s_array.shape,s.shape)

    """
    plt.plot(t,s,label = "s")
    plt.plot(t,d_s_array[0,:],label = "s de ds_array")
    plt.plot(t,d_s_array[1,:],label = "ds")
    plt.plot(t,d_s_array[2,:],label = "dds")
    plt.plot(t,d_s_array[3,:],label = "dds")

    J = [ 1 ,57,  750, 2800]
    Js = np.zeros_like(s)
    for i in range(4):
        Js += d_s_array[3-i,:]*J[i]

    #plt.plot(t,Js,label = "recon")
    
    
    plt.legend()
    plt.show()
    """

    Y = np.ones(order+1)
    M = d_s_array[:,(order+1):2*(order+1)]
    print(M)

    #print(M.shape)
    #print(M,"\n\n",M.T)
    result = np.dot(np.linalg.inv(M.T),Y)
    return result

def test_get_tf_from_step_response():
    Den = Polynome([1,1])*Polynome([1,1])*Polynome([1,1])
    Num = Polynome([1,])

    H = SystemeLineaire(Num,Den)
    t = np.linspace(0,7,1000)
    C = np.zeros(Den.coeff.shape[0]-1)
    C[0] = 1
    X0 = np.zeros(Den.coeff.shape[0]-1)


    srk4 = rk4(H,t,U,C,X0)

    simpli = euler_implicite(H,t,U,C,X0)
    
    den_coeff = get_tf_from_step_response(t,srk4,3)
    print(Den,den_coeff)

    test_den = Polynome(den_coeff)
    test_H = SystemeLineaire(Num,test_den)
    test_sk4 = rk4(test_H,t,U,C,X0)


    plt.plot(t,srk4,label = "true")
    plt.plot(t,test_sk4,label = "guess")
    plt.legend()
    plt.show()


    """
    plt.plot(t,srk4,label = "rk4")
    plt.plot(t,simpli,label = "euleur implicite")
    plt.legend()
    print(Den)
    plt.show()

    """
    

if __name__ == "__main__":
    """
    t = np.array([i**2 for i in range(100)])
    dt = np.array([2*i for i in range(100)])
    mydt = np.convolve(t,np.array([1,-1]),"same")
    plt.plot(t,label = "t")
    plt.plot(dt,label = "dt")
    plt.plot(mydt,label = "mydt")
    plt.legend()
    plt.show()
    """
    test_get_tf_from_step_response()