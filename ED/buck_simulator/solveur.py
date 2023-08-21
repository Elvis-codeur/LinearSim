import numpy as np 



def euler_implicite(A,B,C,D,U,X0,temps,input_type="compteur"):
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

