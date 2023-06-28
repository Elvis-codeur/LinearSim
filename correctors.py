import numpy as np 
from numpy import linalg

from core  import Polynome,SystemeLineaire
from correctors_lib import * 

def polynomial_correction(H:SystemeLineaire,D):
    """
    H est la fonction de transfert de votre transfert de votre système actuel
    D est le dénominateur de la fonction de transfert que vous voulez avoir une fois correcteur inséré
    et le sytème bouclé. 
    Le résulat est sous la forme [Num,Den] où Num est le numérateur et Den le dénominateur 
    tous écrit sous la forme a_n, ...,a_0
    """
    Num = H.Num
    Den = H.Den
    m = get_main_matrice(Num,Den)  
    return get_correcteur(m,D)


if __name__ == "__main__":
    Num = Polynome([1])
    Den = Polynome([1,-1])*Polynome([1,1])
    D = Polynome([1,2])*Polynome([1,2])*Polynome([1,2])
    print(D)
    H = SystemeLineaire(Num,Den)
    m_matrice = get_main_matrice(Num,Den)   
    print(m_matrice)
    #print(get_correcteur(m_matrice,D))
    print(polynomial_correction(H,D))
