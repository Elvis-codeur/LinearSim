import numpy as np 
from numpy import linalg

from core  import Polynome,SystemeLineaire
from correctors_lib import * 

"""
This corrector is inpired from 

https://en.wikibooks.org/wiki/Control_Systems/Polynomial_Design
"""

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
    Num, Den = get_correcteur(m,D)
    return SystemeLineaire(Polynome(Num),Polynome(Den))


def feedback(H:SystemeLineaire):
    return SystemeLineaire(H.Num,H.Num + H.Den)

def test_1():
    Num = Polynome([23,90])
    Den = Polynome([1,-5,10])
    D = Polynome([1,2])*Polynome([1,3])*Polynome([1,4])
    print(D)
    H = SystemeLineaire(Num,Den)
    m_matrice = get_main_matrice(Num,Den)   
    print(m_matrice)
    #print(get_correcteur(m_matrice,D))
    C =  polynomial_correction(H,D)
    CH = C*H
    Hfinal = feedback(CH)
    print(Hfinal.Den.roots)
    print(C)
if __name__ == "__main__":
    test_1()
