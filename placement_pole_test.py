import numpy as np 
from numpy import linalg

from core  import Polynome,SystemeLineaire


def get_main_matrice(SysNum:Polynome,SysDen:Polynome):
    n = SysDen.coeff.shape[0] -1
    m = n - 1 
    result = np.zeros(shape =(n+m+1,2*m + 2),dtype="complex_")

    Num_coeff = np.append(np.zeros((SysDen.coeff.shape[0] - SysNum.coeff.shape[0],)),[SysNum.coeff,]
        )
    Den_coeff = np.flip(SysDen.coeff,0)
    Num_coeff = np.flip(Num_coeff,0)

    #print(Num_coeff,Den_coeff,sep="\n")
    #print(result)   
    for i in range(n):
        result[i:n+i+1,2*i] = Den_coeff
        result[i:n+i+1,2*i+1] = Num_coeff
    #print(result)
    return result

def get_correcteur(main_matrice,desired_response:Polynome):
    #print(main_matrice.shape,desired_response.coeff.shape)
    if(np.linalg.det(main_matrice) != 0):
        result = np.dot(linalg.inv(main_matrice),np.flip(desired_response.coeff,0))
        return np.flip(result[1::2]),np.flip(result[0::2])
    else:
        raise ValueError("Discriminant nul")
    

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


def test_final_placement_pole():
    Num = Polynome([1])
    Den = Polynome([1,-1])*Polynome([1,1])
    D = Polynome([1,2])*Polynome([1,2])*Polynome([1,2])
    print(D)
    H = SystemeLineaire(Num,Den)
    m_matrice = get_main_matrice(Num,Den)   
    print(m_matrice)
    #print(get_correcteur(m_matrice,D))
    print(polynomial_correction(H,D))

def test_pid_corrector():
    R = 10
    L = 100e-6
    C = 500e-6
    Vi = 12
    Num = Polynome([Vi/L*C])
    Den = Polynome([1,1/(R*C),1/(L*C)])

    D = Polynome([1,-2.5e4])*Polynome([1,complex(-0.1e4,3100)])*Polynome([1,complex(-0.1e4,-3100)])
    H = SystemeLineaire(Num,Den)
    Num,Den = polynomial_correction(H,D)
    print(Num,Den)

if __name__ == "__main__":
    test_pid_corrector()
