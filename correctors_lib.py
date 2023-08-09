import numpy as np 
from numpy import linalg

from core  import Polynome,SystemeLineaire


def get_main_matrice(SysNum:Polynome,SysDen:Polynome):
    n = SysDen.coeff.shape[0] -1
    m = n - 1 
    result = np.zeros(shape =(n+m+1,2*m + 2))

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