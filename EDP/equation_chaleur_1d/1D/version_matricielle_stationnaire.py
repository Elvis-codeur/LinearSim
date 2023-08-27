import numpy as np 
import matplotlib.pyplot as plt 
import scipy.sparse as sp 
from scipy.sparse.linalg.dsolve import spsolve 
S = 5
NX = 100
L = 1
k = 0.5
dx = L/(NX - 1)
x = np.linspace(0,L,NX)

data = [np.ones(NX),-2*np.ones(NX),np.ones(NX)]
offsets = np.array([0,1,2])

LAP = sp.dia_matrix((data,offsets),shape = (NX,NX))
row = np.arange(0,NX,1).astype(int)
cols = np.zeros((NX)).astype(int)
print(row,cols)
LAP2 = sp.bsr_matrix((np.ones(NX),(row,cols)), shape = (NX,NX))
print((LAP).toarray())

f =  -S*np.ones(NX)*dx**2

T = spsolve(LAP,f)

plt.plot(x,T)
plt.show()