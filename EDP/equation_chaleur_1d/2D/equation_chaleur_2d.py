import numpy as np 
import matplotlib.pyplot as plt 

NX = 50
NY = 50 
NT = 5000
Lx = 1
Ly = 1
dx = Lx/(NX-1)
dy = Ly/(NY-1)
Time = 0.4
dt = Time/NT
k = 0.5
S = 1.0

x = np.linspace(-Lx,Lx,NX)
y = np.linspace(-Ly,Ly,NY)
t = np.linspace(0,Time,NT)

X,Y = np.meshgrid(x,y)
T = np.zeros((NX,NY))
RHS = np.zeros((NX,NY))


def code():
   for i in range(NT):
    RHS[1:-1,1:-1] =  dt*k*( (T[:-2,1:-1] -2*T[1:-1,1:-1] + T[2:,1:-1])/(dx**2) + \
                               (T[1:-1,:-2] -2*T[1:-1,1:-1]+T[1:-1,2:])/(dy**2))
    
    T[1:-1,1:-1] += (RHS[1:-1,1:-1] + dt*S)

    if i%500 == 0:
       plt.imshow(T)
       plt.title("t = {}".format(t[i]))
       plt.show()
    

if __name__ == "__main__":
    code()
