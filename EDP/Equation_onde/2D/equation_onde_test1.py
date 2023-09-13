import numpy as np 
import matplotlib.pyplot as plt 

NX = 500
NY = 500 
NT = 5000
Lx = 6
Ly = 4
dx = Lx/(NX-1)
dy = Ly/(NY-1)
Time = 0.4
dt = Time/NT
k = 0.5
c = 0.1
S = 1.0
a = 40

x = np.linspace(-Lx,Lx,NX)
y = np.linspace(-Ly,Ly,NY)
t = np.linspace(0,Time,NT)

X,Y = np.meshgrid(x,y)
T = np.zeros((NX,NY))
T = np.exp(-a*((X-1.5)**2+(Y-1.5)**2))
PreviousT = np.zeros((NX,NY))
RHS = np.zeros((NX,NY))

def code():
    PreviousT = np.zeros((NX,NY))
    for i in range(NT):
        RHS[1:-1,1:-1] = ((dt*k)**2) *(
                (T[:-2,1:-1] -2*T[1:-1,1:-1] + T[2:,1:-1])/(dx**2)
             +(T[1:-1,:-2] -2*T[1:-1,1:-1] + T[1:-1,2:])/(dx**2)
             )
        T[1:-1,1:-1] = 2*T[1:-1,1:-1] - PreviousT[1:-1,1:-1] + RHS[1:-1,1:-1] 

        PreviousT = T

        if i % 500 == 0:
            plt.imshow(T)
            plt.show()

if __name__ == "__main__":
    code()