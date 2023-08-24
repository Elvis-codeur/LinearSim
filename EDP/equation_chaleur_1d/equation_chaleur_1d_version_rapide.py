import numpy as np 
import matplotlib.pyplot as plt 

NX = 100
NT = 1000
L = 1
Time = 0.1 
k = 0.5
dt = Time/NT
dx = L/(NX - 1)

x =  np.linspace(0.0,1.0,NX)
T = 0*x #np.cos(2*np.pi*x)
T[NX//2] = 1
RHS = np.zeros((NX))

for n in range(0,NT):
    RHS[1:-1] = dt*k*(T[:-2] - 2*T[1:-1] + T[2:])/(dx**2)
    T += RHS 

    if n % 100 == 0:
        plt.plot(x,T,label = "%1.2f"%(n*dt))

plt.legend()
plt.show() 
    
