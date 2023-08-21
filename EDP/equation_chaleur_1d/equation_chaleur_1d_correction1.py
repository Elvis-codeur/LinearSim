import numpy as np 
import matplotlib.pyplot as plt 

L = 1
Time = 0.1 
NX = 100
NT = 1000 

dx = L/(NX - 1)
dt = Time/NT
time = 1
k = 0.5
c = dt*k/(dx**2)

x = np.linspace(0.0,1.0,NX)
T = np.sin(2*np.pi*x)
rhs = np.zeros((NX))

#print(x)

for t in range(0,NT):
    for j in range(1,NX-1):
        rhs[j] = dt*k*(T[j-1] - 2*T[j] + T[j+1])/(dx**2)
        
    for j in range(1,NX-1):
        T[j] += rhs[j]

    if t % 100 == 0:
        #print(T)
        plt.plot(x,T,label = "%1.2f"%(t*dt))
    

plt.legend()
plt.show()
