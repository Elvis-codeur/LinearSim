import numpy as np 
import matplotlib.pyplot as plt 

dx = 1e-3
dt = 1e-1
time = 1
k = 0.5
c = dt*k/(dx**2)

x = np.arange(0,1,dx)
T = np.sin(2*np.pi*x)

MT = np.zeros((int(time/dt),x.shape[0]))# Chaque ligne repésente la température à un temps précis
MT[0,:] = T
for t_pos in range(1,int(time/dt)):
    for x_pos in range(1,x.shape[0]-1):
        MT[t_pos,x_pos] = MT[t_pos-1,x_pos] +\
              c*(MT[t_pos-1,x_pos-1] -2*MT[t_pos-1,x_pos] + MT[t_pos-1,x_pos+1])


if __name__ == "__main__":
    t = 0
    for t_pos in range(0,3):
        plt.plot(x,MT[t_pos,:],label =  "{}".format(t))
        t += dt 

    plt.legend()
    plt.show()