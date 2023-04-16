import matplotlib.pyplot as plt 
import numpy as np 
from core import SystemeLineaire
def impulse():
    a = 0

def step():
    a = 0

def bode(H:SystemeLineaire,freq_debut=0.001,freq_fin=100,ppd=50):
    """
    freq_debut: Fréquence de début
    freq_fin: Fréqunce à laquelle arrêter le diagramme
    ppd: Le nombre de point par décade
    """
    # On génère l'echelle frequencielle
    x = np.logspace(np.log10(freq_debut),np.log10(freq_fin),num=int(ppd*(np.log10(freq_fin/freq_debut))))
    # On le transforme en nombre complexe 
    # On evalue avec le polynome 
    y = H(np.emath.sqrt(-1)*x*2*np.pi)

    #On dessine
    fig,axes = plt.subplots(2,1)
    axes[0].set_xscale("log")
    axes[0].plot(x,20*np.log10(np.abs(y)))
    axes[0].grid(True,"both")

    axes[1].set_xscale("log")
    axes[1].plot(x,np.angle(y)*180/np.pi)
    axes[1].grid(True,"both")

    plt.show()


def nyquist(H:SystemeLineaire,freq_debut=0.001,freq_fin=100,ppd=50):
    # On génère l'echelle frequencielle
    x = np.logspace(np.log10(freq_debut),np.log10(freq_fin),num=int(ppd*(np.log10(freq_fin/freq_debut))))
    # On le transforme en nombre complexe 
    # On evalue avec le polynome 
    y = H(np.emath.sqrt(-1)*x*2*np.pi)

    fig,ax = plt.subplots()
    real = np.real(y)
    imag = np.imag(y)
    ax.plot(real,imag)

    step = x.shape[0]//10
    for i in range(1,step):
        print(i)
        indice = i*10
        ax.arrow(real[indice-1], imag[indice-1],
                 real[indice] -real[indice-1],
                 imag[indice]-imag[indice-1],
                 shape='full', lw=0, length_includes_head=True,
                 head_width=.01,color="r")
    
    ax.grid(True,"both",linewidth=0.8)
    plt.show()

def nichols(H:SystemeLineaire,freq_debut=0.001,freq_fin=100,ppd=50):
    # On génère l'echelle frequencielle
    x = np.logspace(np.log10(freq_debut),
                    np.log10(freq_fin),
                    num=int(ppd*(np.log10(freq_fin/freq_debut))))
    # On le transforme en nombre complexe 
    # On evalue avec le polynome 
    y = H(np.emath.sqrt(-1)*x*2*np.pi)

    fig,ax = plt.subplots()
    angle = np.angle(y)*180/np.pi
    moduleDB = 20*np.log10(np.abs(y))
    ax.plot(angle,moduleDB)

    step = x.shape[0]//10
    for i in range(1,step):
        print(i, " Elvis")
        indice = i*10
        ax.arrow(angle[indice-1], moduleDB[indice-1],
                 angle[indice] -angle[indice-1],
                 moduleDB[indice]-moduleDB[indice-1],
                 shape='full', lw=0, length_includes_head=True,
                 head_width=.01,color="r")
    
    ax.grid(True,"both",linewidth=0.8)
    plt.show()