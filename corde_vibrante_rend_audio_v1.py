# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 16:19:25 2026

@author: adele
"""

import numpy as np
from scipy.io import wavfile



# =============================================================================
# Constantes 
# =============================================================================


L = 1.0          # Longueur de la corde (m)
x0 = 0.3 * L     # Point de pincement initial
h = 0.08 * L     # Hauteur du pincement (déplacement initial max)

N_modes = 5  # Nombre de modes de Fourier conservés
Nx = 500   # Nombre de points de maillage/discrétisation de l'espace
x = np.linspace(0, L, Nx)  # Abscisse des points de discrétisation 


# Amortissement
gamma = 1.5         # Coefficient d'amortissement 

# Nouvelles variables 
f = 440   # Fréquence de la note (Hz)

c = 2.0 * L * f    # Vitesse de l'onde
 

# Paramètres audio
fs = 44100          # Fréquence d'échantillonnage cd (44 100 Hz)
dt = 1 / fs       # Pas ou temps entre deux échantillons audio (~0.0000226 s)
duree = 3.5         # Durée du fichier (en secondes)
Nt = int(duree * fs)# Nombre total d'échantillons audio à produire

# =============================================================================
# Fonctions préalables 
# =============================================================================


def An(N,L,x0,h) :    
    """
    

    Parameters
    ----------
    N : Nombre de modes propres conservés.
    L : Longueur de la corde.
    x0 : Position initiale du pincement de la corde.
    h : Hauteur du pincement de la corde.

    Returns
    -------
    An : Tableau des N+1 coefficients de Fourier conservés 
    Avec à la place 0 un 0 (pas de coef de F associé).
    
    Pour un corde pincée en x0

    """
    An = np.zeros(N+1)
    
    for n in range(1,N+1) : 
        An[n] = 2 * h * L**2 / (n**2 * np.pi**2 * x0 * (L - x0)) * np.sin(n * np.pi * x0 / L)
                 
    return An 




def y(x,t,An,L,c) : 
    """
    

    Parameters
    ----------
    x : Position à laquelle est l'émetteur
    t : Temps t auquel il faut générer un tableau avec les position y(x,t)
    An : Coefs de Fourier.
    L : Longueur de la corde.
    c : Vitesse des ondes sur la corde.

    Returns
    -------
    y : Tableau de la taille de x (du maillage spatial) 
    avec pour chaque position x_i, la valeur y[i] correspondant à sa hauteur dans l'espace, selon l'axe vertical. 

    """
    N_modes = len(An) - 1   # Nombre de modes à conserver
    
    y = 0
    
    for n in range(1,N_modes) : 
        k = n * np.pi / L
        w = c * n * np.pi /L
        y += An[n] * np.sin(k * x) * np.cos(w * t)
                
    return y



# =============================================================================
# Formation du signal audio 
# =============================================================================

An = An(N_modes, L, x0, h)

# Emplacement du capteur placé à 80% de la corde
idx_micro = int(0.8 * (Nx - 1))

# Tableau avec les différentes valeurs de y(x,t) - 1D ici 
signal_audio = np.zeros(Nt)

t_audio = np.linspace(0, duree, Nt)       # Tableau des temps 

for i in range(Nt) :
    t = t_audio[i]
    signal_audio[i] = y(x[idx_micro],t,An,L,c)    # On ne demande que la valeur en le point où il y a le capteur


# On normalise le signal pour pas qu'il y est saturation (on ramène les valeurs entre -1 et 1)
maxi = np.max(np.abs(signal_audio))
if maxi > 0:
    signal_normalise = signal_audio / maxi
    
else:
    signal_normalise = signal_audio

# Conversion au format entier 16-bits (askip ce qu'il faut pour le format WAV)
donnees = np.int16(signal_normalise * 32767 * 0.95)

# Cette ligne adapte l'onde mathématique flottante [-1, 1] en un signal entier [-31128, 31128] 
# parfait pour être gravé ou lu dans un fichier .wav sans saturation




nom_fichier = "son_corde_pincee_v2.wav"
wavfile.write(nom_fichier, fs, donnees)



print(f"Fichier audio : {nom_fichier}")
print(f"Corde : {L}m | Note : {f}Hz | Fréquence d'échantillonnage : {fs} Hz")




