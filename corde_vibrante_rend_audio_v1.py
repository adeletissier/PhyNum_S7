# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 16:19:25 2026

@author: adele
"""

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation


# =============================================================================
# Constantes 
# =============================================================================


L = 1.0          # Longueur de la corde (m)
c = 1.0          # Célérité de l'onde sur la corde (m/s)
x0 = 0.3 * L     # Point de pincement initial
h = 0.08 * L     # Hauteur du pincement (déplacement initial max)

N_modes = 10  # Nombre de modes de Fourier conservés
N_x = 500      # Nombre de points de maillage/discrétisation de l'espace
x = np.linspace(0, L, N_x)  # Abscisse des points de discrétisation 

# Période du mode fondamental pour n = 1
T1 = 2.0 * L / c  # La période général pour un mode n est : 2 * L / (c * n)

# Amortissement
gamma = 1.5         # Coefficient d'amortissement (dissipation du son dans l'air)

# Nouvelles variables 
f = 440   # fréquence de la note 



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
    x : Tableau du maillage spatial.
    t : Temps t auquel il faut générer un tableau avec les position y(x,t)
    An : Coefs de Fourier.
    L : Longueur de la corde.
    c : Vitesse des ondes sur la corde.

    Returns
    -------
    y : Tableau de la taille de x (du maillage spatial) 
    avec pour chaque position x_i, la valeur y[i] correspondant à sa hauteur dans l'espace, selon l'axe vertical. 

    """
    Nx = len(x)   # Taille du maillage spacial
    N_modes = len(An) - 1   # Nombre de modes à conserver
    
    y = np.zeros(Nx)
    
    for i in range(Nx) : 
            for n in range(1,N_modes) : 
                k = n * np.pi / L
                w = c * n * np.pi /L
                y[i]  += An[n] * np.sin(k * x[i]) * np.cos(w * t)
                
    return y
