"""
Institution:    TU Delft
Authors:        A11
Date:           17-02-2020

This tool gives a first estimate of the structural deformation of an aileron attached to a wing at limit aerodynamic load under a certain deflection. 

Coordinate System: x' (span-wise), y'(perpendicular to x',z' from symmetry line of airfoil), z' (chord wise, LE to TE)
"""

##  =========== Library Imports: =========== 
import numpy as np
import math as m
import matplotlib.pyplot as plt

##  =========== Functions: ===========
def integrator_trap(f, x, x0):
    F = np.empty(x.shape)
    F[0] = x0
    for i in range(1,x.shape[0]):
        F[i] = F[i-1] + (x[i] - x[i-1])/2*(f[i] + f[i-1])
    return F

def integrator_cumutrap(f, x, x0):
    F = x0 + np.sum( ((x[1:] - x[:-1])/2) * (f[1:] + f[:-1]) )
    return F 

def integrator_simp(f, x):
    Flst = []
    Flst.append( ((x[2::2] - x[:-2:2])/6) * (f[2::2] + 4*(f[1:-1:2]) + f[:-2:2]) )
    Farr = np.array(Flst)
    F = np.sum(Farr)
    return Farr, F

def integrator_cumusimp(f, x):
    F = np.sum( ((x[1:] - x[:-1])/6) * (f[1:] + 2*(f[1:]+f[:-1]) + f[:-1]) )
    return F

def st_locations(C_a, h_a, n_st):
    """
    S - Actual perimeter
    s - perimeter position (starting from TE counter-clockwise all the way back to TE again)

    The idea behind this is that we want to "wrap" a straight line around the aileron. This straight line has all the points of the stiffeners on it. The wrapping starts from the top of the TE, goes around the LE, and returns at the bottom of the TE.
    """

    ## =========== Parameters of Function ===========
    S_circ  = np.pi*h_a/2                                                       ## Half-Arc Perimeter
    S_tail  = m.sqrt( (C_a - 0.5*h_a)*(C_a - 0.5*h_a) + (0.5*h_a)*(0.5*h_a) )   ## Tail Diagonal
    S       = S_circ + 2*S_tail                                                 ## Total Perimeter
    alpha_tail  = m.asin( (h_a/2)/S_tail )

    ## Creating straight line and all stiffener locations on it, we wrap the line from the TE (top) to the LE, and back to TE (bottom)
    s       = np.linspace(0,S,n_st+2)[1:-1]                                     ## Ignore TE end points    
    ## Finding all the stiffeners on the top / bottom part of the tail
    tail    = s[s<S_tail]
    ## Finding all the stiffener locations on the arc. Normalized such that the points start from 0 on the arc at the top.
    circ    = s[s>S_tail] - S_tail
    circ    = circ[circ<S_circ]
    ts      = tail.shape[0]
    cs      = circ.shape[0]
    
    YZ              = np.zeros((3, n_st))                                       ## [(Z,Y,Beta),(0,1,...,n_st-1)]
    ## Top Tail
    YZ[0,:ts]       = C_a - tail * m.cos(alpha_tail)
    YZ[1,:ts]       = tail * m.sin(alpha_tail)
    YZ[2,:ts]       = alpha_tail
    ## LE arc
    YZ[0,ts:ts+cs]  = h_a/2 * (1 - np.sin(circ/(h_a/2)) )                       ## angle = s/(2 pi r) * 360 * pi/180 = s/(2 pi r) * 2 pi = s / r
    YZ[1,ts:ts+cs]  = h_a/2 * np.cos(circ/(h_a/2))
    YZ[2,ts:ts+cs]  = 2*np.pi-circ/(h_a/2)
    ## Bottom Tail
    YZ[0,ts+cs:]    = C_a - tail[::-1] * m.cos(alpha_tail)
    YZ[1,ts+cs:]    = -tail[::-1] * m.sin(alpha_tail)
    YZ[2,ts+cs:]    = 2*np.pi-alpha_tail

    return YZ

def McCauley(x,x0):
    return max(x-x0,0)

##  =========== Input Parameters: =========== 
C_a     = 0.515 ## aileron chord                                [m]
l_a     = 2.691 ## span of aileron                              [m]
x_1     = 0.174 ## x-location of hinge 1                        [m]
x_2     = 1.051 ## x-location of hinge 2                        [m]
x_3     = 2.512 ## x-location of hinge 3                        [m]
x_a     = 30    ## distance between Actuator I and Actuator II  [cm]
h_a     = 24.8  ## aileron height                               [cm]
t_sk    = 1.1   ## skin thickness                               [mm]
t_sp    = 2.2   ## spar thickness                               [mm]
t_st    = 1.2   ## stiffener thickness                          [mm]
h_st    = 1.5   ## stiffener height                             [cm]
w_st    = 3.0   ## stiffener width                              [cm]
n_st    = 11    ## number of stiffeners                         [-]
d_1     = 1.034 ## vertical displacement hinge 1                [cm]
d_3     = 2.066 ## vertical displacement hinge 3                [cm]
phi     = 25    ## aileron deflection                           [deg]
P       = 20.6  ## actuator load                                [kN]
E       = 73.1  ## material Young's modulus                     [GPa]
G       = 28.0  ## material shear moduus                        [GPa]
rho     = 2780  ## material density                             [kg m^-3]

##  =========== Switch to SI: =========== 
x_a     /= 100                              ##[cm -> m]
h_a     /= 100                              ##[cm -> m]
h_st    /= 100                              ##[cm -> m]
w_st    /= 100                              ##[cm -> m]
d_1     /= 100                              ##[cm -> m]
d_3     /= 100                              ##[cm -> m]
t_sk    /= 1000                             ##[mm -> m]
t_sp    /= 1000                             ##[mm -> m]
t_st    /= 1000                             ##[mm -> m]
E       *= 1000000000                       ##[GPa -> Pa]
G       *= 1000000000                       ##[GPa -> Pa]
phi     *= np.pi/180                        ##[deg -> rad]
P       *= 1000                             ##[kN -> N]

## =========== Import Aero Data: ===========
AeroData = np.loadtxt("aerodynamicloaddo228.dat", delimiter=",")

## =========== Cross Section Properties:  =========== 
ZY_stif     = st_locations(C_a, h_a, n_st)      ## [Z,Y,beta] of all stiffeners along airfoil
l_tail      = m.sqrt( (C_a - 0.5*h_a)*(C_a - 0.5*h_a) + (0.5*h_a)*(0.5*h_a) )
alpha_tail  = m.asin( (0.5*h_a)/l_tail )
beta_stif   = ZY_stif[2]                        ## Angles of all stiffeners along airfoil

## Areas
A_circ = np.pi*(0.5*h_a)*t_sk                   ## Area of arc
A_spar = h_a*t_sp                               ## Area of spar
A_tail = l_tail*t_sk                            ## Area of single tail piece
A_stif = (h_st-t_st)*t_st + w_st*t_st           ## Area of single stiffener

## Centroids
C_circ      = 0.5*h_a - 2*(0.5*h_a)/np.pi                       ## Centroid of semi-circle from leading edge
C_spar      = 0.5*h_a + 0.5*t_sp                                ## Centroid of spar from leading edge
C_tailz     = 0.5*h_a + t_sp + 0.5*(C_a - 0.5*h_a)              ## Centroid of tail sections in z' direction from leading edge
C_taily     = 0.5*l_tail*m.sin(alpha_tail)                      ## Centroid of tail sections in y' direction from symmetry line (absolute value)
C_stifz     = ZY_stif[0]
C_stify     = ZY_stif[1]
C_airfoil   = (C_circ*A_circ + C_spar*A_spar + 2*A_tail*C_tailz + np.sum(C_stifz)*A_stif)/(A_circ+A_spar+2*A_tail + n_st*A_stif)

## Inertias
Izz_circ        = 0.125*np.pi*( (0.5*h_a)**4 - (0.5*h_a - t_sk)**4) - 4/np.pi * (0.5*h_a)**3 * t_sk
Izz_circ_stein  = A_circ*(C_circ-C_airfoil)*(C_circ-C_airfoil)
Iyy_circ        = 0.125*np.pi*( (0.5*h_a)**4 - (0.5*h_a - t_sk)**4)
## Iyy_circ_stein = 0

## Izz_spar = 0
Izz_spar_stein  = A_spar*(C_spar-C_airfoil)*(C_spar-C_airfoil)
Iyy_spar        = 1/12 * h_a**3 * t_sp
## Iyy_spar_stein = 0

Izz_tail        = 1/12*l_tail**3*t_sk*m.cos(alpha_tail)**2
Izz_tail_stein  = A_tail*(C_tailz-C_airfoil)*(C_tailz-C_airfoil)
Iyy_tail        = 1/12*l_tail**3*t_sk*m.sin(alpha_tail)**2
Iyy_tail_stein  = A_tail*(C_taily-C_airfoil)*(C_taily-C_airfoil)
  
## MOMENT OF INERTIA OF STIFFENERS STILL NEED TO BE PLACED IN
Izz_stif        = 1/12 * w_st**3*t_st*np.cos(-beta_stif)*np.cos(beta_stif) + 1/12 * (h_st-t_st)**3*t_st*np.sin(-beta_stif)*np.sin(beta_stif)
Izz_stif_stein  = A_stif*(C_stifz - C_airfoil)*(C_stifz - C_airfoil)
Iyy_stif        = 1/12 * w_st**3*t_st*np.sin(-beta_stif)*np.sin(beta_stif) + 1/12 * (h_st-t_st)**3*t_st*np.cos(-beta_stif)*np.cos(beta_stif)
Iyy_stif_stein  = A_stif*(C_stify - C_airfoil)*(C_stify - C_airfoil)

Izz_airfoil     = Izz_circ  + 2*Izz_tail    + Izz_circ_stein    + np.sum(Izz_stif)  + Izz_spar_stein    + 2*Izz_tail_stein      + np.sum(Izz_stif_stein)
Iyy_airfoil     = Iyy_circ  + Iyy_spar      + np.sum(Iyy_stif)  + 2*Iyy_tail        + 2*Iyy_tail_stein  + np.sum(Iyy_stif_stein)

#print(C_circ, A_circ)
#print(C_spar, A_spar)
#print(A_tail, C_tailz)
#print(C_airfoil)
#
#print(Izz_circ)
print(Izz_airfoil, Iyy_airfoil)
print(beta_stif)

plt.plot(C_stifz,C_stify,marker="o")
plt.show()

