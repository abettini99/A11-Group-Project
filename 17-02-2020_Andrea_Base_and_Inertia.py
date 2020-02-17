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

##  =========== Functions: =========== 
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

## =========== Cross Section Properties:  =========== 
l_tail      = (C_a - 0.5*h_a)*(C_a - 0.5*h_a) + (0.5*h_a)*(0.5*h_a)
alpha_tail  = m.asin( (0.5*h_a)/l_tail )

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
## CENTROID OF STIFFENER NEEDS TO BE PLACED IN
C_airfoil   = (C_circ*A_circ + C_spar*A_spar + 2*A_tail*C_tailz)/(A_circ+A_spar+2*A_tail)

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

Izz_airfoil     = Izz_circ + 2*Izz_tail + Izz_circ_stein + Izz_spar_stein + 2*Izz_tail_stein
Iyy_airfoil     = Iyy_circ + Iyy_spar + 2*Iyy_tail + 2*Iyy_tail_stein

print(C_circ, A_circ)
print(C_spar, A_spar)
print(A_tail, C_tailz)
print(C_airfoil)

print(Izz_circ)
print(Izz_airfoil, Iyy_airfoil)
