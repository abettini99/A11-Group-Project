"""
Institution:    TU Delft
Authors:        A11
Date:           17-02-2020

This tool gives a first estimate of the structural deformation of an aileron attached to a wing at limit aerodynamic load under a certain deflection. 

Coordinate System: x' (span-wise), y'(perpendicular to x',z' from symmetry line of airfoil), z' (chord wise, LE to TE)
"""

##  =========== Library Imports: =========== 
import numpy as np
import scipy.linalg as sp
import math as m
import matplotlib.pyplot as plt

##  =========== Functions: ===========
def linear_interpolate(f,x,xk):
    """
    == inputs:
    f   -- np.array of function values in "y" direction
    x   -- np.array of x locations of functions values in "y" direction
    xk  -- np.array of all x locations to get an interpolation
    == outputs:
    s   -- np.array of all interpolated y locations
 
    interpolate a given array at some location X = xk[i]. idx is index of lower bound, with the lower bound value given by:
    ## lower_bound = f[x <= X][-1]     (= f[idx])
    ## upper_bound = f[idx + 1]
    """

    s = np.empty(xk.shape)
    for i in range(xk.shape[0]-1):
        idx         = np.where(f == f[x <= xk[i]][-1]) [0][0]
        s[i]        = f[idx] + (f[idx+1] - f[idx])/(x[idx+1] - x[idx]) * (xk[i] - x[idx])
    s[-1] = f[-1]
    return s
#
#y = np.array([2,3,4,5,6,7])
#x = np.array([1,2,3,4,5,6])
#xk = np.linspace(1,6,2000)
#a = linear_interpolate(y,x,xk)
#
#plt.plot(xk,a)
#plt.plot(x,y,marker = 'o', linestyle="none")
#plt.show()
#
def integrator_trap(f, x, x0):
    """
    == inputs:
    f   -- np.array of function values in "y" direction
    x   -- np.array of x locations of functions values in "y" direction
    x0  -- integration value at x0
    == outputs:
    F   -- np.array of all the integration values of "f" up to the respective "x" value
 
    Integrate a given array along the x array using the Trapezoid Rule. Each output F[i] corresponds to the integration from x0 to x[i]. 
    """

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
    == inputs:
    C_a -- Aileron Chord Length
    h_a -- Aileron Height
    n_st-- Number of Stiffeners
    == ouputs:
    YZ  -- [(Z,Y,Beta),(0,1,...,n_st-1)] positions (and angles) of the stiffeners along the aileron
    == intermediates:
    S - Actual perimeter
    s - perimeter position (starting from TE counter-clockwise all the way back to TE again)

    The idea behind this is that we want to "wrap" a straight line around the aileron. This straight line has all the points of the stiffeners on it. The wrapping starts from the top of the LE, goes around the TE, and returns at the bottom of the LE. ONLY WORKS WITH ODD NUMBERS OF STIFFENERS
    """

    ## =========== Parameters of Function ===========
    S_circ  = np.pi*h_a/2                                                       ## Half-Arc Perimeter
    S_tail  = m.sqrt( (C_a - 0.5*h_a)*(C_a - 0.5*h_a) + (0.5*h_a)*(0.5*h_a) )   ## Tail Diagonal
    S       = S_circ + 2*S_tail                                                 ## Total Perimeter
    alpha_tail  = m.asin( (h_a/2)/S_tail )

    ## =========== Main Function ===========
    ## Creating straight line and all stiffener locations on it, we wrap the line from the TE (top) to the LE, and back to TE (bottom)
    s       = np.linspace(0,S,n_st+1)[1:]                                       ## Ignore point on LE   
    ## Finding all the stiffeners on the top / bottom part of the tail
    circ    = s[s<S_circ/2]
    tail    = s[s>S_circ/2]
    tail    = tail - S_circ/2
    tail    = tail[tail<S_tail]
    ## Finding all the stiffener locations on the arc. Normalized such that the points start from 0 on the arc at the top.
    ts      = tail.shape[0]
    cs      = circ.shape[0]
    
    
    YZ              = np.zeros((3, n_st))                                       ## [(Z,Y,Beta),(0,1,...,n_st-1)]
    YZ[0,0]         = 0
    YZ[1,0]         = 0
    YZ[2,0]         = 2*np.pi - np.pi/2
    ## LE top half arc
    YZ[0,1:cs+1]  = h_a/2 * (1 - np.cos(circ/(h_a/2)) )                       ## angle = s/(2 pi r) * 360 * pi/180 = s/(2 pi r) * 2 pi = s / r
    YZ[1,1:cs+1]  = h_a/2 * np.sin(circ/(h_a/2))
    YZ[2,1:cs+1]  = 2*np.pi-(np.pi/2 - circ/(h_a/2) )

    ## Top Tail
    YZ[0,cs+1:ts+cs+1]       = C_a - (S_tail-tail) * m.cos(alpha_tail)
    YZ[1,cs+1:ts+cs+1]       = (S_tail-tail) * m.sin(alpha_tail)
    YZ[2,cs+1:ts+cs+1]       = alpha_tail

    ## Bottom Tail
    YZ[0,ts+cs+1:2*ts+cs+1:]    = C_a - (S_tail-tail)[::-1] * m.cos(alpha_tail)
    YZ[1,ts+cs+1:2*ts+cs+1:]    = -(S_tail-tail)[::-1] * m.sin(alpha_tail)
    YZ[2,ts+cs+1:2*ts+cs+1:]    = np.pi/2 + alpha_tail

    ## LE bottom half arc
    YZ[0,2*ts+cs+1:]  = h_a/2 * (1 - np.cos(circ/(h_a/2)) )                       ## angle = s/(2 pi r) * 360 * pi/180 = s/(2 pi r) * 2 pi = s / r
    YZ[1,2*ts+cs+1:]  = -h_a/2 * np.sin(circ/(h_a/2))
    YZ[2,2*ts+cs+1:]  = np.pi + circ/(h_a/2)

    return YZ

def McCauley(x,x0):
    return np.maximum(x-x0,[0]*x.shape[0])

def McCauley0(x,x0):
    return np.maximum(x-x0, np.zeros(x.shape))/(x-x0)

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
dx      = 1e-3  ## step size                                    [m]
#
## B737
#C_a     = 0.605 ## aileron chord                                [m]
#l_a     = 2.661 ## span of aileron                              [m]
#x_1     = 0.172 ## x-location of hinge 1                        [m]
#x_2     = 1.211 ## x-location of hinge 2                        [m]
#x_3     = 2.591 ## x-location of hinge 3                        [m]
#x_a     = 35    ## distance between Actuator I and Actuator II  [cm]
#h_a     = 20.5  ## aileron height                               [cm]
#t_sk    = 1.1   ## skin thickness                               [mm]
#t_sp    = 2.8   ## spar thickness                               [mm]
#t_st    = 1.2   ## stiffener thickness                          [mm]
#h_st    = 1.6   ## stiffener height                             [cm]
#w_st    = 1.9   ## stiffener width                              [cm]
#n_st    = 15    ## number of stiffeners                         [-]
#d_1     = 1.154 ## vertical displacement hinge 1                [cm]
#d_3     = 1.840 ## vertical displacement hinge 3                [cm]
#phi     = 28    ## aileron deflection                           [deg]
#P       = 97.4  ## actuator load                                [kN]
#E       = 73.1  ## material Young's modulus                     [GPa]
#G       = 28.0  ## material shear moduus                        [GPa]
#rho     = 2780  ## material density                             [kg m^-3]
#dx      = 1e-3  ## step size                                    [m]

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
AeroDataLoad    = np.loadtxt("aerodynamicloaddo228.dat", delimiter=",")     ##[kN m^-1]
theta_zi        = np.arange(1,82+1,1)/81*m.pi
theta_xi        = np.arange(1,42+1,1)/41*m.pi
AeroDataZ       = C_a - 0.5*( 0.5*C_a*(1-np.cos(theta_zi[:-1])) + 0.5*C_a/2*(1-np.cos(theta_zi[1:])) )
AeroDataX       = 0.5*( 0.5*l_a*(1-np.cos(theta_xi[:-1])) + 0.5*l_a/2*(1-np.cos(theta_xi[1:])) )

## =========== Cross Section Properties:  =========== 
ZY_stif     = st_locations(C_a, h_a, n_st)      ## [(Z,Y,Beta),(0,1,...,n_st-1)] of all stiffeners along airfoil
l_tail      = m.sqrt( (C_a - 0.5*h_a)*(C_a - 0.5*h_a) + (0.5*h_a)*(0.5*h_a) )
alpha_tail  = m.asin( (0.5*h_a)/l_tail )
beta_stif   = ZY_stif[2]                        ## Angles of all stiffeners along airfoil

## Areas
A_circ = np.pi*(0.5*h_a)*t_sk                   ## Area of arc
A_spar = h_a*t_sp                               ## Area of spar
A_tail = l_tail*t_sk                            ## Area of single tail piece
A_stif = (h_st-t_st)*t_st + w_st*t_st           ## Area of single stiffener

AA_circ = 0.5 * np.pi * (h_a/2)*(h_a/2)         ## Internal Area of arc
AA_tail = 0.5 * h_a * (C_a - h_a/2)             ## Internal Area of both triangles

## Centroids
C_circ      = 0.5*h_a - 2*(0.5*h_a)/np.pi                       ## Centroid of semi-circle from leading edge
C_spar      = 0.5*h_a + 0.5*t_sp                                ## Centroid of spar from leading edge
C_tailz     = 0.5*h_a + t_sp + 0.5*l_tail*m.cos(alpha_tail)     ## Centroid of tail sections in z' direction from leading edge
C_taily     = 0.5*l_tail*m.sin(alpha_tail)                      ## Centroid of tail sections in y' direction from symmetry line (absolute value)
C_stifz     = ZY_stif[0]
C_stify     = ZY_stif[1]
C_airfoil   = (C_circ*A_circ + C_spar*A_spar + 2*A_tail*C_tailz + np.sum(C_stifz)*A_stif)/(A_circ+A_spar+2*A_tail + n_st*A_stif)

## Inertias
Iyy_circ        = 0.125*np.pi*( (0.5*h_a)**4 - (0.5*h_a - t_sk)**4) - 4/np.pi * (0.5*h_a)**3 * t_sk
Iyy_circ_stein  = A_circ*(C_circ-C_airfoil)*(C_circ-C_airfoil)
Izz_circ        = 0.125*np.pi*( (0.5*h_a)**4 - (0.5*h_a - t_sk)**4)
## Izz_circ_stein = 0

## Iyy_spar = 0
Iyy_spar_stein  = A_spar*(C_spar-C_airfoil)*(C_spar-C_airfoil)
Izz_spar        = 1/12 * h_a**3 * t_sp
## Izz_spar_stein = 0

Iyy_tail        = 1/12*l_tail**3*t_sk*m.cos(alpha_tail)**2
Iyy_tail_stein  = A_tail*(C_tailz-C_airfoil)*(C_tailz-C_airfoil)
Izz_tail        = 1/12*l_tail**3*t_sk*m.sin(alpha_tail)**2
Izz_tail_stein  = A_tail*(C_taily)*(C_taily)
  
Iyy_stif        = 1/12 * w_st**3*t_st*np.cos(-beta_stif)*np.cos(beta_stif) + 1/12 * (h_st-t_st)**3*t_st*np.sin(-beta_stif)*np.sin(beta_stif)
Iyy_stif_stein  = A_stif*(C_stifz - C_airfoil)*(C_stifz - C_airfoil)
Izz_stif        = 1/12 * w_st**3*t_st*np.sin(-beta_stif)*np.sin(beta_stif) + 1/12 * (h_st-t_st)**3*t_st*np.cos(-beta_stif)*np.cos(beta_stif)
Izz_stif_stein  = A_stif*(C_stify)*(C_stify)

Iyy_airfoil     = Iyy_circ  + 2*Iyy_tail    + Iyy_circ_stein    + np.sum(Iyy_stif)  + Iyy_spar_stein    + 2*Iyy_tail_stein      + np.sum(Iyy_stif_stein)
Izz_airfoil     = Izz_circ  + Izz_spar      + np.sum(Izz_stif)  + 2*Izz_tail        + 2*Izz_tail_stein  + np.sum(Izz_stif_stein)

## =========== Shear Center:  =========== 
## Define some coefficients and constants used
Coeffz  = -1/Izz_airfoil 
Coeffy  = -1/Iyy_airfoil
s_reso  = 1000

## Recall [(Z,Y,Beta),(0,1,...,n_st-1)] of all stiffeners along airfoil, we want to find all stiffeners on LE or TE, and on Bottom or Top
circ_stify = ZY_stif[1, ZY_stif[0,:] < h_a/2]    ## all stiffeners in the semi-circle -- The Y value returned
tail_stify = ZY_stif[1, ZY_stif[0,:] > h_a/2]    ## all stiffeners in the tail
topcirc_stify = circ_stify[circ_stify>0]
botcirc_stify = circ_stify[circ_stify<0]
toptail_stify = tail_stify[tail_stify>0]
bottail_stify = tail_stify[tail_stify<0]

## Create Curve Space
s_qcirc = np.linspace(0,np.pi/2*h_a/2,s_reso)     ## Quarter Circle
s_spar = np.linspace(0,h_a/2,s_reso)              ## Spar
s_tail = np.linspace(0,l_tail,s_reso)             ## Single Tail

## Create list of all z,y,s list for integrals --> play around with plotting s[0] vs s[1] and plotting s[0][:400] to s[1][:400]
s0 = [h_a/2 - h_a/2*np.cos(s_qcirc/(h_a/2)) - C_airfoil                                         , h_a/2*np.sin(s_qcirc/(h_a/2))             , s_qcirc]
s1 = [(h_a/2 + t_sp - C_airfoil)*np.ones((s_reso))                                              , s_spar                                    , s_spar]
s2 = [h_a/2 + t_sp + ( l_tail*m.cos(alpha_tail) - s_tail[::-1]*m.cos(alpha_tail) ) - C_airfoil  , s_tail[::-1]*m.sin(alpha_tail)            , s_tail]
s3 = [h_a/2 + t_sp + ( l_tail*m.cos(alpha_tail) - s_tail*m.cos(alpha_tail) ) - C_airfoil        , -s_tail*m.sin(alpha_tail)                 , s_tail]
s4 = [(h_a/2 + t_sp - C_airfoil)*np.ones((s_reso))                                              , -s_spar                                   , s_spar]
s5 = [h_a/2 - h_a/2*np.cos(s_qcirc[::-1]/(h_a/2)) - C_airfoil                                   , -h_a/2*np.sin(s_qcirc[::-1]/(h_a/2))      , s_qcirc]

## Create stiffener influences
y0 = np.zeros((s_reso))
y2 = np.zeros((s_reso))
y3 = np.zeros((s_reso))
y5 = np.zeros((s_reso))
for stif in topcirc_stify:
    y0 += McCauley0(h_a/2*np.sin(s_qcirc/(h_a/2)), stif)*stif     ## this flips it, trust me --> [1,1,1,1,0,0,0,0,0] we want to flip it -->  -1 -> [0,0,0,0,-1,-1,-1,-1] --> *(-1) -> [0,0,0,0,1,1,1,1]
for stif in toptail_stify:
    y2 += (McCauley0(s_tail[::-1]*m.sin(alpha_tail), stif) -1)*(-1)*stif
for stif in bottail_stify:
    y3 += (McCauley0(-s_tail*m.sin(alpha_tail), stif) -1)*(-1)*stif
for stif in botcirc_stify:
    y5 += McCauley0(-h_a/2*np.sin(s_qcirc[::-1]/(h_a/2)), stif)*stif

#plt.plot(s_qcirc, y0)
#plt.plot(s_tail, y2)
#plt.plot(s_tail, y3)
#plt.plot(s_qcirc, y5)
#plt.show()

#plt.plot(s0[0][:400],s0[1][:400])
#plt.plot(s1[0][:400],s1[1][:400])
#plt.plot(s2[0][:400],s2[1][:400])
#plt.plot(s3[0][:400],s3[1][:400])
#plt.plot(s4[0][:400],s4[1][:400])
#plt.plot(s5[0][:400],s5[1][:400])
#plt.plot(s0[0],s0[1])
#plt.plot(s1[0],s1[1])
#plt.plot(s2[0],s2[1])
#plt.plot(s3[0],s3[1])
#plt.plot(s4[0],s4[1])
#plt.plot(s5[0],s5[1])
#plt.show()

## Calculating open shear flows
qV0y =                          Coeffz * (t_sk * integrator_trap(s0[1],s0[2],0)    + A_stif*y0)
qV1y =                          Coeffz *  t_sp * integrator_trap(s1[1],s1[2],0)
qV2y = qV0y[-1] +   qV1y[-1] +  Coeffz * (t_sk * integrator_trap(s2[1],s2[2],0)    + A_stif*y2)
qV3y = qV2y[-1] +               Coeffz * (t_sk * integrator_trap(s3[1],s3[2],0)    + A_stif*y3)
qV4y =                          Coeffz *  t_sp * integrator_trap(s4[1],s4[2],0)
qV5y = qV3y[-1] +   qV4y[-1] +  Coeffz * (t_sk * integrator_trap(s5[1],s5[2],0)    + A_stif*y5)


plt.plot(s_qcirc, qV0y)
plt.plot(s_qcirc[-1] + s_spar, qV1y[::-1])
plt.plot(s_qcirc[-1] + s_tail, qV2y)
plt.plot(s_qcirc[-1] + s_tail[-1] + s_tail, qV3y)
plt.plot(s_qcirc[-1] + s_tail[-1] + s_tail[-1] + s_spar, qV4y[::-1])
plt.plot(s_qcirc[-1] + s_tail[-1] + s_tail[-1] + s_qcirc, qV5y)
plt.show()

## Solving for closed structure flows
# Ax = b
A = np.zeros((2,2))
A[0,0],A[0,1] = (np.pi*h_a/2)/t_sk+h_a/t_sp,    -h_a/t_sp
A[1,0],A[1,1] = -h_a/t_sp,                       h_a/t_sp+2*l_tail/t_sk

b = np.zeros((2))
b[0] = -(integrator_cumutrap(qV0y,s0[2],0)/t_sk - integrator_cumutrap(qV1y,s1[2],0)/t_sp - integrator_cumutrap(qV4y,s4[2],0)/t_sp + integrator_cumutrap(qV5y,s5[2],0)/t_sk)
b[1] = -(integrator_cumutrap(qV1y,s1[2],0)/t_sp + integrator_cumutrap(qV2y,s2[2],0)/t_sk + integrator_cumutrap(qV3y,s3[2],0)/t_sk + integrator_cumutrap(qV4y,s4[2],0)/t_sp)

x = sp.solve(A,b)
q0V0 = x[0]
q0V1 = x[1]

## Calculating for Shear Center
p0 = h_a/2                      ## np.sqrt((s0[0]+C_airfoil)*(s0[0]+C_airfoil) + s0[1]*s0[1])
p1 = 0                          ## np.sqrt((s1[0]+C_airfoil)*(s1[0]+C_airfoil) + s1[1]*s1[1])
p2 = h_a/2 * m.cos(alpha_tail)  ## np.sqrt((s2[0]+C_airfoil)*(s2[0]+C_airfoil) + s2[1]*s2[1])
p3 = h_a/2 * m.cos(alpha_tail)  ## np.sqrt((s3[0]+C_airfoil)*(s3[0]+C_airfoil) + s3[1]*s3[1])
p4 = 0                          ## np.sqrt((s4[0]+C_airfoil)*(s4[0]+C_airfoil) + s4[1]*s4[1])
p5 = h_a/2                      ## np.sqrt((s5[0]+C_airfoil)*(s5[0]+C_airfoil) + s5[1]*s5[1])

## Take moment around Spar 
eta =  ( integrator_cumutrap(p0*qV0y,s0[2],0) + integrator_cumutrap(p1*qV1y,s1[2],0) + integrator_cumutrap(p2*qV2y,s2[2],0) + integrator_cumutrap(p3*qV3y,s3[2],0) + integrator_cumutrap(p4*qV4y,s4[2],0) + integrator_cumutrap(p5*qV5y,s5[2],0) + 2*AA_circ*q0V0 + 2*AA_tail*q0V1 )
eta += h_a/2

#plt.plot(s_qcirc,qV0z)
#plt.plot(s_spar+s_qcirc[-1],qV1z)
#plt.plot(s_qcirc+s_spar[-1]+s_qcirc[-1],qV2z)
#plt.plot(s_tail, qV3z)
#plt.plot(s_tail[-1] + s_spar, qV4z)
#plt.plot(s_tail[-1] + s_spar[-1] + s_tail, qV5z)
#plt.show()

print("=========== Statistics: ===========")
print("Total Area \t\t= {:.4g} \tm^2".format(A_circ+A_spar+2*A_tail + n_st*A_stif))
print("Airfoil Centroid \t= {:4f} \tm".format(C_airfoil))
print("Cross Section Izz \t= {:4g} \tm^4".format(Izz_airfoil))
print("Cross Section Iyy \t= {:4g} \tm^4".format(Iyy_airfoil))
print("Shear Center from LE \t= {:4f} \tm".format(eta))

#print(ZY_stif)
#
#test = np.array([[-0.        ,  0.        ],
#       [-0.03692049,  0.07877549],
#       [-0.12081074,  0.09876497],
#       [-0.20884515,  0.08080771],
#       [-0.29687956,  0.06285044],
#       [-0.38491397,  0.04489317],
#       [-0.47294838,  0.0269359 ],
#       [-0.56098279,  0.00897863],
#       [-0.56098279, -0.00897863],
#       [-0.47294838, -0.0269359 ],
#       [-0.38491397, -0.04489317],
#       [-0.29687956, -0.06285044],
#       [-0.20884515, -0.08080771],
#       [-0.12081074, -0.09876497],
#       [-0.03692049, -0.07877549]])
#
#plt.plot(-test[:,0], test[:,1],marker="o",linestyle="none")
#plt.plot(ZY_stif[0,:], ZY_stif[1,:],marker="o",linestyle="none")
plt.show()
#
