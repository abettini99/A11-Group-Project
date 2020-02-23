"""
Institution:    TU Delft
Authors:        A11
Date:           17-02-2020

This tool gives a first estimate of the structural deformation of an aileron attached to a wing at limit aerodynamic load under a certain deflection. 

Coordinate System: x' (span-wise), y'(perpendicular to x',z' from symmetry line of airfoil), z' (chord wise, LE to TE)
"""

##  =========== Library Imports: =========== 
import os
import numpy as np
import scipy.linalg as sp
import math as m
import matplotlib
import matplotlib.pyplot as plt
import time

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
def integrator_trap(f, x, F0=0):
    """
    == inputs:
    f   -- np.array of function values in "y" direction
    x   -- np.array of x locations of functions values in "y" direction
    f0  -- integration value at x0
    == outputs:
    F   -- np.array of all the integration values of "f" up to the respective "x" value
 
    Integrate a given array along the x array using the Trapezoid Rule. Each output F[i] corresponds to the integration from x0 to x[i]. 
    """

    F = np.empty(x.shape)
    F[0] = F0
    for i in range(1,x.shape[0]):
        F[i] = F[i-1] + np.abs(x[i] - x[i-1])/2*(f[i] + f[i-1])
    return F

def integrator_cumutrap(f, x, F0=0):
    F = F0 + np.sum( np.abs(x[1:] - x[:-1])/2 * (f[1:] + f[:-1]) )
    return F 

def central_derivative(F,x):
    f = np.empty(x.shape)
    f[0] = 1/(x[1]-x[0])*(F[1]-F[0])
    f[1:-1] = 1/(x[2:] - x[:-2])*(F[2:] - F[:-2])
    f[-1] = 1/(x[-1]-x[-2])*(F[-1]-F[-2])
    return f
#
#def integrator_simp(f, x):
#    Flst = []
#    Flst.append( ((x[2::2] - x[:-2:2])/6) * (f[2::2] + 4*(f[1:-1:2]) + f[:-2:2]) )
#    Farr = np.array(Flst)
#    F = np.sum(Farr)
#    return Farr, F
#
#def integrator_cumusimp(f, x):
#    F = np.sum( ((x[1:] - x[:-1])/6) * (f[1:] + 2*(f[1:]+f[:-1]) + f[:-1]) )
#    return F

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

def MC(x,x0):  ## McCauley
    if type(x) == type(np.array([])):
        return np.maximum(x-x0,[0]*x.shape[0])
    else:    
        return max(x-x0,0)

def MC0(x,x0):
    if type(x) == type(np.array([])):
        return np.maximum(x-x0, np.zeros(x.shape))/(x-x0)
    else:
        return 1 if x-x0 > 0 else 0

##  =========== Input Parameters: =========== 
Aircraft= "Do228"       ## Aircraft type, relevant for aerodynamic load
C_a     = 0.515         ## aileron chord                                [m]
l_a     = 2.691         ## span of aileron                              [m]
x_1     = 0.174         ## x-location of hinge 1                        [m]
x_2     = 1.051         ## x-location of hinge 2                        [m]
x_3     = 2.512         ## x-location of hinge 3                        [m]
x_a     = 30            ## distance between Actuator I and Actuator II  [cm]
h_a     = 24.8          ## aileron height                               [cm]
t_sk    = 1.1           ## skin thickness                               [mm]
t_sp    = 2.2           ## spar thickness                               [mm]
t_st    = 1.2           ## stiffener thickness                          [mm]
h_st    = 1.5           ## stiffener height                             [cm]
w_st    = 3.0           ## stiffener width                              [cm]
n_st    = 11            ## number of stiffeners                         [-]
d_1     = 1.034         ## vertical displacement hinge 1                [cm]
d_3     = 2.066         ## vertical displacement hinge 3                [cm]
phi     = 25            ## aileron deflection                           [deg]
P       = 20.6          ## actuator load                                [kN]

E       = 73.1          ## material Young's modulus                     [GPa]
G       = 28.0          ## material shear moduus                        [GPa]
rho     = 2780          ## material density                             [kg m^-3]

x_loc   = 0.5           ## Point along aileron to visualize stress/shear
dx      = 1e-3          ## step size                                    [m]
textsize= [12,14,16]    ## graph text size                              [small, medium, large] fonts respectively
colors  = ["blue","green","yellow","orange","red"]                   ## Graph colors --> from lowest value color to highest value color

export  = True          ## want to export graphs?                       True , False
texpsize= [26,28,30]    ## graph text export size                       [small, medium, large] fonts respectively

##  =========== Main Program: =========== 
## Start simulation timer
t_ss    = time.time()

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
if Aircraft == "Do228":
    AeroDataLoad    = np.loadtxt("aero_data/aerodynamicloaddo228.dat", delimiter=",")     ## Aero Data along [Z,X] [kN m^-1]
elif Aircraft == "a320":
    AeroDataLoad    = np.loadtxt("aero_data/aerodynamicloada320.dat", delimiter=",")
elif Aircraft == "crj700":
    AeroDataLoad    = np.loadtxt("aero_data/aerodynamicloadcrj700.dat", delimiter=",")
elif Aircraft == "f100":
    AeroDataLoad    = np.loadtxt("aero_data/aerodynamicloadf100.dat", delimiter=",")
else:
    AeroDataLoad    = np.zeros((81,41))

AeroDataLoad   *= 1000                                                      ## [kN m^-1 -> N m^-1] 
theta_zi        = (np.arange(1,82+1,1)-1)/81*m.pi
theta_xi        = (np.arange(1,42+1,1)-1)/41*m.pi
AeroDataZ       = 0.5*( 0.5*C_a*(1-np.cos(theta_zi[:-1])) + 0.5*C_a*(1-np.cos(theta_zi[1:])) ) ## Original equation gives data from LE, with z+ being in direction from TE to LE. Hence from LE to TE +, must multiply by -1
AeroDataX       = 0.5*( 0.5*l_a*(1-np.cos(theta_xi[:-1])) + 0.5*l_a*(1-np.cos(theta_xi[1:])) )
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#X, Z = np.meshgrid(AeroDataX, AeroDataZ)
#ax.plot_surface(X, Z, AeroDataLoad)
#plt.show()
q               = np.zeros((AeroDataLoad.shape[1]))
qm              = np.zeros((AeroDataLoad.shape[1]))
for i in range(AeroDataLoad.shape[1]):
    q[i]        = integrator_cumutrap(AeroDataLoad[:,i], AeroDataZ, 0)
    qm[i]       = integrator_cumutrap(AeroDataLoad[:,i]*(AeroDataZ-0.25*C_a), AeroDataZ, 0)
AeroDataX       = np.hstack((0,AeroDataX,l_a))
q               = np.hstack((0,q,0))
qm              = np.hstack((0,qm,0))

#print(AeroDataLoad[:,2]*(AeroDataZ-0.25*C_a), AeroDataZ)
#print(AeroDataZ-0.25*C_a)
#print(AeroDataLoad[:,2])

#plt.plot(AeroDataZ-0.25*C_a, AeroDataLoad[:,2]*(AeroDataZ-0.25*C_a))
#plt.show()

#plt.plot(AeroDataX, q)
#plt.plot(AeroDataX, qm)
#plt.show()

## =========== Cross Section Properties:  =========== 
ZY_stif     = st_locations(C_a, h_a, n_st)      ## [(Z,Y,Beta),(0,1,...,n_st-1)] of all stiffeners along airfoil
l_tail      = m.sqrt( (C_a - 0.5*h_a)*(C_a - 0.5*h_a) + (0.5*h_a)*(0.5*h_a) )
alpha_tail  = m.asin( (0.5*h_a)/l_tail )
beta_stif   = ZY_stif[2]                        ## Angles of all stiffeners along airfoil

#plt.plot(ZY_stif[0],ZY_stif[1], marker = "o")
#plt.show()

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
l_qcirc = np.pi/2*h_a/2

## Recall [(Z,Y,Beta),(0,1,...,n_st-1)] of all stiffeners along airfoil, we want to find all stiffeners on LE or TE, and on Bottom or Top
circ_stify = ZY_stif[1, ZY_stif[0,:] < h_a/2]    ## all stiffeners in the semi-circle -- The Y value returned
tail_stify = ZY_stif[1, ZY_stif[0,:] > h_a/2]    ## all stiffeners in the tail
topcirc_stify = circ_stify[circ_stify>0]
botcirc_stify = circ_stify[circ_stify<0]
toptail_stify = tail_stify[tail_stify>0]
bottail_stify = tail_stify[tail_stify<0]

top_stifx = ZY_stif[0, ZY_stif[1,:] > 0]
bot_stifx = ZY_stif[0, ZY_stif[1,:] < 0]
topcirc_stifx = top_stifx[top_stifx < h_a/2] - C_airfoil
botcirc_stifx = bot_stifx[bot_stifx < h_a/2] - C_airfoil
toptail_stifx = top_stifx[top_stifx > h_a/2] - C_airfoil
bottail_stifx = bot_stifx[bot_stifx > h_a/2] - C_airfoil

## Create Curve Space
s_qcirc = np.linspace(0,l_qcirc,s_reso)     ## Quarter Circle
s_spar = np.linspace(0,h_a/2,s_reso)              ## Spar
s_tail = np.linspace(0,l_tail,s_reso)             ## Single Tail

## Create list of all z,y,s list for integrals --> play around with plotting s[0] vs s[1] and plotting s[0][:400] to s[1][:400]
s0 = np.array(  [h_a/2 - h_a/2*np.cos(s_qcirc/(h_a/2)) -C_airfoil                                        , h_a/2*np.sin(s_qcirc/(h_a/2))             , s_qcirc]   )
s1 = np.array(  [(h_a/2 + t_sp/2)*np.ones((s_reso)) -C_airfoil                                             , s_spar                                    , s_spar]    )
s2 = np.array(  [h_a/2 + ( l_tail*m.cos(alpha_tail) - s_tail[::-1]*m.cos(alpha_tail) ) -C_airfoil , s_tail[::-1]*m.sin(alpha_tail)            , s_tail]    )
s3 = np.array(  [h_a/2 + ( l_tail*m.cos(alpha_tail) - s_tail*m.cos(alpha_tail) ) -C_airfoil       , -s_tail*m.sin(alpha_tail)                 , s_tail]    )
s4 = np.array(  [(h_a/2 + t_sp/2)*np.ones((s_reso)) -C_airfoil                                             , -s_spar                                   , s_spar]    )
s5 = np.array(  [h_a/2 - h_a/2*np.cos(s_qcirc[::-1]/(h_a/2)) -C_airfoil                                  , -h_a/2*np.sin(s_qcirc[::-1]/(h_a/2))      , s_qcirc]   )

## Create stiffener influences
y0 = np.zeros((s_reso))
y2 = np.zeros((s_reso))
y3 = np.zeros((s_reso))
y5 = np.zeros((s_reso))
for stif in topcirc_stify:
    y0 += MC0(s0[1], stif)*stif     
for stif in toptail_stify:
    y2 += (MC0(s2[1], stif) -1)*(-1)*stif ## this flips it, trust me --> [1,1,1,1,0,0,0,0] we want to flip it -->  -1 -> [0,0,0,0,-1,-1,-1,-1] --> *(-1) -> [0,0,0,0,1,1,1,1]
for stif in bottail_stify:
    y3 += (MC0(s3[1], stif) -1)*(-1)*stif
for stif in botcirc_stify:
    y5 += MC0(s5[1], stif)*stif

x0 = np.zeros((s_reso))
x2 = np.zeros((s_reso))
x3 = np.zeros((s_reso))
x5 = np.zeros((s_reso))
for stif in topcirc_stifx:
    x0 += MC0(s0[0], stif)*stif
for stif in toptail_stifx:
    x2 += MC0(s2[0], stif)*stif
for stif in bottail_stifx:
    x3 += (MC0(s3[0], stif) -1)*(-1)*stif
for stif in botcirc_stifx:
    x5 += (MC0(s5[0], stif) -1)*(-1)*stif
#
#plt.plot(s_qcirc, x0)
#plt.plot(s_tail, x2)
#plt.plot(s_tail, x3)
#plt.plot(s_qcirc, x5)
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
qV4y =                         -Coeffz *  t_sp * integrator_trap(s4[1],s4[2],0)                     ## Negative as the integral starts from 0 to -ha/2 whereas official integral starts from -ha/2 to 0.
qV5y = qV3y[-1] -   qV4y[-1] +  Coeffz * (t_sk * integrator_trap(s5[1],s5[2],0)    + A_stif*y5)

qV0z =                          Coeffy * (t_sk * integrator_trap(s0[0],s0[2],0)    + A_stif*x0      + 0.5*A_stif*(-C_airfoil))  ## Cut was between a stringer, so half area top at s = 0, half area bottom s = s.end
qV1z =                          Coeffy *  t_sp * integrator_trap(s1[0],s1[2],0)
qV2z = qV0z[-1] +   qV1z[-1] +  Coeffy * (t_sk * integrator_trap(s2[0],s2[2],0)    + A_stif*x2)
qV3z = qV2z[-1] +               Coeffy * (t_sk * integrator_trap(s3[0],s3[2],0)    + A_stif*x3)
qV4z =                         -Coeffy *  t_sp * integrator_trap(s4[0],s4[2],0)
qV5z = qV3z[-1] -   qV4z[-1] +  Coeffy * (t_sk * integrator_trap(s5[0],s5[2],0)    + A_stif*x5)

## Tiny small error still exists in here --> ending point value isnt anti-symmetric around TE for qVz

#print(qV0z[0], qV5z[-1])
#print(qV0y[0], qV5y[-1])
#print(qV2z[-1])

#plt.plot(s_qcirc, qV0y)
#plt.plot(s_qcirc[-1] + s_spar, qV1y[::-1])
#plt.plot(s_qcirc[-1] + s_tail, qV2y)
#plt.plot(s_qcirc[-1] + s_tail[-1] + s_tail, qV3y)
#plt.plot(s_qcirc[-1] + s_tail[-1] + s_tail[-1] + s_spar, qV4y[::-1])
#plt.plot(s_qcirc[-1] + s_tail[-1] + s_tail[-1] + s_qcirc, qV5y)
#plt.show()
#
#plt.plot(s_qcirc, qV0z)
#plt.plot(s_qcirc[-1] + s_spar, qV1z[::-1])
#plt.plot(s_qcirc[-1] + s_tail, qV2z)
#plt.plot(s_qcirc[-1] + s_tail[-1] + s_tail, qV3z)
#plt.plot(s_qcirc[-1] + s_tail[-1] + s_tail[-1] + s_spar, qV4z[::-1])
#plt.plot(s_qcirc[-1] + s_tail[-1] + s_tail[-1] + s_qcirc, qV5z)
#plt.show()
#

## Solving for closed structure flows
# Ax = b for q0Vy, Ax = c for q0Vz
A = np.zeros((2,2))
A[0,0],A[0,1] = 2*l_qcirc/t_sk+h_a/t_sp,        -h_a/t_sp
A[1,0],A[1,1] = -h_a/t_sp,                       h_a/t_sp+2*l_tail/t_sk

b = np.zeros((2))
b[0] = -(integrator_cumutrap(qV0y,s0[2],0)/t_sk - integrator_cumutrap(qV1y,s1[2],0)/t_sp - integrator_cumutrap(qV4y,s4[2],0)/t_sp + integrator_cumutrap(qV5y,s5[2],0)/t_sk)
b[1] = -(integrator_cumutrap(qV1y,s1[2],0)/t_sp + integrator_cumutrap(qV2y,s2[2],0)/t_sk + integrator_cumutrap(qV3y,s3[2],0)/t_sk + integrator_cumutrap(qV4y,s4[2],0)/t_sp)
c = np.zeros((2))
c[0] = -(integrator_cumutrap(qV0z,s0[2],0)/t_sk - integrator_cumutrap(qV1z,s1[2],0)/t_sp - integrator_cumutrap(qV4z,s4[2],0)/t_sp + integrator_cumutrap(qV5z,s5[2],0)/t_sk)
c[1] = -(integrator_cumutrap(qV1z,s1[2],0)/t_sp + integrator_cumutrap(qV2z,s2[2],0)/t_sk + integrator_cumutrap(qV3z,s3[2],0)/t_sk + integrator_cumutrap(qV4z,s4[2],0)/t_sp)

x = sp.solve(A,b)
q0V0y = x[0]
q0V1y = x[1]
x = sp.solve(A,c)
q0V0z = x[0]
q0V1z = x[1]

## Calculating Shear Center using only the shear force in y-direction, since z-direction shear is always at symmetry.
p0 = h_a/2                      ## np.sqrt((s0[0]+C_airfoil)*(s0[0]+C_airfoil) + s0[1]*s0[1])
p1 = 0                          ## np.sqrt((s1[0]+C_airfoil)*(s1[0]+C_airfoil) + s1[1]*s1[1])
p2 = h_a/2 * m.cos(alpha_tail)  ## np.sqrt((s2[0]+C_airfoil)*(s2[0]+C_airfoil) + s2[1]*s2[1])
p3 = h_a/2 * m.cos(alpha_tail)  ## np.sqrt((s3[0]+C_airfoil)*(s3[0]+C_airfoil) + s3[1]*s3[1])
p4 = 0                          ## np.sqrt((s4[0]+C_airfoil)*(s4[0]+C_airfoil) + s4[1]*s4[1])
p5 = h_a/2                      ## np.sqrt((s5[0]+C_airfoil)*(s5[0]+C_airfoil) + s5[1]*s5[1])

## Take moment around spar 
eta =  ( integrator_cumutrap(p0*qV0y,s0[2],0) + integrator_cumutrap(p1*qV1y,s1[2],0) + integrator_cumutrap(p2*qV2y,s2[2],0) + integrator_cumutrap(p3*qV3y,s3[2],0) + integrator_cumutrap(p4*qV4y,s4[2],0) + integrator_cumutrap(p5*qV5y,s5[2],0) + 2*AA_circ*q0V0y + 2*AA_tail*q0V1y )
eta += h_a/2
#
#print(2*AA_circ*q0V0, 2*AA_tail*q0V1)
#print(integrator_cumutrap(p0*qV0y,s0[2],0), integrator_cumutrap(p5*qV5y,s5[2],0))
#print(integrator_cumutrap(p2*qV2y,s2[2],0), integrator_cumutrap(p3*qV3y,s3[2],0))
#
#plt.plot(s_qcirc,qV0z)
#plt.plot(s_spar+s_qcirc[-1],qV1z)
#plt.plot(s_qcirc+s_spar[-1]+s_qcirc[-1],qV2z)
#plt.plot(s_tail, qV3z)
#plt.plot(s_tail[-1] + s_spar, qV4z)
#plt.plot(s_tail[-1] + s_spar[-1] + s_tail, qV5z)
#plt.show()

## =========== Torsional Stiffness:  =========== 
#Cx = d

C = np.zeros((3,3))
C[0,0],C[0,1],C[0,2] = 2*AA_circ,                                       2*AA_tail,                                  0
C[1,0],C[1,1],C[1,2] = 1/(2*AA_circ)*(2*l_qcirc/t_sk + h_a/t_sp),       -1/(2*AA_circ)*(h_a/t_sp),                  -1
C[2,0],C[2,1],C[2,2] = -1/(2*AA_tail)*(h_a/t_sp),                       1/(2*AA_tail)*(2*l_tail/t_sk + h_a/t_sp),   -1

d = np.zeros((3))
d[0] = 1
d[1] = 0
d[2] = 0

x = sp.solve(C,d)
qT0 = x[0]
qT1 = x[1]
dtheta_dx = x[2]/G ## dtheta_dx for T = 1 !!!

J = 1/(G*dtheta_dx) ## dtheta_dx = T(x)/(GJ)

## =========== Solve Deflection:  =========== 
## Additional Parameters:
x_I     = x_2-x_a/2
x_II    = x_2+x_a/2
Py_II   = -P*m.sin(phi)
Pz_II   = P*m.cos(phi)

## Set up Aero Data properly:
x_reso  = l_a/dx
x       = np.linspace(0,l_a,x_reso)
q       = linear_interpolate(q,AeroDataX,x)
qm      = linear_interpolate(qm,AeroDataX,x)
int_q   = integrator_trap(q, x, 0)      #xx is the value up to which you want to integrate
iint_q  = integrator_trap(int_q, x)

int_xint_q = integrator_trap(x*int_q, x)
iint_xint_q= integrator_trap(int_xint_q, x)

int_qx  = integrator_trap(q*x, x)    #xx is the value up to which you want to integrate
iint_qx = integrator_trap(int_qx, x)
iiint_qx= integrator_trap(iint_qx, x)

int_qm  = integrator_trap(qm, x)     #xx is the value up to which you want to integrate
iint_qm = integrator_trap(int_qm, x)

#plt.plot(x,int_q)
#plt.plot(x,iint_q)
#plt.plot(x,iiint_q)
#plt.show()
#
## Set up Linear System:
M       = np.zeros((13,13))
K       = np.zeros((13))

## Unknowns  :  {vec} = [Ry1, Ry2, Ry3, Rz1, Rz2, Rz3, Cu_p0, Cu0, Cv_p0, Cv0, Ctheta0, Py_I, Pz_I]
## Row 1     :  Force vector inline with Jammed Actuator
## Row 2 -6  :  Force and Moment Equilibrium around x = l_a
## Row 7 -8  :  Hinge 1 Deflection Constraint
## Row 9 -10 :  Hinge 2 Deflection Constraint
## Row 11-12 :  Hinge 3 Deflection Constraint
## Row 13    :  Actuator Deflection Constraint

## RHS should be a scalar of all known terms from your compatability/boundary equations. The (LHS) should be an array -- each element of the array corresponds to the respective coefficient of the unknown vector {vec}. Entries are done such as if everything was kept on one side of the equation -- RHS notation is not what it seems like (i.e its on the RHS), it is just the set of terms that are not affected by an unknown variable, ie: u({vec},x) = u_LHS({vec},x) + u_RHS(x)
u_LHS       = lambda X : -1/(E*Izz_airfoil)* np.array([-1/6*MC(X,x_1)**3, -1/6*MC(X,x_2)**3, -1/6*MC(X,x_3)**3, 0, 0, 0, X, 1, 0, 0, 0, -1/6*MC(X,x_I)**3, 0])       ## Since our shear directions werent originally in the opposite directions to the normal axes
u_RHS       = lambda X : -1/(E*Izz_airfoil)*         (iiint_qx[(x<=X) & (x>=X-dx)]    - iint_xint_q[(x<=X) & (x>=X-dx)]    - 1/6*MC(X,x_II)**3*Py_II)           ## Works for both scaler inputs and vector inputs now!

v_LHS       = lambda X : -1/(E*Iyy_airfoil)* np.array([0, 0, 0, -1/6*MC(X,x_1)**3, -1/6*MC(X,x_2)**3, -1/6*MC(X,x_3)**3, 0, 0, X, 1, 0, 0, -1/6*MC(X,x_I)**3])  
v_RHS       = lambda X : -1/(E*Iyy_airfoil)*         (-1/6*MC(X,x_II)**3*Pz_II)                                                                                 

theta_LHS   = lambda X : 1/(G*J)           * np.array([-(eta-h_a/2)*MC(X,x_1), -(eta-h_a/2)*MC(X,x_2), -(eta-h_a/2)*MC(X,x_3), 0, 0, 0, 0, 0, 0, 0, 1, -eta*MC(X,x_I), -h_a/2**MC(X,x_I)])
theta_RHS   = lambda X : 1/(G*J)           *         (-Py_II*eta*MC(X,x_II)  - Pz_II*h_a/2*MC(X,x_II)   - iint_qm[(x<=X) & (x>=X-dx)]     - (C_a/4-h_a/2)*iint_q[(x<=X) & (x>=X-dx)] )

Ry_LHS      = lambda X :                     np.array([-MC0(X,x_1), -MC0(X,x_2), -MC0(X,x_3), 0, 0, 0, 0, 0, 0, 0, 0, -MC0(X,x_I), 0])
Ry_RHS      = lambda X :                             (int_q[(x<=X) & (x>=X-dx)]  - Py_II*MC0(X,x_II))

Rz_LHS      = lambda X :                     np.array([0, 0, 0, -MC0(X,x_1), -MC0(X,x_2), -MC0(X,x_3), 0, 0, 0, 0, 0, 0, -MC0(X,x_I)])
Rz_RHS      = lambda X :                             (-Pz_II*MC0(X,x_II))

Mx_LHS      = lambda X :                     np.array([-(eta-h_a/2)*MC0(X,x_1), -(eta-h_a/2)*MC0(X,x_2), -(eta-h_a/2)*MC0(X,x_3), 0, 0, 0, 0, 0, 0, 0, 0, -eta*MC0(X,x_I), -h_a/2*MC0(X,x_I)])
Mx_RHS      = lambda X :                             (-int_qm[(x<=X) & (x>=X-dx)]  - (C_a/4-eta)*int_q[(x<=X) & (x>=X-dx)]     - Py_II*eta*MC0(X,x_II) - Pz_II*h_a/2*MC0(X,x_II))

My_LHS      = lambda X :                     np.array([0, 0, 0, -MC(X,x_1), -MC(X,x_2), -MC(X,x_3), 0, 0, 0, 0, 0, 0, -MC(X,x_I)])                              
My_RHS      = lambda X :                             (-Pz_II*MC(X,x_II))                                                                                     

Mz_LHS      = lambda X :                     np.array([-MC(X,x_1), -MC(X,x_2), -MC(X,x_3), 0, 0, 0, 0, 0, 0, 0, 0, -MC(X,x_I), 0])
Mz_RHS      = lambda X :                             (-Py_II*MC(X,x_II) + X*int_q[(x<=X) & (x>=X-dx)] - int_qx[(x<=X) & (x>=X-dx)])
## CHECK MATHETMATICS, WHY THE MINUS SIGNS???

## Solve for M {vec} = K  
M[0,11], M[0,12],   K[0]    =  -1, -m.tan(phi)                                                                                              ,   0
M[1,:] ,            K[1]    =  Ry_LHS(l_a)                                                                                                  ,   0 -   Ry_RHS(l_a)
M[2,:] ,            K[2]    =  Rz_LHS(l_a)                                                                                                  ,   0 -   Rz_RHS(l_a)
M[3,:] ,            K[3]    =  Mx_LHS(l_a)                                                                                                  ,   0 -   Mx_RHS(l_a)
M[4,:] ,            K[4]    =  My_LHS(l_a)                                                                                                  ,   0 -   My_RHS(l_a)
M[5,:] ,            K[5]    =  Mz_LHS(l_a)                                                                                                  ,   0 -   Mz_RHS(l_a)
M[6,:] ,            K[6]    =  m.cos(phi)*( u_LHS(x_1) - theta_LHS(x_1)*(eta-h_a/2) ) + m.sin(phi)*  v_LHS(x_1)                             , d_1 - ( m.cos(phi)*( u_RHS(x_1) - theta_RHS(x_1)*(eta-h_a/2) ) + m.sin(phi)*  v_RHS(x_1) )
M[7,:] ,            K[7]    = -m.sin(phi)*( u_LHS(x_1) - theta_LHS(x_1)*(eta-h_a/2) ) + m.cos(phi)*  v_LHS(x_1)                             ,   0 - (-m.sin(phi)*( u_RHS(x_1) - theta_RHS(x_1)*(eta-h_a/2) ) + m.cos(phi)*  v_RHS(x_1) )
M[8,:] ,            K[8]    =  m.cos(phi)*( u_LHS(x_2) - theta_LHS(x_2)*(eta-h_a/2) ) + m.sin(phi)*  v_LHS(x_2)                             ,   0 - ( m.cos(phi)*( u_RHS(x_2) - theta_RHS(x_2)*(eta-h_a/2) ) + m.sin(phi)*  v_RHS(x_2) )
M[9,:] ,            K[9]    = -m.sin(phi)*( u_LHS(x_2) - theta_LHS(x_2)*(eta-h_a/2) ) + m.cos(phi)*  v_LHS(x_2)                             ,   0 - (-m.sin(phi)*( u_RHS(x_2) - theta_RHS(x_2)*(eta-h_a/2) ) + m.cos(phi)*  v_RHS(x_2) )
M[10,:],            K[10]   =  m.cos(phi)*( u_LHS(x_3) - theta_LHS(x_3)*(eta-h_a/2) ) + m.sin(phi)*  v_LHS(x_3)                             , d_3 - ( m.cos(phi)*( u_RHS(x_3) - theta_RHS(x_3)*(eta-h_a/2) ) + m.sin(phi)*  v_RHS(x_3) ) 
M[11,:],            K[11]   = -m.sin(phi)*( u_LHS(x_3) - theta_LHS(x_3)*(eta-h_a/2) ) + m.cos(phi)*  v_LHS(x_3)                             ,   0 - (-m.sin(phi)*( u_RHS(x_3) - theta_RHS(x_3)*(eta-h_a/2) ) + m.cos(phi)*  v_RHS(x_3) )
M[12,:],            K[12]   = -m.sin(phi)*( u_LHS(x_I) - theta_LHS(x_I)* eta        ) + m.cos(phi)*( v_LHS(x_I) - theta_LHS(x_I)*h_a/2 )    ,   0 - (-m.sin(phi)*( u_RHS(x_I) - theta_RHS(x_I)* eta        ) + m.cos(phi)*( v_RHS(x_I) + theta_RHS(x_I)*h_a/2 ) )

vec = sp.solve(M,K)
Ry1,Ry2,Ry3,Rz1,Rz2,Rz3,Cu_p0,Cu0,Cv_p0,Cv0,Ctheta0,Py_I,Pz_I = vec        ## Note: u0 = Cu0/(EIzz), v0 = Cv0/(EIyy), theta0 = Ctheta0/(GJ)

Ry      = np.sum(Ry_LHS(x)*vec)         + Ry_RHS(x)
Rz      = np.sum(Rz_LHS(x)*vec)         + Rz_RHS(x)
Mx      = np.sum(Mx_LHS(x)*vec)         + Mx_RHS(x)
My      = np.sum(My_LHS(x)*vec)         + My_RHS(x)
Mz      = np.sum(Mz_LHS(x)*vec)         + Mz_RHS(x)
theta   = np.sum(theta_LHS(x)*vec)      + theta_RHS(x)
u       = np.sum(u_LHS(x)*vec)          + u_RHS(x)
v       = np.sum(v_LHS(x)*vec)          + v_RHS(x)
U       = m.cos(phi)*(u - theta*(eta-h_a/2) ) + m.sin(phi)*v
V       = -m.sin(phi)*(u - theta*(eta-h_a/2) ) + m.cos(phi)*v

dudx    = central_derivative(u,x)
dvdx    = central_derivative(v,x)
dUdx    = central_derivative(U,x)
dVdx    = central_derivative(V,x)

V      *= -1 ## Original frame of reference points from TE to LE, hence deflection should be negative
##v      *= -1 ## Corotated frame of reference points from TE to LE in the verification tool, hence deflection should be negative 

""" Slowest Section of all """
## =========== Shear Flow and Stress Calculations:  ===========
## Calculate stress and shear along all components of the cross sections and along the span
sig0_xx = np.outer(My, s0[0])/Iyy_airfoil + np.outer(Mz, s0[1])/Izz_airfoil 
sig1_xx = np.outer(My, s1[0])/Iyy_airfoil + np.outer(Mz, s1[1])/Izz_airfoil
sig2_xx = np.outer(My, s2[0])/Iyy_airfoil + np.outer(Mz, s2[1])/Izz_airfoil
sig3_xx = np.outer(My, s3[0])/Iyy_airfoil + np.outer(Mz, s3[1])/Izz_airfoil
sig4_xx = np.outer(My, s4[0])/Iyy_airfoil + np.outer(Mz, s4[1])/Izz_airfoil
sig5_xx = np.outer(My, s5[0])/Iyy_airfoil + np.outer(Mz, s5[1])/Izz_airfoil

q0 = np.outer(Ry, qV0y     + q0V0y             )    + np.outer(Rz, qV0z     + q0V0z             )    + np.outer(Mx*( qT0         ), np.ones(qV0z.shape))
q1 = np.outer(Ry, qV1y     - q0V0y     + q0V1y )    + np.outer(Rz, qV1z     - q0V0z     + q0V1z )    + np.outer(Mx*(-qT0   + qT1 ), np.ones(qV1z.shape))
q2 = np.outer(Ry, qV2y                 + q0V1y )    + np.outer(Rz, qV2z                 + q0V1z )    + np.outer(Mx*(         qT1 ), np.ones(qV2z.shape))
q3 = np.outer(Ry, qV3y                 + q0V1y )    + np.outer(Rz, qV3z                 + q0V1z )    + np.outer(Mx*(         qT1 ), np.ones(qV3z.shape))
q4 = np.outer(Ry, qV4y     - q0V0y     + q0V1y )    + np.outer(Rz, qV4z     - q0V0z     + q0V1z )    + np.outer(Mx*(-qT0   + qT1 ), np.ones(qV4z.shape))
q5 = np.outer(Ry, qV5y     + q0V0y             )    + np.outer(Rz, qV5z     + q0V0z             )    + np.outer(Mx*( qT0         ), np.ones(qV5z.shape))

## Replace all information into one big array
Z = np.hstack((s0[0],s1[0],s2[0],s3[0],s4[0],s5[0])) + C_airfoil
Y = np.hstack((s0[1],s1[1],s2[1],s3[1],s4[1],s5[1]))

n = 0
sig_xx = np.empty((My.shape[0], s0.shape[1] + s1.shape[1] + s2.shape[1] + s3.shape[1] + s4.shape[1] + s5.shape[1]))
sig_xx[:,n:n+s0.shape[1]], n    = sig0_xx,  n + s0.shape[1]
sig_xx[:,n:n+s1.shape[1]], n    = sig1_xx,  n + s1.shape[1]
sig_xx[:,n:n+s2.shape[1]], n    = sig2_xx,  n + s2.shape[1]
sig_xx[:,n:n+s3.shape[1]], n    = sig3_xx,  n + s3.shape[1]
sig_xx[:,n:n+s4.shape[1]], n    = sig4_xx,  n + s4.shape[1]
sig_xx[:,n:n+s5.shape[1]]       = sig5_xx

n = 0
q = np.empty((Ry.shape[0], qV0y.shape[0] + qV1y.shape[0] + qV2y.shape[0] + qV3y.shape[0] + qV4y.shape[0] + qV5y.shape[0]))
q[:,n:n+qV0y.shape[0]], n       = q0, n + qV0y.shape[0]
q[:,n:n+qV1y.shape[0]], n       = q1, n + qV1y.shape[0]
q[:,n:n+qV2y.shape[0]], n       = q2, n + qV2y.shape[0]
q[:,n:n+qV3y.shape[0]], n       = q3, n + qV3y.shape[0]
q[:,n:n+qV4y.shape[0]], n       = q4, n + qV4y.shape[0]
q[:,n:n+qV5y.shape[0]]          = q5

n = 0
tau_yz = np.empty((Ry.shape[0], qV0y.shape[0] + qV1y.shape[0] + qV2y.shape[0] + qV3y.shape[0] + qV4y.shape[0] + qV5y.shape[0]))
tau_yz[:,n:n+qV0y.shape[0]], n  = q0/t_sk, n + qV0y.shape[0]
tau_yz[:,n:n+qV1y.shape[0]], n  = q1/t_sp, n + qV1y.shape[0]
tau_yz[:,n:n+qV2y.shape[0]], n  = q2/t_sk, n + qV2y.shape[0]
tau_yz[:,n:n+qV3y.shape[0]], n  = q3/t_sk, n + qV3y.shape[0]
tau_yz[:,n:n+qV4y.shape[0]], n  = q4/t_sp, n + qV4y.shape[0]
tau_yz[:,n:n+qV5y.shape[0]]     = q5/t_sk

sig_vm = np.sqrt(sig_xx*sig_xx*(2/2) + 3*tau_yz*tau_yz)

## Find the maximums locations along the span and chord
idx_sigxx   = np.where(np.abs(sig_xx) == np.max(np.abs(sig_xx)))
idx_tauyz  = np.where(np.abs(tau_yz) == np.max(np.abs(tau_yz)))
idx_sigvm  = np.where(np.abs(sig_vm) == np.max(np.abs(sig_vm)))
#
#print(idx_sigxx, idx_tauyz, idx_sigvm)
## End simulation timer:
t_se    = time.time()

## =========== Export Code:  =========== 
if export:
    ## Define general plot parameters for all future plots.
    SMALL_SIZE  = texpsize[0]
    MEDIUM_SIZE = texpsize[1]
    BIGGER_SIZE = texpsize[2]

    plt.style.use('grayscale')
    plt.rc('font', size=MEDIUM_SIZE, family='serif')    ## controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)                ## fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)                ## fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)               ## fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)               ## fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)               ## legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)             ## fontsize of the figure title
    plt.rc('text', usetex=False)
    matplotlib.rcParams['lines.linewidth']  = 1.5
    matplotlib.rcParams['figure.facecolor'] = 'white'
    matplotlib.rcParams['axes.facecolor']   = 'white'
    matplotlib.rcParams["legend.fancybox"]  = False

    print("Checking existence of export folders:") 
    if os.path.exists('SavedFiles'):
        print('SavedFiles\t\t[exists]')
    else:
        os.makedirs('SavedFiles')
        print('SavedFiles\t\t[created]')

    if os.path.exists('SavedFiles/{}'.format(Aircraft)):
        print('SavedFiles/{}\t[exists]'.format(Aircraft))
    else:
        os.makedirs('SavedFiles/{}'.format(Aircraft))
        print('SavedFiles/{}\t[created]'.format(Aircraft))
    print("")

    ## Begin export timer:
    t_es = time.time()
    counter = 0

    totalfiles  = 17
    print("Total files saved : %05d/%05d"%(counter, totalfiles), end="", flush=True)

    fig, ax = plt.subplots(1,1,squeeze=False,figsize=(16,9))
    ax[0,0].plot(x, u)
    ax[0,0].plot([x_1,x_2,x_3],[u[(x<=x_1)&(x>x_1-dx)],u[(x<=x_2)&(x>x_2-dx)],u[(x<=x_3)&(x>x_3-dx)]], marker = "s", linestyle='none', color='black', markerfacecolor='none', markeredgewidth=2, markersize=6)
    ax[0,0].set_ylabel(r"Deflection in $y^{\prime}$ direction $u\,\,[m]$")
    ax[0,0].set_xlabel(r"Position along Aileron $x\,\,[m]$")
    ax[0,0].set_xlim(0,l_a)
    ax[0,0].grid(True,which="major",color="#999999")
    ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--")
    ax[0,0].minorticks_on()
    ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
    ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
    fig.savefig("SavedFiles/{}/u.png".format(Aircraft), bbox_inches='tight')
    counter +=1
    print("\rTotal files saved : %05d/%05d"%(counter, totalfiles), end="", flush=True)

    fig, ax = plt.subplots(1,1,squeeze=False,figsize=(16,9))
    ax[0,0].plot(x, dudx)
    ax[0,0].plot([x_1,x_2,x_3],[dudx[(x<=x_1)&(x>x_1-dx)],dudx[(x<=x_2)&(x>x_2-dx)],dudx[(x<=x_3)&(x>x_3-dx)]], marker = "s", linestyle='none', color='black', markerfacecolor='none', markeredgewidth=2, markersize=6)
    ax[0,0].set_ylabel(r"Deflection derivative  in $y^{\prime}$ direction $\frac{du}{dx}\,\,[m\,m^{-1}]$")
    ax[0,0].set_xlabel(r"Position along Aileron $x\,\,[m]$")
    ax[0,0].set_xlim(0,l_a)
    ax[0,0].grid(True,which="major",color="#999999")
    ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--")
    ax[0,0].minorticks_on()
    ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
    ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
    fig.savefig("SavedFiles/{}/dudx.png".format(Aircraft), bbox_inches='tight')
    counter +=1
    print("\rTotal files saved : %05d/%05d"%(counter, totalfiles), end="", flush=True)

    fig, ax = plt.subplots(1,1,squeeze=False,figsize=(16,9))
    ax[0,0].plot(x, Mz)
    ax[0,0].set_ylabel(r"Internal Moment in $z^{\prime}$ direction $M_z^{\prime}\,\,[Nm]$")
    ax[0,0].set_xlabel(r"Position along Aileron $x\,\,[m]$")
    ax[0,0].set_xlim(0,l_a)
    ax[0,0].grid(True,which="major",color="#999999")
    ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--")
    ax[0,0].minorticks_on()
    ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
    ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
    fig.savefig("SavedFiles/{}/Mz.png".format(Aircraft), bbox_inches='tight')
    counter +=1
    print("\rTotal files saved : %05d/%05d"%(counter, totalfiles), end="", flush=True)

    fig, ax = plt.subplots(1,1,squeeze=False,figsize=(16,9))
    ax[0,0].plot(x, Ry)
    ax[0,0].set_ylabel(r"Shear Force in $y^{\prime}$ direction $R_y^{\prime}\,\,[N]$")
    ax[0,0].set_xlabel(r"Position along Aileron $x\,\,[m]$")
    ax[0,0].set_xlim(0,l_a)
    ax[0,0].grid(True,which="major",color="#999999")
    ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--")
    ax[0,0].minorticks_on()
    ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
    ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
    fig.savefig("SavedFiles/{}/Ry.png".format(Aircraft), bbox_inches='tight')
    counter +=1
    print("\rTotal files saved : %05d/%05d"%(counter, totalfiles), end="", flush=True)

    ## -------------------

    fig, ax = plt.subplots(1,1,squeeze=False,figsize=(16,9))
    ax[0,0].plot(x, -v)
    ax[0,0].plot([x_1,x_2,x_3],[-v[(x<=x_1)&(x>x_1-dx)],-v[(x<=x_2)&(x>x_2-dx)],-v[(x<=x_3)&(x>x_3-dx)]], marker = "s", linestyle='none', color='black', markerfacecolor='none', markeredgewidth=2, markersize=6)
    ax[0,0].set_ylabel(r"Deflection in $z^{\prime}$ direction $v\,\,[m]$")
    ax[0,0].set_xlabel(r"Position along Aileron $x\,\,[m]$")
    ax[0,0].set_xlim(0,l_a)
    ax[0,0].grid(True,which="major",color="#999999")
    ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--")
    ax[0,0].minorticks_on()
    ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
    ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
    fig.savefig("SavedFiles/{}/v.png".format(Aircraft), bbox_inches='tight')
    counter +=1
    print("\rTotal files saved : %05d/%05d"%(counter, totalfiles), end="", flush=True)

    fig, ax = plt.subplots(1,1,squeeze=False,figsize=(16,9))
    ax[0,0].plot(x, -dvdx)
    ax[0,0].plot([x_1,x_2,x_3],[-dvdx[(x<=x_1)&(x>x_1-dx)],-dvdx[(x<=x_2)&(x>x_2-dx)],-dvdx[(x<=x_3)&(x>x_3-dx)]], marker = "s", linestyle='none', color='black', markerfacecolor='none', markeredgewidth=2, markersize=6)
    ax[0,0].set_ylabel(r"Deflection derivative  in $z^{\prime}$ direction $\frac{dv}{dx}\,\,[m\,m^{-1}]$")
    ax[0,0].set_xlabel(r"Position along Aileron $x\,\,[m]$")
    ax[0,0].set_xlim(0,l_a)
    ax[0,0].grid(True,which="major",color="#999999")
    ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--")
    ax[0,0].minorticks_on()
    ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
    ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
    fig.savefig("SavedFiles/{}/dvdx.png".format(Aircraft), bbox_inches='tight')
    counter +=1
    print("\rTotal files saved : %05d/%05d"%(counter, totalfiles), end="", flush=True)

    fig, ax = plt.subplots(1,1,squeeze=False,figsize=(16,9))
    ax[0,0].plot(x, -My)
    ax[0,0].set_ylabel(r"Internal Moment in $y^{\prime}$ direction $-M_y^{\prime}\,\,[Nm]$")
    ax[0,0].set_xlabel(r"Position along Aileron $x\,\,[m]$")
    ax[0,0].set_xlim(0,l_a)
    ax[0,0].grid(True,which="major",color="#999999")
    ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--")
    ax[0,0].minorticks_on()
    ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
    ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
    fig.savefig("SavedFiles/{}/My.png".format(Aircraft), bbox_inches='tight')
    counter +=1
    print("\rTotal files saved : %05d/%05d"%(counter, totalfiles), end="", flush=True)

    fig, ax = plt.subplots(1,1,squeeze=False,figsize=(16,9))
    ax[0,0].plot(x, -Rz)
    ax[0,0].set_ylabel(r"Shear Force in $-z^{\prime}$ direction $R_z^{\prime}\,\,[N]$")
    ax[0,0].set_xlabel(r"Position along Aileron $x\,\,[m]$")
    ax[0,0].set_xlim(0,l_a)
    ax[0,0].grid(True,which="major",color="#999999")
    ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--")
    ax[0,0].minorticks_on()
    ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
    ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
    fig.savefig("SavedFiles/{}/Rz.png".format(Aircraft), bbox_inches='tight')
    counter +=1
    print("\rTotal files saved : %05d/%05d"%(counter, totalfiles), end="", flush=True)

    ## -------------------

    fig, ax = plt.subplots(1,1,squeeze=False,figsize=(16,9))
    ax[0,0].plot(x, U)
    ax[0,0].plot([x_1,x_2,x_3],[d_1,0,d_3], marker = "s", linestyle='none', color='black', markerfacecolor='none', markeredgewidth=2, markersize=6)
    ax[0,0].set_ylabel(r"Deflection in $y$ direction $U\,\,[m]$")
    ax[0,0].set_xlabel(r"Position along Aileron $x\,\,[m]$")
    ax[0,0].set_xlim(0,l_a)
    ax[0,0].grid(True,which="major",color="#999999")
    ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--")
    ax[0,0].minorticks_on()
    ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
    ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
    fig.savefig("SavedFiles/{}/U.png".format(Aircraft), bbox_inches='tight')
    counter +=1
    print("\rTotal files saved : %05d/%05d"%(counter, totalfiles), end="", flush=True)

    fig, ax = plt.subplots(1,1,squeeze=False,figsize=(16,9))
    ax[0,0].plot(x, V)
    ax[0,0].plot([x_1,x_2,x_3],[0,0,0], marker = "s", linestyle='none', color='black', markerfacecolor='none', markeredgewidth=2, markersize=6)
    ax[0,0].set_ylabel(r"Deflection in $z$ direction $V\,\,[m]$")
    ax[0,0].set_xlabel(r"Position along Aileron $x\,\,[m]$")
    ax[0,0].set_xlim(0,l_a)
    ax[0,0].grid(True,which="major",color="#999999")
    ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--")
    ax[0,0].minorticks_on()
    ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
    ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
    fig.savefig("SavedFiles/{}/V.png".format(Aircraft), bbox_inches='tight')
    counter +=1
    print("\rTotal files saved : %05d/%05d"%(counter, totalfiles), end="", flush=True)

    fig, ax = plt.subplots(1,1,squeeze=False,figsize=(16,9))
    ax[0,0].plot(x, dUdx)
    ax[0,0].plot([x_1,x_2,x_3],[dUdx[(x<=x_1)&(x>x_1-dx)],dUdx[(x<=x_2)&(x>x_2-dx)],dUdx[(x<=x_3)&(x>x_3-dx)]], marker = "s", linestyle='none', color='black', markerfacecolor='none', markeredgewidth=2, markersize=6)
    ax[0,0].set_ylabel(r"Deflection derivative in $y$ direction $\frac{dU}{dx}\,\,[m\,m^{-1}]$")
    ax[0,0].set_xlabel(r"Position along Aileron $x\,\,[m]$")
    ax[0,0].set_xlim(0,l_a)
    ax[0,0].grid(True,which="major",color="#999999")
    ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--")
    ax[0,0].minorticks_on()
    ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
    ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
    fig.savefig("SavedFiles/{}/dUdx.png".format(Aircraft), bbox_inches='tight')
    counter +=1
    print("\rTotal files saved : %05d/%05d"%(counter, totalfiles), end="", flush=True)

    fig, ax = plt.subplots(1,1,squeeze=False,figsize=(16,9))
    ax[0,0].plot(x, dVdx)
    ax[0,0].plot([x_1,x_2,x_3],[dVdx[(x<=x_1)&(x>x_1-dx)],dVdx[(x<=x_2)&(x>x_2-dx)],dVdx[(x<=x_3)&(x>x_3-dx)]], marker = "s", linestyle='none', color='black', markerfacecolor='none', markeredgewidth=2, markersize=6)
    ax[0,0].set_ylabel(r"Deflection derivative  in $z$ direction $\frac{dV}{dx}\,\,[m\,m^{-1}]$")
    ax[0,0].set_xlabel(r"Position along Aileron $x\,\,[m]$")
    ax[0,0].set_xlim(0,l_a)
    ax[0,0].grid(True,which="major",color="#999999")
    ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--")
    ax[0,0].minorticks_on()
    ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
    ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
    fig.savefig("SavedFiles/{}/dVdx.png".format(Aircraft), bbox_inches='tight')
    counter +=1
    print("\rTotal files saved : %05d/%05d"%(counter, totalfiles), end="", flush=True)

    ## -------------------

    fig, ax = plt.subplots(1,1,squeeze=False,figsize=(16,9))
    ax[0,0].plot(x, theta)
    ax[0,0].set_ylabel(r"Aileron Twist $\theta\,\,[m]$")
    ax[0,0].set_xlabel(r"Position along Aileron $x\,\,[m]$")
    ax[0,0].set_xlim(0,l_a)
    ax[0,0].grid(True,which="major",color="#999999")
    ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--")
    ax[0,0].minorticks_on()
    ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
    ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
    fig.savefig("SavedFiles/{}/theta.png".format(Aircraft), bbox_inches='tight')
    counter +=1
    print("\rTotal files saved : %05d/%05d"%(counter, totalfiles), end="", flush=True)

    fig, ax = plt.subplots(1,1,squeeze=False,figsize=(16,9))
    ax[0,0].plot(x, Mx)
    ax[0,0].set_ylabel(r"Internal Moment in $x$ direction $M_x\,\,[m]$")
    ax[0,0].set_xlabel(r"Position along Aileron $x\,\,[m]$")
    ax[0,0].set_xlim(0,l_a)
    ax[0,0].grid(True,which="major",color="#999999")
    ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--")
    ax[0,0].minorticks_on()
    ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
    ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
    fig.savefig("SavedFiles/{}/Mx.png".format(Aircraft), bbox_inches='tight')
    counter +=1
    print("\rTotal files saved : %05d/%05d"%(counter, totalfiles), end="", flush=True)

    ## -------------------

    fig, ax = plt.subplots(1,1,squeeze=False,figsize=(16,9))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
    im = ax[0,0].scatter(Z,Y,c=sig_xx[x<=x_loc][-1],cmap=cmap)
    fig.colorbar(im, ax=ax[0,0])
    ax[0,0].set_ylabel(r"Height Position from symmetry line $y^{\prime}\,\,[m]$")
    ax[0,0].set_xlabel(r"Position along Chord $z^{\prime}\,\,[m]$")
    ax[0,0].grid(True,which="major",color="#999999")
    ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--")
    ax[0,0].minorticks_on()
    ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
    ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
    fig.savefig("SavedFiles/{}/sig_X={}m.png".format(Aircraft,x_loc), bbox_inches='tight')
    counter +=1
    print("\rTotal files saved : %05d/%05d"%(counter, totalfiles), end="", flush=True)

    ## -------------------

    fig, ax = plt.subplots(1,1,squeeze=False,figsize=(16,9))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
    im = ax[0,0].scatter(Z,Y,c=q[x<=x_loc][-1],cmap=cmap)
    fig.colorbar(im, ax=ax[0,0])
    ax[0,0].set_ylabel(r"Height Position from symmetry line $y^{\prime}\,\,[m]$")
    ax[0,0].set_xlabel(r"Position along Chord $z^{\prime}\,\,[m]$")
    ax[0,0].grid(True,which="major",color="#999999")
    ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--")
    ax[0,0].minorticks_on()
    ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
    ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
    fig.savefig("SavedFiles/{}/q_X={}m.png".format(Aircraft,x_loc), bbox_inches='tight')
    counter +=1
    print("\rTotal files saved : %05d/%05d"%(counter, totalfiles), end="", flush=True)

    ## -------------------

    fig, ax = plt.subplots(1,1,squeeze=False,figsize=(16,9))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
    im = ax[0,0].scatter(Z,Y,c=sig_vm[x<=x_loc][-1],cmap=cmap)
    fig.colorbar(im, ax=ax[0,0])
    ax[0,0].set_ylabel(r"Height Position from symmetry line $y^{\prime}\,\,[m]$")
    ax[0,0].set_xlabel(r"Position along Chord $z^{\prime}\,\,[m]$")
    ax[0,0].grid(True,which="major",color="#999999")
    ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--")
    ax[0,0].minorticks_on()
    ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
    ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
    fig.savefig("SavedFiles/{}/sigvm_X={}m.png".format(Aircraft,x_loc), bbox_inches='tight')
    counter +=1
    print("\rTotal files saved : %05d/%05d"%(counter, totalfiles), end="", flush=True)

    print("")
    ## End export timer:    
    t_ee = time.time()

## =========== Data and Display:  =========== 
print("\n=========== Cross Section Properties: ===========")
print("Total Area \t\t= {:.4g} \tm^2".format(A_circ+A_spar+2*A_tail + n_st*A_stif))
print("Airfoil Centroid \t= {:4f} \tm".format(C_airfoil))
print("Cross Section Izz \t= {:4g} \tm^4".format(Izz_airfoil))
print("Cross Section Iyy \t= {:4g} \tm^4".format(Iyy_airfoil))
print("Shear Center from LE \t= {:4f} \tm \t(= {:4f} m from Spar)".format(eta,eta-h_a/2))
print("Torsional Constant \t= {:4g} \tm^4".format(J))

print("\n=========== Reaction Forces: ===========")
print("Ry1 \t\t\t= {:.1f} \tN".format(Ry1))
print("Ry2 \t\t\t= {:.1f} \tN".format(Ry2))
print("Ry3 \t\t\t= {:.1f} \tN".format(Ry3))
print("Rz1 \t\t\t= {:.1f} \tN".format(Rz1))
print("Rz2 \t\t\t= {:.1f} \tN".format(Rz2))
print("Rz3 \t\t\t= {:.1f} \tN".format(Rz3))
#print("u_p0 \t\t\t= {:.1f} \tm m^-1".format(Cu_p0/(E*Izz_airfoil)))
#print("u0 \t\t\t= {:.1f} \tm".format(Cu0/(E*Izz_airfoil)))
#print("v_p0 \t\t\t= {:.1f} \tm m^-1".format(Cv_p0/(E*Iyy_airfoil)))
#print("v0 \t\t\t= {:.1f} \tm".format(Cv0/(E*Iyy_airfoil)))
#print("theta0 \t\t\t= {:.1f} \trad".format(Ctheta0/(G*J)))
print("P \t\t\t= {:.1f} \tN".format(Pz_I*m.cos(phi)-Py_I*m.sin(phi)))
print("Py_I \t\t\t= {:.1f} \tN".format(Py_I))
print("Pz_I \t\t\t= {:.1f} \tN".format(Pz_I))

print("\n=========== Stress Statistics: ===========")
print("Max direct stress \t= {:.2f}\tMPa\t(Absolute Stress)".format(sig_xx[idx_sigxx][0]/1e6)) 
print("\t\tat ({:.3f}m, {:.3f}m, {:.3f}m)".format(x[idx_sigxx[0][0]], Z[idx_sigxx[1][0]], Y[idx_sigxx[1][0]]))
print("\t\t(Aileron {:.2f}%, Chord Length {:.2f}%, Chord Half Height {:.2f}%)".format(x[idx_sigxx[0][0]]/l_a*100, Z[idx_sigxx[1][0]]/C_a*100, Y[idx_sigxx[1][0]]/(h_a/2)*100))

print("Max shear stress \t= {:.2f}\tMPa\t(Absolute Stress)".format(tau_yz[idx_tauyz][0]/1e6)) 
print("\t\tat ({:.3f}m, {:.3f}m, {:.3f}m)".format(x[idx_tauyz[0][0]], Z[idx_tauyz[1][0]], Y[idx_tauyz[1][0]]))
print("\t\t(Aileron {:.2f}%, Chord Length {:.2f}%, Chord Half Height {:.2f}%)".format(x[idx_tauyz[0][0]]/l_a*100, Z[idx_tauyz[1][0]]/C_a*100, Y[idx_tauyz[1][0]]/(h_a/2)*100))

print("Max von Mises stress \t= {:.2f}\tMPa\t(Absolute Stress)".format(sig_vm[idx_sigvm][0]/1e6)) 
print("\t\tat ({:.3f}m, {:.3f}m, {:.3f}m)".format(x[idx_sigvm[0][0]], Z[idx_sigvm[1][0]], Y[idx_sigvm[1][0]]))
print("\t\t(Aileron {:.2f}%, Chord Length {:.2f}%, Chord Half Height {:.2f}%)".format(x[idx_sigvm[0][0]]/l_a*100, Z[idx_sigvm[1][0]]/C_a*100, Y[idx_sigvm[1][0]]/(h_a/2)*100))

print("\n=========== Timer Statistics: ===========")
print("Execution Time \t\t= {:.3f} \ts".format(t_se-t_ss))
if export:
    print("Export Time \t\t= {:.3f} \ts".format(t_ee-t_es))

## =========== User Graphs: ===========
## Define general plot parameters for all future plots.
SMALL_SIZE  = textsize[0]
MEDIUM_SIZE = textsize[1]
BIGGER_SIZE = textsize[2]

plt.style.use('grayscale')
plt.rc('font', size=MEDIUM_SIZE, family='serif')    ## controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)                ## fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)                ## fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)               ## fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)               ## fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)               ## legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)             ## fontsize of the figure title
plt.rc('text', usetex=False)
matplotlib.rcParams['lines.linewidth']  = 1.5
matplotlib.rcParams['figure.facecolor'] = 'white'
matplotlib.rcParams['axes.facecolor']   = 'white'
matplotlib.rcParams["legend.fancybox"]  = False

plt.close("all") ## All the export graphs need to be closed

fig1,ax1 = plt.subplots(2,2,squeeze=False,figsize=(16,9))
fig1.canvas.set_window_title('u, dudx, Mz, Ry') 
ax1[0,0].plot(x, u)
ax1[0,0].plot([x_1,x_2,x_3],[u[(x<=x_1)&(x>x_1-dx)],u[(x<=x_2)&(x>x_2-dx)],u[(x<=x_3)&(x>x_3-dx)]], marker = "s", linestyle='none', color='black', markerfacecolor='none', markeredgewidth=2, markersize=6)
ax1[0,0].set_ylabel(r"Deflection in $y^{\prime}$ direction $u\,\,[m]$")
ax1[0,0].set_xlabel(r"Position along Aileron $x\,\,[m]$")
ax1[0,0].set_xlim(0,l_a)
ax1[0,0].grid(True,which="major",color="#999999")
ax1[0,0].grid(True,which="minor",color="#DDDDDD",ls="--")
ax1[0,0].minorticks_on()
ax1[0,0].tick_params(which='major', length=10, width=2, direction='inout')
ax1[0,0].tick_params(which='minor', length=5, width=2, direction='in')

ax1[0,1].plot(x, dudx)
ax1[0,1].plot([x_1,x_2,x_3],[dudx[(x<=x_1)&(x>x_1-dx)],dudx[(x<=x_2)&(x>x_2-dx)],dudx[(x<=x_3)&(x>x_3-dx)]], marker = "s", linestyle='none', color='black', markerfacecolor='none', markeredgewidth=2, markersize=6)
ax1[0,1].set_ylabel(r"Deflection derivative  in $y^{\prime}$ direction $\frac{du}{dx}\,\,[m\,m^{-1}]$")
ax1[0,1].set_xlabel(r"Position along Aileron $x\,\,[m]$")
ax1[0,1].set_xlim(0,l_a)
ax1[0,1].grid(True,which="major",color="#999999")
ax1[0,1].grid(True,which="minor",color="#DDDDDD",ls="--")
ax1[0,1].minorticks_on()
ax1[0,1].tick_params(which='major', length=10, width=2, direction='inout')
ax1[0,1].tick_params(which='minor', length=5, width=2, direction='in')

ax1[1,0].plot(x, Mz)
ax1[1,0].set_ylabel(r"Internal Moment in $z^{\prime}$ direction $M_z^{\prime}\,\,[Nm]$")
ax1[1,0].set_xlabel(r"Position along Aileron $x\,\,[m]$")
ax1[1,0].set_xlim(0,l_a)
ax1[1,0].grid(True,which="major",color="#999999")
ax1[1,0].grid(True,which="minor",color="#DDDDDD",ls="--")
ax1[1,0].minorticks_on()
ax1[1,0].tick_params(which='major', length=10, width=2, direction='inout')
ax1[1,0].tick_params(which='minor', length=5, width=2, direction='in')

ax1[1,1].plot(x, Ry)
ax1[1,1].set_ylabel(r"Shear Force in $y^{\prime}$ direction $R_y^{\prime}\,\,[N]$")
ax1[1,1].set_xlabel(r"Position along Aileron $x\,\,[m]$")
ax1[1,1].set_xlim(0,l_a)
ax1[1,1].grid(True,which="major",color="#999999")
ax1[1,1].grid(True,which="minor",color="#DDDDDD",ls="--")
ax1[1,1].minorticks_on()
ax1[1,1].tick_params(which='major', length=10, width=2, direction='inout')
ax1[1,1].tick_params(which='minor', length=5, width=2, direction='in')

fig1.tight_layout()

fig2,ax2 = plt.subplots(2,2,squeeze=False,figsize=(16,9))
fig2.canvas.set_window_title('v, dvdx, My, Rz') 
ax2[0,0].plot(x, -v)
ax2[0,0].plot([x_1,x_2,x_3],[-v[(x<=x_1)&(x>x_1-dx)],-v[(x<=x_2)&(x>x_2-dx)],-v[(x<=x_3)&(x>x_3-dx)]], marker = "s", linestyle='none', color='black', markerfacecolor='none', markeredgewidth=2, markersize=6)
ax2[0,0].set_ylabel(r"Deflection in $z^{\prime}$ direction $v\,\,[m]$")
ax2[0,0].set_xlabel(r"Position along Aileron $x\,\,[m]$")
ax2[0,0].set_xlim(0,l_a)
ax2[0,0].grid(True,which="major",color="#999999")
ax2[0,0].grid(True,which="minor",color="#DDDDDD",ls="--")
ax2[0,0].minorticks_on()
ax2[0,0].tick_params(which='major', length=10, width=2, direction='inout')
ax2[0,0].tick_params(which='minor', length=5, width=2, direction='in')

ax2[0,1].plot(x, -dvdx)
ax2[0,1].plot([x_1,x_2,x_3],[-dvdx[(x<=x_1)&(x>x_1-dx)],-dvdx[(x<=x_2)&(x>x_2-dx)],-dvdx[(x<=x_3)&(x>x_3-dx)]], marker = "s", linestyle='none', color='black', markerfacecolor='none', markeredgewidth=2, markersize=6)
ax2[0,1].set_ylabel(r"Deflection derivative  in $z^{\prime}$ direction $\frac{dv}{dx}\,\,[m\,m^{-1}]$")
ax2[0,1].set_xlabel(r"Position along Aileron $x\,\,[m]$")
ax2[0,1].set_xlim(0,l_a)
ax2[0,1].grid(True,which="major",color="#999999")
ax2[0,1].grid(True,which="minor",color="#DDDDDD",ls="--")
ax2[0,1].minorticks_on()
ax2[0,1].tick_params(which='major', length=10, width=2, direction='inout')
ax2[0,1].tick_params(which='minor', length=5, width=2, direction='in')

ax2[1,0].plot(x, -My)
ax2[1,0].set_ylabel(r"Internal Moment in $y^{\prime}$ direction $-M_y^{\prime}\,\,[Nm]$")
ax2[1,0].set_xlabel(r"Position along Aileron $x\,\,[m]$")
ax2[1,0].set_xlim(0,l_a)
ax2[1,0].grid(True,which="major",color="#999999")
ax2[1,0].grid(True,which="minor",color="#DDDDDD",ls="--")
ax2[1,0].minorticks_on()
ax2[1,0].tick_params(which='major', length=10, width=2, direction='inout')
ax2[1,0].tick_params(which='minor', length=5, width=2, direction='in')

ax2[1,1].plot(x, -Rz)
ax2[1,1].set_ylabel(r"Shear Force in $-z^{\prime}$ direction $R_z^{\prime}\,\,[N]$")
ax2[1,1].set_xlabel(r"Position along Aileron $x\,\,[m]$")
ax2[1,1].set_xlim(0,l_a)
ax2[1,1].grid(True,which="major",color="#999999")
ax2[1,1].grid(True,which="minor",color="#DDDDDD",ls="--")
ax2[1,1].minorticks_on()
ax2[1,1].tick_params(which='major', length=10, width=2, direction='inout')
ax2[1,1].tick_params(which='minor', length=5, width=2, direction='in')

fig2.tight_layout()

fig3,ax3 = plt.subplots(2,2,squeeze=False,figsize=(16,9))
fig3.canvas.set_window_title('U, dUdx, V, dVdx') 
ax3[0,0].plot(x, U)
ax3[0,0].plot([x_1,x_2,x_3],[d_1,0,d_3], marker = "s", linestyle='none', color='black', markerfacecolor='none', markeredgewidth=2, markersize=6)
ax3[0,0].set_ylabel(r"Deflection in $y$ direction $U\,\,[m]$")
ax3[0,0].set_xlabel(r"Position along Aileron $x\,\,[m]$")
ax3[0,0].set_xlim(0,l_a)
ax3[0,0].grid(True,which="major",color="#999999")
ax3[0,0].grid(True,which="minor",color="#DDDDDD",ls="--")
ax3[0,0].minorticks_on()
ax3[0,0].tick_params(which='major', length=10, width=2, direction='inout')
ax3[0,0].tick_params(which='minor', length=5, width=2, direction='in')

ax3[0,1].plot(x, V)
ax3[0,1].plot([x_1,x_2,x_3],[0,0,0], marker = "s", linestyle='none', color='black', markerfacecolor='none', markeredgewidth=2, markersize=6)
ax3[0,1].set_ylabel(r"Deflection in $z$ direction $V\,\,[m]$")
ax3[0,1].set_xlabel(r"Position along Aileron $x\,\,[m]$")
ax3[0,1].set_xlim(0,l_a)
ax3[0,1].grid(True,which="major",color="#999999")
ax3[0,1].grid(True,which="minor",color="#DDDDDD",ls="--")
ax3[0,1].minorticks_on()
ax3[0,1].tick_params(which='major', length=10, width=2, direction='inout')
ax3[0,1].tick_params(which='minor', length=5, width=2, direction='in')

ax3[1,0].plot(x, dUdx)
ax3[1,0].plot([x_1,x_2,x_3],[dUdx[(x<=x_1)&(x>x_1-dx)],dUdx[(x<=x_2)&(x>x_2-dx)],dUdx[(x<=x_3)&(x>x_3-dx)]], marker = "s", linestyle='none', color='black', markerfacecolor='none', markeredgewidth=2, markersize=6)
ax3[1,0].set_ylabel(r"Deflection derivative in $y$ direction $\frac{dU}{dx}\,\,[m\,m^{-1}]$")
ax3[1,0].set_xlabel(r"Position along Aileron $x\,\,[m]$")
ax3[1,0].set_xlim(0,l_a)
ax3[1,0].grid(True,which="major",color="#999999")
ax3[1,0].grid(True,which="minor",color="#DDDDDD",ls="--")
ax3[1,0].minorticks_on()
ax3[1,0].tick_params(which='major', length=10, width=2, direction='inout')
ax3[1,0].tick_params(which='minor', length=5, width=2, direction='in')

ax3[1,1].plot(x, dVdx)
ax3[1,1].plot([x_1,x_2,x_3],[dVdx[(x<=x_1)&(x>x_1-dx)],dVdx[(x<=x_2)&(x>x_2-dx)],dVdx[(x<=x_3)&(x>x_3-dx)]], marker = "s", linestyle='none', color='black', markerfacecolor='none', markeredgewidth=2, markersize=6)
ax3[1,1].set_ylabel(r"Deflection derivative  in $z$ direction $\frac{dV}{dx}\,\,[m\,m^{-1}]$")
ax3[1,1].set_xlabel(r"Position along Aileron $x\,\,[m]$")
ax3[1,1].set_xlim(0,l_a)
ax3[1,1].grid(True,which="major",color="#999999")
ax3[1,1].grid(True,which="minor",color="#DDDDDD",ls="--")
ax3[1,1].minorticks_on()
ax3[1,1].tick_params(which='major', length=10, width=2, direction='inout')
ax3[1,1].tick_params(which='minor', length=5, width=2, direction='in')

fig3.tight_layout()

fig4,ax4 = plt.subplots(1,2,squeeze=False,figsize=(16,9))
fig4.canvas.set_window_title('theta, Mx') 
ax4[0,0].plot(x, theta)
ax4[0,0].set_ylabel(r"Aileron Twist $\theta\,\,[m]$")
ax4[0,0].set_xlabel(r"Position along Aileron $x\,\,[m]$")
ax4[0,0].set_xlim(0,l_a)
ax4[0,0].grid(True,which="major",color="#999999")
ax4[0,0].grid(True,which="minor",color="#DDDDDD",ls="--")
ax4[0,0].minorticks_on()
ax4[0,0].tick_params(which='major', length=10, width=2, direction='inout')
ax4[0,0].tick_params(which='minor', length=5, width=2, direction='in')

ax4[0,1].plot(x, Mx)
ax4[0,1].set_ylabel(r"Internal Moment in $x$ direction $M_x\,\,[m]$")
ax4[0,1].set_xlabel(r"Position along Aileron $x\,\,[m]$")
ax4[0,1].set_xlim(0,l_a)
ax4[0,1].grid(True,which="major",color="#999999")
ax4[0,1].grid(True,which="minor",color="#DDDDDD",ls="--")
ax4[0,1].minorticks_on()
ax4[0,1].tick_params(which='major', length=10, width=2, direction='inout')
ax4[0,1].tick_params(which='minor', length=5, width=2, direction='in')

fig4.tight_layout()

fig5,ax5 = plt.subplots(1,1,squeeze=False,figsize=(16,9))
fig5.canvas.set_window_title('sigmaxx_X={:.02f}m'.format(x_loc)) 
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
im = ax5[0,0].scatter(Z,Y,c=sig_xx[x<=x_loc][-1],cmap=cmap)
fig5.colorbar(im, ax=ax5[0,0])
ax5[0,0].set_ylabel(r"Height Position from symmetry line $y^{\prime}\,\,[m]$")
ax5[0,0].set_xlabel(r"Position along Chord $z^{\prime}\,\,[m]$")
ax5[0,0].grid(True,which="major",color="#999999")
ax5[0,0].grid(True,which="minor",color="#DDDDDD",ls="--")
ax5[0,0].minorticks_on()
ax5[0,0].tick_params(which='major', length=10, width=2, direction='inout')
ax5[0,0].tick_params(which='minor', length=5, width=2, direction='in')
fig5.tight_layout()

fig6,ax6 = plt.subplots(1,1,squeeze=False,figsize=(16,9))
fig6.canvas.set_window_title('q_X={:.02f}m'.format(x_loc)) 
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
im = ax6[0,0].scatter(Z,Y,c=q[x<=x_loc][-1],cmap=cmap)
fig6.colorbar(im, ax=ax6[0,0])
ax6[0,0].set_ylabel(r"Height Position from symmetry line $y^{\prime}\,\,[m]$")
ax6[0,0].set_xlabel(r"Position along Chord $z^{\prime}\,\,[m]$")
ax6[0,0].grid(True,which="major",color="#999999")
ax6[0,0].grid(True,which="minor",color="#DDDDDD",ls="--")
ax6[0,0].minorticks_on()
ax6[0,0].tick_params(which='major', length=10, width=2, direction='inout')
ax6[0,0].tick_params(which='minor', length=5, width=2, direction='in')
fig6.tight_layout()

fig7,ax7 = plt.subplots(1,1,squeeze=False,figsize=(16,9))
fig7.canvas.set_window_title('sigvm_X={:.02f}m'.format(x_loc)) 
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
im = ax7[0,0].scatter(Z,Y,c=sig_vm[x<=x_loc][-1],cmap=cmap)
fig7.colorbar(im, ax=ax7[0,0])
ax7[0,0].set_ylabel(r"Height Position from symmetry line $y^{\prime}\,\,[m]$")
ax7[0,0].set_xlabel(r"Position along Chord $z^{\prime}\,\,[m]$")
ax7[0,0].grid(True,which="major",color="#999999")
ax7[0,0].grid(True,which="minor",color="#DDDDDD",ls="--")
ax7[0,0].minorticks_on()
ax7[0,0].tick_params(which='major', length=10, width=2, direction='inout')
ax7[0,0].tick_params(which='minor', length=5, width=2, direction='in')
fig7.tight_layout()

plt.show()


