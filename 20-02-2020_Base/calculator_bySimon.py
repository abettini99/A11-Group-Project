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
dx      = 1e-3  ## step size                                    [m]

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

## =========== Additional parameters
x_I = x_2-x_a/2
x_II = x_2+x_a/2
Py_II = -P*np.sin(phi)
Pz_II = P*np.cos(phi)




## =========== Import Aero Data: ===========
AeroDataLoad    = np.loadtxt("aerodynamicloaddo228.dat", delimiter=",")     ##[kN m^-1]
theta_zi        = np.arange(1,82+1,1)/81*m.pi
theta_xi        = np.arange(1,42+1,1)/41*m.pi
AeroDataX       = C_a - 0.5*( 0.5*C_a*(1-np.cos(theta_zi[:-1])) + 0.5*C_a/2*(1-np.cos(theta_zi[1:])) )
AeroDataZ       = 0.5*( 0.5*l_a*(1-np.cos(theta_xi[:-1])) + 0.5*l_a/2*(1-np.cos(theta_xi[1:])) )

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

print("=========== Statistics: ===========")
print("Total Area \t\t= {:.4g} \tm^2".format(A_circ+A_spar+2*A_tail + n_st*A_stif))
print("Airfoil Centroid \t= {:4f} \tm".format(C_airfoil))
print("Cross Section Izz \t= {:4g} \tm^4".format(Izz_airfoil))
print("Cross Section Iyy \t= {:4g} \tm^4".format(Iyy_airfoil))
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
#plt.show()
#



# Part by Simon: Matrix A_big to solve for the 13 unknowns, x_big, with 13 compatibility equations
# x_big: [C0, C1, C2, C3, C4, Ry_1, Ry_2, Ry_3, Rz_1, Rz_2, Rz_3, Py_I, Pz_I] These are all in the tilted coordinate frame, so actually x => x'

# !!!!! REMOVE THIS SOON WHEN INTEGRATION IS LOOKED AT
q = np.array([1])
qm = np.array([1])
laa = np.array([0])
x = 1
x11 = np.array([0])
x22 = np.array([0])
x33 = np.array([0])
xII = np.array([0]) #This is representing integration up to x_I!!

#I need the following variables:
Lab_I = 1       # Length of the semi-circle [m]
Lba_I = 1       # Length of the spar
Lab_II = Lba_I  #
Lba_II = 1      # Total length of both straight parts of aileron
Am_I = 1        # Cell I enclosed area [m^2]
Am_II = 1       # Cell II enclosed area
eta = 1         # Distance from LE to shear center

#Redefining variables
Iyy = Iyy_airfoil
Izz = Izz_airfoil

#Defining matrices for torsion of multicell structure
A_shearflow = np.array([
    [1/(2*Am_I*G)*(Lab_I/t_sk+Lba_I/t_sp)-1/(2*Am_II*G)*(Lab_II/t_sp),1/(2*Am_I*G)*(Lba_I/t_sp)-1/(2*Am_II*G)*(Lab_II/t_sp+Lba_I/t_sk)],
    [2*Am_I,2*Am_II]
    ])

B_twist = np.array([
    [1/(2*Am_I*G)*(Lab_I/t_sk+Lba_I/t_sp),-1/(2*Am_I*G)*(Lba_I/t_sp)]
    ])

C_twist = np.dot(B_twist,np.linalg.inv(A_shearflow))



A_big = np.array([
    [0,0,0,0,0,0,0,0,0,0,0,1,np.tan(phi)],      # [1]
    [0,0,0,0,0,1,1,1,0,0,0,1,0],                # [2]
    [0,0,0,0,0,0,0,0,1,1,1,0,1],                # [3]
    [0,0,0,0,0,0,0,0,0,0,0,h_a/2,h_a/2],                # [4]
    [0,0,0,0,0,0,0,0,(l_a-x_1),(l_a-x_2),(l_a-x_3),(l_a-x_I),0],                # [5]
    [0,0,0,0,0,-(l_a-x_1),-(l_a-x_2),-(l_a-x_3),0,0,0,-(l_a-x_I),0],                # [6]
    [np.cos(phi)*-1/(E*Izz)*x_1  ,  np.cos(phi)*-1/(E*Izz)  ,  np.sin(phi)*-1/(E*Iyy)*x_1  ,  np.sin(phi)*-1/(E*Iyy)  ,  np.cos(phi)*(eta-h_a/2)*C_twist@np.array([[0],[1]])  ,  np.cos(phi)*-1/(E*Izz)/6*(x_1-x_1)**3 + np.cos(phi)*(eta-h_a/2)*C_twist@np.array([[0],[1]])*(eta-h_a/2)*-(x_1-x_1)  ,  np.cos(phi)*-1/(E*Izz)/6*(x_2-x_2)**3 + np.cos(phi)*(eta-h_a/2)*C_twist@np.array([[0],[1]])*(eta-h_a/2)*-(x_2-x_2)  ,  np.cos(phi)*-1/(E*Izz)/6*(x_3-x_3)**3 + np.cos(phi)*(eta-h_a/2)*C_twist@np.array([[0],[1]])*(eta-h_a/2)*-(x_3-x_3)  ,  np.sin(phi)*-1/(E*Iyy)*-1/6*(x_1-x_1)**3  ,  np.sin(phi)*-1/(E*Iyy)*-1/6*(x_2-x_2)**3  ,  np.sin(phi)*-1/(E*Iyy)*-1/6*(x_3-x_3)**3  ,  np.cos(phi)*-1/(E*Izz)/6*(x_I-x_I)**3 + np.cos(phi)*(eta-h_a/2)*C_twist@np.array([[0],[1]])*(-h_a/2*(x_I-x_I)+(eta-h_a/2)*-(x_I-x_I))  ,  np.cos(phi)*(eta-h_a/2)*C_twist@np.array([[0],[1]])*(eta-h_a/2)*-h_a/2*(x_I-x_I)] + np.sin(phi)*-1/(E*Iyy)*-1/6*(x_I-x_I)**3,                # [7]
    [-np.sin(phi)*-1/(E*Izz)*x_1  ,  -np.sin(phi)*-1/(E*Izz)  ,  np.cos(phi)*-1/(E*Iyy)*x_1  ,  np.cos(phi)*-1/(E*Iyy)  ,  -np.sin(phi)*(eta-h_a/2)*C_twist@np.array([[0],[1]])  ,  -np.sin(phi)*-1/(E*Izz)/6*(x_1-x_1)**3 + -np.sin(phi)*(eta-h_a/2)*C_twist@np.array([[0],[1]])*(eta-h_a/2)*-(x_1-x_1)  ,  -np.sin(phi)*-1/(E*Izz)/6*(x_2-x_2)**3 + -np.sin(phi)*(eta-h_a/2)*C_twist@np.array([[0],[1]])*(eta-h_a/2)*-(x_2-x_2)  ,  -np.sin(phi)*-1/(E*Izz)/6*(x_3-x_3)**3 + -np.sin(phi)*(eta-h_a/2)*C_twist@np.array([[0],[1]])*(eta-h_a/2)*-(x_3-x_3)  ,  np.cos(phi)*-1/(E*Iyy)*-1/6*(x_1-x_1)**3  ,  np.cos(phi)*-1/(E*Iyy)*-1/6*(x_2-x_2)**3  ,  np.cos(phi)*-1/(E*Iyy)*-1/6*(x_3-x_3)**3  ,  -np.sin(phi)*-1/(E*Izz)/6*(x_I-x_I)**3 + -np.sin(phi)*(eta-h_a/2)*C_twist@np.array([[0],[1]])*(-h_a/2*(x_I-x_I)+(eta-h_a/2)*-(x_I-x_I))  ,  -np.sin(phi)*(eta-h_a/2)*C_twist@np.array([[0],[1]])*(eta-h_a/2)*-h_a/2*(x_I-x_I)] + np.cos(phi)*-1/(E*Iyy)*-1/6*(x_I-x_I)**3,                # [8]
    [np.cos(phi)*-1/(E*Izz)*x_2  ,  np.cos(phi)*-1/(E*Izz)  ,  np.sin(phi)*-1/(E*Iyy)*x_2  ,  np.sin(phi)*-1/(E*Iyy)  ,  np.cos(phi)*(eta-h_a/2)*C_twist@np.array([[0],[1]])  ,  np.cos(phi)*-1/(E*Izz)/6*(x_2-x_1)**3 + np.cos(phi)*(eta-h_a/2)*C_twist@np.array([[0],[1]])*(eta-h_a/2)*-(x_2-x_1)  ,  np.cos(phi)*-1/(E*Izz)/6*(x_2-x_2)**3 + np.cos(phi)*(eta-h_a/2)*C_twist@np.array([[0],[1]])*(eta-h_a/2)*-(x_2-x_2)  ,  np.cos(phi)*-1/(E*Izz)/6*(x_3-x_3)**3 + np.cos(phi)*(eta-h_a/2)*C_twist@np.array([[0],[1]])*(eta-h_a/2)*-(x_3-x_3)  ,  np.sin(phi)*-1/(E*Iyy)*-1/6*(x_2-x_1)**3  ,  np.sin(phi)*-1/(E*Iyy)*-1/6*(x_2-x_2)**3  ,  np.sin(phi)*-1/(E*Iyy)*-1/6*(x_3-x_3)**3  ,  np.cos(phi)*-1/(E*Izz)/6*(x_2-x_I)**3 + np.cos(phi)*(eta-h_a/2)*C_twist@np.array([[0],[1]])*(-h_a/2*(x_2-x_I)+(eta-h_a/2)*-(x_2-x_I))  ,  np.cos(phi)*(eta-h_a/2)*C_twist@np.array([[0],[1]])*(eta-h_a/2)*-h_a/2*(x_2-x_I)] + np.sin(phi)*-1/(E*Iyy)*-1/6*(x_2-x_I)**3,                # [9]
    [-np.sin(phi)*-1/(E*Izz)*x_2  ,  -np.sin(phi)*-1/(E*Izz)  ,  np.cos(phi)*-1/(E*Iyy)*x_2  ,  np.cos(phi)*-1/(E*Iyy)  ,  -np.sin(phi)*(eta-h_a/2)*C_twist@np.array([[0],[1]])  ,  -np.sin(phi)*-1/(E*Izz)/6*(x_2-x_1)**3 + -np.sin(phi)*(eta-h_a/2)*C_twist@np.array([[0],[1]])*(eta-h_a/2)*-(x_2-x_1)  ,  -np.sin(phi)*-1/(E*Izz)/6*(x_2-x_2)**3 + -np.sin(phi)*(eta-h_a/2)*C_twist@np.array([[0],[1]])*(eta-h_a/2)*-(x_2-x_2)  ,  -np.sin(phi)*-1/(E*Izz)/6*(x_3-x_3)**3 + -np.sin(phi)*(eta-h_a/2)*C_twist@np.array([[0],[1]])*(eta-h_a/2)*-(x_3-x_3)  ,  np.cos(phi)*-1/(E*Iyy)*-1/6*(x_2-x_1)**3  ,  np.cos(phi)*-1/(E*Iyy)*-1/6*(x_2-x_2)**3  ,  np.cos(phi)*-1/(E*Iyy)*-1/6*(x_3-x_3)**3  ,  -np.sin(phi)*-1/(E*Izz)/6*(x_2-x_I)**3 + -np.sin(phi)*(eta-h_a/2)*C_twist@np.array([[0],[1]])*(-h_a/2*(x_2-x_I)+(eta-h_a/2)*-(x_2-x_I))  ,  -np.sin(phi)*(eta-h_a/2)*C_twist@np.array([[0],[1]])*(eta-h_a/2)*-h_a/2*(x_2-x_I)] + np.cos(phi)*-1/(E*Iyy)*-1/6*(x_2-x_I)**3,                # [10]
    [np.cos(phi)*-1/(E*Izz)*x_3  ,  np.cos(phi)*-1/(E*Izz)  ,  np.sin(phi)*-1/(E*Iyy)*x_3  ,  np.sin(phi)*-1/(E*Iyy)  ,  np.cos(phi)*(eta-h_a/2)*C_twist@np.array([[0],[1]])  ,  np.cos(phi)*-1/(E*Izz)/6*(x_3-x_1)**3 + np.cos(phi)*(eta-h_a/2)*C_twist@np.array([[0],[1]])*(eta-h_a/2)*-(x_3-x_1)  ,  np.cos(phi)*-1/(E*Izz)/6*(x_3-x_2)**3 + np.cos(phi)*(eta-h_a/2)*C_twist@np.array([[0],[1]])*(eta-h_a/2)*-(x_3-x_2)  ,  np.cos(phi)*-1/(E*Izz)/6*(x_3-x_3)**3 + np.cos(phi)*(eta-h_a/2)*C_twist@np.array([[0],[1]])*(eta-h_a/2)*-(x_3-x_3)  ,  np.sin(phi)*-1/(E*Iyy)*-1/6*(x_3-x_1)**3  ,  np.sin(phi)*-1/(E*Iyy)*-1/6*(x_3-x_2)**3  ,  np.sin(phi)*-1/(E*Iyy)*-1/6*(x_3-x_3)**3  ,  np.cos(phi)*-1/(E*Izz)/6*(x_3-x_I)**3 + np.cos(phi)*(eta-h_a/2)*C_twist@np.array([[0],[1]])*(-h_a/2*(x_3-x_I)+(eta-h_a/2)*-(x_3-x_I))  ,  np.cos(phi)*(eta-h_a/2)*C_twist@np.array([[0],[1]])*(eta-h_a/2)*-h_a/2*(x_3-x_I)] + np.sin(phi)*-1/(E*Iyy)*-1/6*(x_3-x_I)**3,                # [11]
    [-np.sin(phi)*-1/(E*Izz)*x_3  ,  -np.sin(phi)*-1/(E*Izz)  ,  np.cos(phi)*-1/(E*Iyy)*x_3  ,  np.cos(phi)*-1/(E*Iyy)  ,  -np.sin(phi)*(eta-h_a/2)*C_twist@np.array([[0],[1]])  ,  -np.sin(phi)*-1/(E*Izz)/6*(x_3-x_1)**3 + -np.sin(phi)*(eta-h_a/2)*C_twist@np.array([[0],[1]])*(eta-h_a/2)*-(x_3-x_1)  ,  -np.sin(phi)*-1/(E*Izz)/6*(x_3-x_2)**3 + -np.sin(phi)*(eta-h_a/2)*C_twist@np.array([[0],[1]])*(eta-h_a/2)*-(x_3-x_2)  ,  -np.sin(phi)*-1/(E*Izz)/6*(x_3-x_3)**3 + -np.sin(phi)*(eta-h_a/2)*C_twist@np.array([[0],[1]])*(eta-h_a/2)*-(x_3-x_3)  ,  np.cos(phi)*-1/(E*Iyy)*-1/6*(x_3-x_1)**3  ,  np.cos(phi)*-1/(E*Iyy)*-1/6*(x_3-x_2)**3  ,  np.cos(phi)*-1/(E*Iyy)*-1/6*(x_3-x_3)**3  ,  -np.sin(phi)*-1/(E*Izz)/6*(x_3-x_I)**3 + -np.sin(phi)*(eta-h_a/2)*C_twist@np.array([[0],[1]])*(-h_a/2*(x_3-x_I)+(eta-h_a/2)*-(x_3-x_I))  ,  -np.sin(phi)*(eta-h_a/2)*C_twist@np.array([[0],[1]])*(eta-h_a/2)*-h_a/2*(x_3-x_I)] + np.cos(phi)*-1/(E*Iyy)*-1/6*(x_3-x_I)**3,                # [12]
    [-np.sin(phi)*-1/(E*Izz)*x_I  ,  -np.sin(phi)*-1/(E*Izz)  ,  np.cos(phi)*-1/(E*Iyy)*x_I  ,  np.cos(phi)*-1/(E*Iyy)  ,  -np.sin(phi)*(eta)*C_twist@np.array([[0],[1]]) + np.cos(phi)*(h_a/2)*C_twist@np.array([[0],[1]])  ,  -np.sin(phi)*-1/(E*Izz)/6*(x_I-x_1)**3 + -np.sin(phi)*(eta)*C_twist@np.array([[0],[1]])*(eta-h_a/2)*-(x_I-x_1) + np.cos(phi)*(h_a/2)*C_twist@np.array([[0],[1]])*(eta-h_a/2)*-(x_I-x_1)  ,  -np.sin(phi)*-1/(E*Izz)/6*(x_2-x_2)**3 + -np.sin(phi)*(eta)*C_twist@np.array([[0],[1]])*(eta-h_a/2)*-(x_2-x_2) + np.cos(phi)*(h_a/2)*C_twist@np.array([[0],[1]])*(eta-h_a/2)*-(x_2-x_2)  ,  -np.sin(phi)*-1/(E*Izz)/6*(x_3-x_3)**3 + -np.sin(phi)*(eta)*C_twist@np.array([[0],[1]])*(eta-h_a/2)*-(x_3-x_3)+ np.cos(phi)*(h_a/2)*C_twist@np.array([[0],[1]])*(eta-h_a/2)*-(x_3-x_3)  ,  np.cos(phi)*-1/(E*Iyy)*-1/6*(x_I-x_1)**3  ,  np.cos(phi)*-1/(E*Iyy)*-1/6*(x_2-x_2)**3  ,  np.cos(phi)*-1/(E*Iyy)*-1/6*(x_3-x_3)**3  ,  -np.sin(phi)*-1/(E*Izz)/6*(x_I-x_I)**3 + -np.sin(phi)*(eta)*C_twist@np.array([[0],[1]])*(-h_a/2*(x_I-x_I)+(eta-h_a/2)*-(x_I-x_I)) + np.cos(phi)*(h_a/2)*C_twist@np.array([[0],[1]])*(-h_a/2*(x_I-x_I)+(eta-h_a/2)*-(x_I-x_I))  ,  -np.sin(phi)*(eta)*C_twist@np.array([[0],[1]])*(eta-h_a/2)*-h_a/2*(x_I-x_I) + np.cos(phi)*(h_a/2)*C_twist@np.array([[0],[1]])*(eta-h_a/2)*-h_a/2*(x_I-x_I)] + np.cos(phi)*-1/(E*Iyy)*-1/6*(x_I-x_I)**3,                 # [13]
    ])

RHS_big = np.array([[0],
                [integrator_simp(q, laa)[-1]]-Py_II,
                [-Pz_II],
                [-integrator_simp(qm, laa)[-1]-(C_a/4-h_a/2)*integrator_simp(q, laa)[-1]-h_a/2*(Py_II+Pz_II)],
                [-(l_a-x_II)*Pz_II],
                [-l_a*integrator_simp(q, laa)[-1]+integrator_simp(q*x, laa)[-1]+Py_II*(l_a-x_II)],
                [d_1+np.cos(phi)*-1/(E*Iyy)*(1/6*x_1**3*integrator_simp(q, x11)[-1]-1/2*x_1**2*integrator_simp(q*x, x11)[-1]-1/6*(x_II-x_II)**3*Py_II) + np.cos(phi)*(eta-h_a/2)*C_twist@np.array([[0],[1]])*(x_1*integrator_simp(qm,x11)[1]+x_1*integrator_simp(q,x11)[1]*(C_a/4-h_a/2)+(Py_II+Pz_II)*h_a/2*(x_II-x_II) + (eta-h_a/2)*(-x_1*integrator_simp(q,x11)[1]+Py_II*(x_II-x_II))) + np.sin(phi)*-1/(E*Iyy)*Pz_II/6*(x_II-x_II)**3],
                [-np.sin(phi)*-1/(E*Iyy)*(1/6*x_1**3*integrator_simp(q, x11)[-1]-1/2*x_1**2*integrator_simp(q*x, x11)[-1]-1/6*(x_II-x_II)**3*Py_II) + -np.sin(phi)*(eta-h_a/2)*C_twist@np.array([[0],[1]])*(x_1*integrator_simp(qm,x11)[1]+x_1*integrator_simp(q,x11)[1]*(C_a/4-h_a/2)+(Py_II+Pz_II)*h_a/2*(x_II-x_II) + (eta-h_a/2)*(-x_1*integrator_simp(q,x11)[1]+Py_II*(x_II-x_II))) + np.cos(phi)*-1/(E*Iyy)*Pz_II/6*(x_II-x_II)**3],
                [np.cos(phi)*-1/(E*Iyy)*(1/6*x_2**3*integrator_simp(q, x22)[-1]-1/2*x_2**2*integrator_simp(q*x, x22)[-1]-1/6*(x_II-x_II)**3*Py_II) + np.cos(phi)*(eta-h_a/2)*C_twist@np.array([[0],[1]])*(x_2*integrator_simp(qm,x22)[1]+x_2*integrator_simp(q,x22)[1]*(C_a/4-h_a/2)+(Py_II+Pz_II)*h_a/2*(x_II-x_II) + (eta-h_a/2)*(-x_2*integrator_simp(q,x22)[1]+Py_II*(x_II-x_II))) + np.sin(phi)*-1/(E*Iyy)*Pz_II/6*(x_II-x_II)**3],
                [-np.sin(phi)*-1/(E*Iyy)*(1/6*x_2**3*integrator_simp(q, x22)[-1]-1/2*x_2**2*integrator_simp(q*x, x22)[-1]-1/6*(x_II-x_II)**3*Py_II) + -np.sin(phi)*(eta-h_a/2)*C_twist@np.array([[0],[1]])*(x_2*integrator_simp(qm,x22)[1]+x_2*integrator_simp(q,x22)[1]*(C_a/4-h_a/2)+(Py_II+Pz_II)*h_a/2*(x_II-x_II) + (eta-h_a/2)*(-x_2*integrator_simp(q,x22)[1]+Py_II*(x_II-x_II))) + np.cos(phi)*-1/(E*Iyy)*Pz_II/6*(x_II-x_II)**3],
                [d_3+np.cos(phi)*-1/(E*Iyy)*(1/6*x_3**3*integrator_simp(q, x33)[-1]-1/2*x_3**2*integrator_simp(q*x, x33)[-1]-1/6*(x_3-x_II)**3*Py_II) + np.cos(phi)*(eta-h_a/2)*C_twist@np.array([[0],[1]])*(x_3*integrator_simp(qm,x33)[1]+x_3*integrator_simp(q,x33)[1]*(C_a/4-h_a/2)+(Py_II+Pz_II)*h_a/2*(x_3-x_II) + (eta-h_a/2)*(-x_3*integrator_simp(q,x33)[1]+Py_II*(x_3-x_II))) + np.sin(phi)*-1/(E*Iyy)*Pz_II/6*(x_3-x_II)**3],
                [-np.sin(phi)*-1/(E*Iyy)*(1/6*x_3**3*integrator_simp(q, x33)[-1]-1/2*x_3**2*integrator_simp(q*x, x33)[-1]-1/6*(x_3-x_II)**3*Py_II) + -np.sin(phi)*(eta-h_a/2)*C_twist@np.array([[0],[1]])*(x_3*integrator_simp(qm,x33)[1]+x_3*integrator_simp(q,x33)[1]*(C_a/4-h_a/2)+(Py_II+Pz_II)*h_a/2*(x_3-x_II) + (eta-h_a/2)*(-x_3*integrator_simp(q,x33)[1]+Py_II*(x_3-x_II))) + np.cos(phi)*-1/(E*Iyy)*Pz_II/6*(x_3-x_II)**3],
                [-np.sin(phi)*-1/(E*Iyy)*(1/6*x_I**3*integrator_simp(q, xII)[-1]-1/2*x_I**2*integrator_simp(q*x, xII)[-1]-1/6*(x_II-x_II)**3*Py_II) + -np.sin(phi)*(eta)*C_twist@np.array([[0],[1]])*(x_I*integrator_simp(qm,xII)[1]+x_I*integrator_simp(q,xII)[1]*(C_a/4-h_a/2)+(Py_II+Pz_II)*h_a/2*(x_II-x_II) + (eta-h_a/2)*(-x_I*integrator_simp(q,xII)[1]+Py_II*(x_II-x_II))) + np.cos(phi)*-1/(E*Iyy)*Pz_II/6*(x_II-x_II)**3 + np.cos(phi)*(h_a/2)*C_twist@np.array([[0],[1]])*(x_I*integrator_simp(qm,xII)[1]+x_I*integrator_simp(q,xII)[1]*(C_a/4-h_a/2)+(Py_II+Pz_II)*h_a/2*(x_II-x_II) + (eta-h_a/2)*(-x_I*integrator_simp(q,xII)[1]+Py_II*(x_II-x_II)))],
])

print(np.linalg.inv(A_big)@RHS_big)
