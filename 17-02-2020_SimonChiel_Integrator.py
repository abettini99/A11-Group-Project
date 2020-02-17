import numpy as np

def integrator_trap(f, x):
    F = np.sum( ((x[1:] - x[:-1])/2) * (f[1:] + f[:-1]) )
    return F 

def integrator_simp(f, x):
    Flst = []
    Flst.append( ((x[2::2] - x[:-2:2])/6) * (f[2::2] + 4*(f[1:-1:2]) + f[:-2:2]) )
    Farr = np.array(Flst)
    F = np.sum(Farr)
    return Farr, F

def integrator_simp1(f, x):
    F = np.sum( ((x[1:] - x[:-1])/6) * (f[1:] + 2*(f[1:]+f[:-1]) + f[:-1]) )
    return F


x = np.arange(0,11,1)
f = x[:]**2
print(f)
print(integrator_simp(f,x))
