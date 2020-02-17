import numpy as np
def st_locations(C_a, h, n_st):
    s = np.pi*h/4 + np.sqrt(C_a**2 - h*C_a + h**2/2)
    s_semi = np.pi*h/4
    s_triangle = np.sqrt(C_a**2 - h*C_a + h**2/2)
    ds = s/n_st
    st = np.zeros((3,5))
    st[0,:] = np.arange(ds,s,ds)
    i = 0
    while ds < s_semi:
        theta = (ds/s_semi) * (np.pi/2)
        y = np.sin(theta)*(h/2)
        z = np.cos(theta)*(h/2)
        st[1:,i] = y,z
        ds += ds
        i += 1

    while ds < s:
        

    

    
    return st
n_st = 6
C_a = 0.515
h = 0.248
#ds, st = st_locations(C_a, h)
print(st_locations(C_a,h,n_st))
