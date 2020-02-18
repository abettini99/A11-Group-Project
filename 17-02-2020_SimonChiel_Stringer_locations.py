import numpy as np
def st_locations(C_a, h_a, n_st):
    s = np.pi*h_a/4 + np.sqrt(C_a**2 - h_a*C_a + h_a**2/2)
    s_semi = np.pi*h_a/4
    s_triangle = np.sqrt(C_a**2 - h_a*C_a + h_a**2/2)
    dsi = s/n_st
    ds = 0
    st = np.zeros((3,11))
    st[0,:6] = np.arange(ds,s,dsi)
    i = 0
    while ds < s_semi:
        theta = (ds/s_semi) * (np.pi/2)
        y = np.sin(theta)*(h_a/2)
        z = (h_a/2) - np.cos(theta)*(h_a/2)
        st[1:,i] = y,z
        ds += dsi
        i += 1

    i = 1
    ds = ds - s_semi
    while i < 6:
        alpha = np.arccos((C_a - (h_a/2))/s_triangle)
        y = np.sin(alpha) * (s_triangle - ds)
        z = C_a - np.cos(alpha) * (s_triangle - ds)
        st[1:,i] = y,z
        ds += dsi
        i += 1
      
    while i < 11:
        st[1:,i] = -st[1:,-i]
        i +=1
    
    return st
n_st = 6
C_a = 0.515
h_a = 0.248

print(st_locations(C_a,h_a,n_st))
