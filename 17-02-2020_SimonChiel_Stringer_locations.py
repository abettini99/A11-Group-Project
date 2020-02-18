import numpy as np





def st_locations(C_a, h_a, n_st):
    s = np.pi*h_a/4 + np.sqrt(C_a**2 - h_a*C_a + h_a**2/2)      #total length from LE to TE following the skin
    s_semi = np.pi*h_a/4                                        #total length from LE to spar following the skin
    s_triangle = np.sqrt(C_a**2 - h_a*C_a + h_a**2/2)           #total length from spar to TE following the skin
    dsi = (2*s)/(n_st+1)                                        #stepsize between the stringers
    ds = 0                                                      #starting point of s-location, on the LE

    st = np.zeros((3,n_st))                                     #producing an array of size 3x11
    st[0,:6] = np.arange(ds,s,dsi)                              #first row, up to the sixth column, stringer s-coordinates for the upper half of the cross-section
    st[0,6:] = np.arange(s+dsi,2*s,dsi)                         #first row, from 7th column up to the last, stringer s-coordinates for the lower half of the cross-section

    i = 0                                                       #first step of the iteration

    while ds < s_semi:                                          #calculation of stringer location on the semi-circle
        theta = (ds/s_semi) * (np.pi/2)                         #angle on the z-location of the spar, clockwise from negative z-direction
        y = np.sin(theta)*(h_a/2)                               #y-coordinate of stringer in the upper half of semi-circle
        z = (h_a/2) - np.cos(theta)*(h_a/2)                     #z-coordinate of stringer in the upper half of semi-circle
        st[1:,i] = y,z                                          #addition of coordinates to the array
        ds += dsi
        i += 1

    ds = ds - s_semi                                            #new starting point of s-location, for stringers in the triangular part of the cross-section

    while i <= ((n_st-1)/2):                                    #calculation of stringer location on the triangular part of the cross-section
        alpha = np.arccos((C_a - (h_a/2))/s_triangle)           #angle on the TE, clockwise from negative z-direction
        y = np.sin(alpha) * (s_triangle - ds)                   #y-coordinate of stringer in the upper half of the triangle
        z = C_a - np.cos(alpha) * (s_triangle - ds)             #z-coordinate of stringer in the upper half of the triangle
        st[1:,i] = y,z                                          #addition of coordinates to the array
        ds += dsi
        i += 1
      
    while i < n_st:                                             #mirroring stringer locations upper half to lower half, z-axis is symmetry axis
        st[1,i] = -st[1,-i]                                     #all y-coordinates of the upper half stringers are mirrored and added to the array
        st[2,i] = st[2,-i]                                      #all z-coordinates of the upper half stringers are copied and added to the array
        i +=1
    
    return st                                                   #output is an array of size 3x11, first row are s-coordinates, second row are y-coordinates, third row are z-coordinates
n_st = 11
C_a = 0.515
h_a = 0.248

print(st_locations(C_a,h_a,n_st))
